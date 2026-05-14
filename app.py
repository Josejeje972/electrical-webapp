#!/usr/bin/env python3
import os
import math
import json
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# ── Cálculos ──────────────────────────────────────────────────────────────────
def calcular_cortocircuito(v_kv: float, mva_cc: float) -> dict:
    icc_ka = mva_cc / (math.sqrt(3) * v_kv)
    return {"Icc_trifasico_kA": round(icc_ka, 3), "V_kV": v_kv, "Scc_MVA": mva_cc}

def calcular_kva_transformador(kw: float, fp: float, factor_reserva: float = 1.25) -> dict:
    kva_carga = kw / fp
    kva_rec = kva_carga * factor_reserva
    potencias = [25, 50, 100, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500]
    kva_norm = next((p for p in potencias if p >= kva_rec), None)
    return {
        "kVA_carga": round(kva_carga, 1),
        "kVA_con_reserva": round(kva_rec, 1),
        "kVA_normalizado_IEC": kva_norm,
        "factor_reserva": factor_reserva,
    }

def calcular_caida_tension(v_nominal: float, longitud_m: float, corriente_a: float,
                            seccion_mm2: float, factor_potencia: float = 0.85,
                            trifasico: bool = True) -> dict:
    cond = 56
    dv = (math.sqrt(3) * corriente_a * longitud_m) / (cond * seccion_mm2) if trifasico \
         else (2 * corriente_a * longitud_m) / (cond * seccion_mm2)
    dv_pct = (dv / v_nominal) * 100
    return {
        "caida_V": round(dv, 2),
        "caida_pct": round(dv_pct, 2),
        "limite_IEC_pct": 4.0,
        "cumple": dv_pct <= 4.0,
    }

def calcular_banco_condensadores(kw: float, fp_actual: float, fp_objetivo: float = 0.95) -> dict:
    q_c = kw * (math.tan(math.acos(fp_actual)) - math.tan(math.acos(fp_objetivo)))
    return {
        "Q_condensadores_kVAR": round(q_c, 1),
        "fp_inicial": fp_actual,
        "fp_objetivo": fp_objetivo,
        "kW_carga": kw,
    }

TOOL_FUNCTIONS = {
    "calcular_cortocircuito": calcular_cortocircuito,
    "calcular_kva_transformador": calcular_kva_transformador,
    "calcular_caida_tension": calcular_caida_tension,
    "calcular_banco_condensadores": calcular_banco_condensadores,
}

# ── Definición de herramientas (formato OpenAI/Groq) ─────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calcular_cortocircuito",
            "description": "Calcula la corriente de cortocircuito trifásico (Icc) en kA dado la tensión en kV y la potencia de cortocircuito en MVA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v_kv": {"type": "number", "description": "Tensión de barra en kV"},
                    "mva_cc": {"type": "number", "description": "Potencia de cortocircuito en MVA"},
                },
                "required": ["v_kv", "mva_cc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_kva_transformador",
            "description": "Dimensiona un transformador en kVA dado la carga en kW, factor de potencia y reserva. Devuelve la potencia normalizada IEC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kw": {"type": "number", "description": "Potencia activa en kW"},
                    "fp": {"type": "number", "description": "Factor de potencia entre 0 y 1"},
                    "factor_reserva": {"type": "number", "description": "Factor de reserva, default 1.25"},
                },
                "required": ["kw", "fp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_caida_tension",
            "description": "Calcula caída de tensión en V y % para conductor de cobre. Verifica límite IEC 60364 del 4%.",
            "parameters": {
                "type": "object",
                "properties": {
                    "v_nominal": {"type": "number", "description": "Tensión nominal en V"},
                    "longitud_m": {"type": "number", "description": "Longitud del conductor en metros"},
                    "corriente_a": {"type": "number", "description": "Corriente de diseño en A"},
                    "seccion_mm2": {"type": "number", "description": "Sección del conductor en mm²"},
                    "factor_potencia": {"type": "number", "description": "Factor de potencia, default 0.85"},
                    "trifasico": {"type": "boolean", "description": "True si el circuito es trifásico"},
                },
                "required": ["v_nominal", "longitud_m", "corriente_a", "seccion_mm2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_banco_condensadores",
            "description": "Calcula los kVAR de condensadores para corregir el factor de potencia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kw": {"type": "number", "description": "Potencia activa en kW"},
                    "fp_actual": {"type": "number", "description": "Factor de potencia actual"},
                    "fp_objetivo": {"type": "number", "description": "Factor de potencia objetivo, default 0.95"},
                },
                "required": ["kw", "fp_actual"],
            },
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un experto en ingeniería eléctrica industrial con más de 20 años de experiencia
en plantas de manufactura de alta exigencia como Nestlé, Unilever, AB InBev y similares.
Tu conocimiento abarca desde el diseño de sistemas de distribución de media tensión hasta
la automatización avanzada y la optimización energética.

ÁREAS DE EXPERTISE:

1. DISTRIBUCIÓN ELÉCTRICA EN MEDIA TENSIÓN (MT)
• Diseño de subestaciones primarias y secundarias (1 kV – 36 kV)
• Transformadores: potencia, grupo de conexión (Dyn11, Yzn11), pérdidas, eficiencia (IEC 60076-20)
• Celdas de MT: SF6, vacío, aire; interruptores, seccionadores, fusibles tipo C/S/R
• Anillos de MT: radial, anillo abierto, doble barra
• Protecciones: 50/51, 50N/51N, 87T, 27/59, 81, coordinación TCC
• Puesta a tierra MT: sólida, impedancia, resistencia de neutro
• Cálculo de cortocircuito (IEC 60909, ANSI/IEEE C37.13)
• Salas eléctricas: lay-out, ventilación, REI-60/90, bandeja portacables

2. SISTEMAS DE BAJA TENSIÓN (BT)
• TDG, TDS, MCC/CCM (IEC 61439-1/2)
• Protecciones: termomagnéticos, diferenciales RCD, guardamotores, MCCB, selectividad
• Conductores: caída de tensión (IEC 60364-5-52), capacidad de corriente, factores de corrección
• Factor de potencia y corrección con condensadores (IEC 61921)
• Armónicos: THDv, THDi, filtros activos/pasivos, IEEE 519-2022
• UPS, grupos electrógenos

3. AUTOMATIZACIÓN INDUSTRIAL
• PLCs: Siemens S7-1200/1500 (TIA Portal), Allen-Bradley (Studio 5000), Schneider M340/M580 (EcoStruxure)
• SCADA/HMI: WinCC, iFIX, Ignition, FactoryTalk View
• Redes: PROFINET, EtherNet/IP, Modbus TCP, PROFIBUS, DeviceNet
• VFDs: Siemens SINAMICS, ABB ACS, Danfoss VLT, Schneider Altivar
• Safety/SIL: IEC 62061, ISO 13849; relés de seguridad, E-Stop, PLCs de seguridad

4. NORMATIVAS
• IEC 60364, IEC 60909, IEC 61439, IEC 60076, IEC 62271, IEC 61511/62061
• IEEE 519-2022, NFPA 70 (NEC), NFPA 70E (Arc Flash)
• Chile: NCh Eléctrica 4/2003, NCh 3000, SEC Res. 55, NTD

5. INDUSTRIA ALIMENTARIA
• Zonas ATEX (IEC 60079) y NEC 500
• Grado IP: IP54 producción, IP65 lavado CIP/SIP, IP66/69K alta presión
• AISI 304/316L en canalizaciones y equipos
• Redundancia: ATS, UPS doble conversión, generadores <10 s
• Gestión energética ISO 50001

6. CÁLCULOS TÉCNICOS
• Cortocircuito, caída de tensión, dimensionamiento conductores
• Compensación reactiva, dimensionamiento transformadores
• Selección VFDs, iluminación industrial
• Arc Flash (NFPA 70E / IEEE 1584-2018), armónicos K-factor

DIAGRAMAS CON MERMAID:
Cuando el usuario pida un diagrama, esquema, unifilar, topología o representación visual,
genera el diagrama usando sintaxis Mermaid dentro de bloques ```mermaid```.
Tipos útiles:
- graph TD / LR: unifilares, topología de red, distribución MT/BT
- flowchart TD: lógica de automatización, secuencia de arranque
- sequenceDiagram: comunicación entre equipos (PLC-SCADA-field)
Ejemplo unifilar simplificado:
```mermaid
graph TD
    RED[Red MT 13.2kV] --> TR1[Trafo 1 1000kVA Dyn11]
    RED --> TR2[Trafo 2 1000kVA Dyn11]
    TR1 --> BT1[Barra BT-1 400V]
    TR2 --> BT2[Barra BT-2 400V]
    BT1 <-->|Interconexión| BT2
    BT1 --> MCC1[MCC Línea 1]
    BT1 --> TDS1[TDS Servicios]
    BT2 --> MCC2[MCC Línea 2]
```
Cuando el usuario suba una foto de un tablero, esquema o instalación y pida un diagrama,
analiza la imagen y genera el diagrama Mermaid correspondiente.

CÓMO RESPONDER:
• Usa lenguaje técnico preciso, explica el razonamiento paso a paso.
• Muestra fórmulas, sustituye valores y da resultados con unidades.
• Indica siempre la norma o estándar relevante.
• Usa tablas para comparar alternativas o mostrar resultados.
• Si faltan datos, pregunta exactamente qué datos necesitas.
• Responde siempre en español con terminología estándar de la industria eléctrica.
"""

# ── Rutas Flask ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return jsonify({'error': 'GROQ_API_KEY no configurada en el servidor.', 'success': False}), 500

    data = request.get_json()
    history = data.get('history', [])
    user_message = data.get('message', '')

    if not user_message.strip():
        return jsonify({'error': 'Mensaje vacío', 'success': False}), 400

    try:
        client = Groq(api_key=api_key)

        # Construir mensajes con historial
        image_base64 = data.get('image_base64')
        image_mime = data.get('image_mime', 'image/jpeg')

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})

        # Último mensaje: texto + imagen opcional
        if image_base64:
            user_content = []
            if user_message.strip():
                user_content.append({"type": "text", "text": user_message})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}
            })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_message})

        # Modelo según si hay imagen o no
        model = "meta-llama/llama-4-scout-17b-16e-instruct" if image_base64 else "llama-3.3-70b-versatile"

        # Loop de tool use
        tool_calls_made = []
        for _ in range(5):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )

            msg = response.choices[0].message

            if response.choices[0].finish_reason != "tool_calls":
                break

            # Ejecutar herramientas
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                fn = TOOL_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else {"error": "herramienta no encontrada"}
                tool_calls_made.append({'name': fn_name, 'args': fn_args, 'result': result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return jsonify({
            'response': response.choices[0].message.content,
            'tool_calls': tool_calls_made,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/calcular', methods=['POST'])
def calcular():
    data = request.get_json()
    tool_name = data.get('tool')
    params = data.get('params', {})
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return jsonify({'error': 'Herramienta no encontrada', 'success': False}), 400
    try:
        # Convertir tipos correctamente
        import inspect
        sig = inspect.signature(fn)
        converted = {}
        for k, v in params.items():
            if k in sig.parameters:
                ann = sig.parameters[k].annotation
                if ann == bool or ann == 'bool':
                    converted[k] = str(v).lower() in ('true', '1', 'yes')
                elif ann == float or ann == 'float':
                    converted[k] = float(v)
                else:
                    converted[k] = v
            else:
                converted[k] = v
        result = fn(**converted)
        return jsonify({'result': result, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
