import streamlit as st
import pandas as pd
import base64
from PIL import Image
import io
import json
import re
from collections import Counter
import google.generativeai as genai

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sistematización Biblioteca - OCR Gemini",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 Sistematización Inteligente de Grupos Focales</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>🎯 ¿Cómo funciona?</b>
    Sube una foto de tus notas → Gemini extrae el texto estructurado →
    Edita la tabla → Aplica análisis NLP → Exporta a Excel o JSON.<br><br>
    <b>✅ Usa Google Gemini API — tiene nivel gratuito generoso (15 requests/min, 1500/día).</b>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    api_key = st.text_input(
        "🔑 API Key de Google Gemini",
        type="password",
        help="Obtén una GRATIS en: aistudio.google.com/apikey"
    )
    st.caption("👉 [Obtener API Key gratis](https://aistudio.google.com/apikey)")

    st.markdown("---")
    st.subheader("📋 Segmento Poblacional")
    segmento = st.selectbox(
        "Grupo focal:",
        ["Infancia", "Población General", "Académicos/Investigadores",
         "Artistas/Creadores", "Editores/Escritores", "Bibliotecas Municipales"]
    )

    st.markdown("---")
    st.subheader("🤖 Análisis NLP")
    analisis_tematico  = st.checkbox("Categorización temática automática", value=True)
    extraer_keywords   = st.checkbox("Extraer palabras clave", value=True)
    analizar_prioridad = st.checkbox("Detectar nivel de prioridad", value=True)

if not api_key:
    st.warning("⚠️ Ingresa tu API Key de Google Gemini en la barra lateral para continuar.")
    st.info("👉 Obtén una gratis en: https://aistudio.google.com/apikey (solo necesitas una cuenta de Google)")
    st.stop()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key, default in [
    ('df_resultado', None),
    ('texto_crudo', ""),
    ('df_analizado', None),
    ('mostrar_viz', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# OCR con Gemini
# ─────────────────────────────────────────────
def ocr_with_gemini(image: Image.Image, api_key: str, segmento: str) -> str:
    """Envía la imagen a Gemini y obtiene JSON estructurado."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # Modelo gratuito y rápido

    prompt = f"""Analiza esta imagen de notas de un grupo focal sobre una biblioteca.
Segmento participante: {segmento}

Extrae TODA la información visible y devuelve SOLO un JSON válido con esta estructura exacta,
sin texto adicional antes ni después, sin bloques de código markdown:

{{
    "registros": [
        {{
            "hora": "hora si aparece, si no deja vacío",
            "cita": "texto exacto o resumen de lo que dijeron",
            "tema": "Espacio|Servicio|Barrera|Otro",
            "subtema": "descripción específica del subtema",
            "importancia": "Alta|Media|Baja"
        }}
    ]
}}

Si hay varias ideas o comentarios en la imagen, crea un registro separado por cada uno.
Si la imagen no contiene texto legible, devuelve {{"registros": []}}.
Responde ÚNICAMENTE con el JSON, sin explicaciones adicionales."""

    response = model.generate_content([prompt, image])
    return response.text


def extraer_json(texto: str) -> dict:
    """Extrae el JSON de la respuesta del modelo."""
    try:
        # Limpiar posibles bloques markdown ```json ... ```
        texto_limpio = re.sub(r'```(?:json)?\s*', '', texto).strip()
        texto_limpio = texto_limpio.replace('```', '').strip()

        json_match = re.search(r'\{.*\}', texto_limpio, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"registros": []}

# ─────────────────────────────────────────────
# NLP (Python puro, sin dependencias pesadas)
# ─────────────────────────────────────────────
CATEGORIAS_ESPACIO = [
    'sala', 'espacio', 'lugar', 'ambiente', 'zona', 'área', 'cuarto',
    'iluminación', 'luz', 'ventilación', 'aire', 'ruido', 'silencio',
    'silla', 'mesa', 'escritorio', 'computador', 'equipo', 'mobiliario'
]
CATEGORIAS_SERVICIO = [
    'servicio', 'préstamo', 'consulta', 'asesoría', 'taller', 'actividad',
    'programa', 'evento', 'capacitación', 'formación', 'wifi', 'internet',
    'digital', 'base de datos', 'catálogo', 'web'
]
CATEGORIAS_BARRERA = [
    'horario', 'tiempo', 'lejos', 'dificultad', 'problema', 'falta', 'no hay',
    'caro', 'costo', 'pago', 'limitación', 'restricción', 'barrera'
]
STOPWORDS_ES = set([
    'que', 'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las',
    'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'más',
    'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta',
    'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta',
    'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos',
    'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos',
    'esto', 'antes', 'algunos', 'unos', 'yo', 'otro', 'otras', 'otra',
    'tanto', 'esa', 'estos', 'mucho', 'cual', 'poco', 'ella', 'estas',
    'algo', 'nosotros', 'nada', 'muchos', 'quienes', 'estar', 'como',
    'han', 'fue', 'son', 'ser', 'has', 'era', 'sido', 'está'
])


def clasificar_tema(texto: str) -> str:
    t = texto.lower()
    scores = {
        'Espacio':  sum(1 for p in CATEGORIAS_ESPACIO  if p in t),
        'Servicio': sum(1 for p in CATEGORIAS_SERVICIO if p in t),
        'Barrera':  sum(1 for p in CATEGORIAS_BARRERA  if p in t),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Otro'


def extraer_palabras_clave(texto: str, n: int = 5) -> str:
    palabras = re.findall(r'\b[a-záéíóúüñ]{4,}\b', texto.lower())
    palabras = [p for p in palabras if p not in STOPWORDS_ES]
    freq = Counter(palabras)
    return ', '.join([p for p, _ in freq.most_common(n)])


def detectar_prioridad(texto: str) -> str:
    t = texto.lower()
    alta = ['urgente', 'necesitamos', 'falta', 'importante', 'crítico',
            'todos', 'siempre', 'nunca', 'imprescindible']
    baja = ['quizás', 'tal vez', 'podría', 'sería bueno', 'me gustaría', 'preferiría']
    score_alta = sum(1 for p in alta if p in t)
    score_baja = sum(1 for p in baja if p in t)
    if score_alta > score_baja:
        return 'Alta'
    elif score_baja > score_alta:
        return 'Baja'
    return 'Media'

# ─────────────────────────────────────────────
# 1. Subir imagen
# ─────────────────────────────────────────────
st.markdown('<div class="sub-header">📤 1. Sube tu imagen</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Arrastra o selecciona una foto de tus notas del grupo focal",
    type=['png', 'jpg', 'jpeg'],
    help="Foto de papel, pizarra, cuaderno, etc."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagen subida")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        procesar = st.button("🚀 Procesar con Gemini OCR", type="primary", use_container_width=True)

    with col2:
        st.subheader("Estado del procesamiento")
        st.info("Gemini analizará la imagen y extraerá el texto estructurado automáticamente.")

        if procesar:
            try:
                with st.spinner("🔍 Analizando imagen con Gemini..."):
                    resultado_ocr = ocr_with_gemini(image, api_key, segmento)
                    st.session_state.texto_crudo = resultado_ocr

                    datos_json = extraer_json(resultado_ocr)

                    if datos_json.get('registros'):
                        df = pd.DataFrame(datos_json['registros'])
                    else:
                        # Fallback: tabla vacía editable
                        df = pd.DataFrame({
                            'hora':        [''],
                            'cita':        [resultado_ocr[:300]],
                            'tema':        ['Por clasificar'],
                            'subtema':     [''],
                            'importancia': ['Media']
                        })

                    # Asegurar que existan todas las columnas necesarias
                    for col in ['hora', 'cita', 'tema', 'subtema', 'importancia']:
                        if col not in df.columns:
                            df[col] = ''

                    df['segmento'] = segmento
                    st.session_state.df_resultado = df
                    st.success("✅ ¡Procesamiento completado!")

            except Exception as e:
                err = str(e)
                if 'API_KEY_INVALID' in err or 'API key' in err.lower():
                    st.error("❌ API Key inválida. Verifica tu clave en aistudio.google.com/apikey")
                elif 'quota' in err.lower() or 'rate' in err.lower():
                    st.error("⏳ Límite de uso alcanzado. Espera un minuto e intenta de nuevo.")
                else:
                    st.error(f"Error en el procesamiento: {e}")

# ─────────────────────────────────────────────
# 2. Tabla editable
# ─────────────────────────────────────────────
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado

    st.markdown('<div class="sub-header">📊 2. Tabla Extraída (Editable)</div>', unsafe_allow_html=True)
    st.markdown("Edita directamente las celdas si el OCR necesita correcciones:")

    df_editado = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "hora":        st.column_config.TextColumn("Hora", width="small"),
            "cita":        st.column_config.TextColumn("Cita / Lo que dijeron", width="large"),
            "tema":        st.column_config.SelectboxColumn("Tema", options=["Espacio", "Servicio", "Barrera", "Otro"], width="medium"),
            "subtema":     st.column_config.TextColumn("Subtema específico", width="medium"),
            "importancia": st.column_config.SelectboxColumn("Importancia", options=["Alta", "Media", "Baja"], width="small"),
            "segmento":    st.column_config.TextColumn("Segmento", width="medium", disabled=True),
        }
    )

    # ─────────────────────────────────────────────
    # 3. Análisis NLP
    # ─────────────────────────────────────────────
    st.markdown('<div class="sub-header">🤖 3. Análisis NLP y Sistematización</div>', unsafe_allow_html=True)

    col_nlp1, col_nlp2 = st.columns(2)

    with col_nlp1:
        if st.button("🔍 Aplicar Análisis Automático", type="secondary", use_container_width=True):
            with st.spinner("Analizando con NLP..."):
                df_analizado = df_editado.copy()

                if analisis_tematico:
                    df_analizado['tema_detectado'] = df_analizado['cita'].apply(clasificar_tema)
                if extraer_keywords:
                    df_analizado['palabras_clave'] = df_analizado['cita'].apply(extraer_palabras_clave)
                if analizar_prioridad:
                    df_analizado['prioridad_nlp'] = df_analizado['cita'].apply(detectar_prioridad)

                st.session_state.df_analizado = df_analizado
                st.success("✅ Análisis NLP completado")

    with col_nlp2:
        if st.button("📈 Generar Visualizaciones", use_container_width=True):
            if st.session_state.df_analizado is not None:
                st.session_state.mostrar_viz = True
            else:
                st.warning("Primero aplica el análisis NLP")

    # ─────────────────────────────────────────────
    # 4. Resultados y visualizaciones
    # ─────────────────────────────────────────────
    if st.session_state.df_analizado is not None:
        df_viz = st.session_state.df_analizado

        st.markdown("**Resultado del análisis:**")
        st.dataframe(df_viz, use_container_width=True)

        if st.session_state.mostrar_viz:
            st.markdown('<div class="sub-header">📈 4. Visualizaciones</div>', unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["Distribución de Temas y Prioridades", "Matriz de Consolidación"])

            with tab1:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.subheader("Distribución por Tema")
                    col_tema = 'tema_detectado' if 'tema_detectado' in df_viz.columns else 'tema'
                    st.bar_chart(df_viz[col_tema].value_counts())
                with col_c2:
                    st.subheader("Distribución por Prioridad")
                    col_prio = 'prioridad_nlp' if 'prioridad_nlp' in df_viz.columns else 'importancia'
                    st.bar_chart(df_viz[col_prio].value_counts())

            with tab2:
                if 'tema_detectado' in df_viz.columns and 'prioridad_nlp' in df_viz.columns:
                    matriz = pd.crosstab(
                        df_viz['tema_detectado'],
                        df_viz['prioridad_nlp'],
                        margins=True
                    )
                    st.dataframe(matriz, use_container_width=True)

                    st.markdown("**🎯 Insights detectados:**")
                    tema_frecuente = df_viz['tema_detectado'].mode()[0]
                    prio_dominante = df_viz['prioridad_nlp'].mode()[0]
                    st.markdown(f"""
- **Tema más mencionado:** {tema_frecuente}
- **Nivel de prioridad predominante:** {prio_dominante}
- **Total de registros identificados:** {len(df_viz)}
""")
                    alta_espacio = len(df_viz[
                        (df_viz['tema_detectado'] == 'Espacio') & (df_viz['prioridad_nlp'] == 'Alta')
                    ])
                    if alta_espacio > 0:
                        st.warning(f"⚠️ {alta_espacio} necesidad(es) de ESPACIO con prioridad ALTA detectadas.")
                else:
                    st.info("Aplica el análisis NLP para ver la matriz de consolidación.")

    # ─────────────────────────────────────────────
    # 5. Exportar
    # ─────────────────────────────────────────────
    st.markdown('<div class="sub-header">💾 5. Exportar Resultados</div>', unsafe_allow_html=True)

    df_export = st.session_state.df_analizado if st.session_state.df_analizado is not None else df_editado

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Registros', index=False)

            if st.session_state.df_analizado is not None:
                if 'tema_detectado' in df_export.columns and 'prioridad_nlp' in df_export.columns:
                    resumen = df_export.groupby(['tema_detectado', 'prioridad_nlp']).size().reset_index(name='conteo')
                    resumen.to_excel(writer, sheet_name='Resumen_Temas', index=False)
                if 'palabras_clave' in df_export.columns:
                    todas_kw = ', '.join(df_export['palabras_clave'].dropna())
                    pd.DataFrame({'palabras_clave_consolidadas': [todas_kw]}).to_excel(
                        writer, sheet_name='Keywords', index=False
                    )

        nombre_archivo = f"sistematizacion_{segmento.replace('/', '_')}.xlsx"
        st.download_button(
            label="📥 Descargar Excel completo",
            data=buffer.getvalue(),
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_exp2:
        json_str = df_export.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar JSON",
            data=json_str,
            file_name=f"sistematizacion_{segmento.replace('/', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

# ─────────────────────────────────────────────
# Guía de uso
# ─────────────────────────────────────────────
with st.expander("📖 ¿Cómo usar esta aplicación?"):
    st.markdown("""
### Guía paso a paso

1. **Obtén tu API Key gratis** en [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (solo necesitas una cuenta de Google).
2. **Ingresa la API Key** en la barra lateral.
3. **Selecciona el segmento** poblacional del grupo focal.
4. **Sube la foto** de tus notas (papel, pizarra, cuaderno, etc.).
5. **Haz clic en "Procesar con Gemini OCR"** → Gemini extraerá el texto automáticamente.
6. **Revisa y edita** la tabla generada directamente en las celdas.
7. **Aplica el análisis NLP** para categorizar, extraer palabras clave y detectar prioridades.
8. **Genera visualizaciones** para identificar patrones.
9. **Exporta** el Excel o JSON con el análisis completo.

### Límites gratuitos de Gemini
- 15 requests por minuto
- 1,500 requests por día
- Sin necesidad de tarjeta de crédito

### Tips para mejores resultados
- Usa buena iluminación al tomar la foto
- Foto lo más frontal posible (evita ángulos extremos)
- Si el OCR no detecta bien, edita directamente la tabla
- Procesa un grupo focal a la vez para mantener los segmentos organizados
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Desarrollado para sistematización de diagnóstico participativo · Biblioteca Departamental</div>",
    unsafe_allow_html=True
)
