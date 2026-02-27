
import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io
import json
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuración API HuggingFace
API_URL = "https://api-inference.huggingface.co/models/kristaller486/dots.ocr-1.5"

st.set_page_config(
    page_title="Sistematización Biblioteca - OCR API",
    page_icon="📚",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">📚 Sistematización Inteligente de Grupos Focales</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>🎯 ¿Qué hace esta app?</h4>
    <ol>
        <li><strong>Sube una foto</strong> de tus notas de grupo focal (papel, pizarra, etc.)</li>
        <li><strong>OCR Inteligente</strong> con DOTS.OCR-1.5 extrae el texto estructurado</li>
        <li><strong>Visualización</strong> en tabla tipo Excel editable</li>
        <li><strong>NLP/ML</strong> extrae palabras clave, categoriza y sistematiza automáticamente</li>
        <li><strong>Exporta</strong> a Excel con el análisis completo</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Token HuggingFace")
    api_token = st.text_input(
        "🔑 Token de HuggingFace",
        type="password",
        help="Obtén uno gratis en huggingface.co/settings/tokens"
    )
    if api_token:
        st.success("Token configurado ✅")
    else:
        st.info("Ingresa tu token para usar el OCR")

    st.subheader("Segmento Poblacional")
    segmento = st.selectbox(
        "Selecciona el grupo:",
        ["Infancia", "Población General", "Académicos/Investigadores",
         "Artistas/Creadores", "Editores/Escritores", "Bibliotecas Municipales"]
    )

    st.subheader("Análisis NLP")
    analisis_tematico = st.checkbox("Activar categorización temática", value=True)
    extraer_keywords = st.checkbox("Extraer palabras clave", value=True)
    analizar_sentimiento = st.checkbox("Análisis de prioridad (intensidad)", value=True)

# Inicializar session state
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None
if 'texto_crudo' not in st.session_state:
    st.session_state.texto_crudo = ""

def query_ocr_api(image, token):
    """Consulta la API de HuggingFace para OCR"""
    headers = {"Authorization": f"Bearer {token}"}
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    response = requests.post(API_URL, headers=headers, data=img_bytes)

    if response.status_code in (401, 403):
        raise Exception("Token de HuggingFace inválido o sin permisos. Verifica tu token en huggingface.co/settings/tokens.")

    if response.status_code == 429:
        raise Exception("Límite de solicitudes alcanzado. Espera un momento e intenta de nuevo.")

    if response.status_code == 503:
        try:
            error_data = response.json()
            raw_wait = error_data.get("estimated_time", 20)
            wait_time = float(raw_wait) if raw_wait is not None else 20
            raise Exception(f"El modelo está cargando en HuggingFace. Espera ~{wait_time:.0f} segundos e intenta de nuevo.")
        except (json.JSONDecodeError, ValueError):
            raise Exception("El modelo está cargando en HuggingFace. Espera unos segundos e intenta de nuevo.")

    if not response.content:
        raise Exception(f"La API devolvió una respuesta vacía (HTTP {response.status_code}). Intenta de nuevo en unos segundos.")

    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        raise Exception(f"Respuesta inesperada de la API (HTTP {response.status_code}): {response.text[:300]}")

def extraer_json(texto):
    """Extrae el JSON de la respuesta del modelo"""
    try:
        json_match = re.search(r'\{.*\}', texto, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"registros": []}
    except Exception:
        return {"registros": []}

# Funciones de NLP para sistematización
def cargar_nlp():
    """Carga modelo spaCy para español"""
    try:
        import spacy
        return spacy.load("es_core_news_md")
    except Exception:
        st.warning("Modelo spaCy no encontrado. Instalando...")
        import os
        os.system("python -m spacy download es_core_news_md")
        import spacy
        return spacy.load("es_core_news_md")

# Diccionarios de categorización para bibliotecas
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

def clasificar_tema(texto):
    """Clasifica el texto en Espacio, Servicio o Barrera"""
    texto_lower = texto.lower()
    score_espacio = sum(1 for palabra in CATEGORIAS_ESPACIO if palabra in texto_lower)
    score_servicio = sum(1 for palabra in CATEGORIAS_SERVICIO if palabra in texto_lower)
    score_barrera = sum(1 for palabra in CATEGORIAS_BARRERA if palabra in texto_lower)
    scores = {'Espacio': score_espacio, 'Servicio': score_servicio, 'Barrera': score_barrera}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'Otro'

def extraer_palabras_clave(texto, nlp_model, n=5):
    """Extrae las n palabras clave más importantes"""
    doc = nlp_model(texto.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
        and len(token.text) > 3 and token.pos_ in ['NOUN', 'ADJ', 'VERB']
    ]
    freq = Counter(tokens)
    return ', '.join([palabra for palabra, _ in freq.most_common(n)])

def detectar_prioridad(texto):
    """Detecta nivel de prioridad basado en intensidad lingüística"""
    texto_lower = texto.lower()
    alta = ['urgente', 'necesitamos', 'falta', 'importante', 'crítico', 'todos', 'siempre', 'nunca']
    baja = ['quizás', 'tal vez', 'podría', 'sería bueno', 'me gustaría', 'preferiría']
    score_alta = sum(1 for palabra in alta if palabra in texto_lower)
    score_baja = sum(1 for palabra in baja if palabra in texto_lower)
    if score_alta > score_baja:
        return 'Alta'
    elif score_baja > score_alta:
        return 'Baja'
    return 'Media'

def generar_nube_palabras(textos):
    """Genera nube de palabras de los textos"""
    texto_completo = ' '.join(textos).lower()
    stopwords = set(['que', 'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad'])
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap='viridis'
    ).generate(texto_completo)
    return wordcloud

# Área principal de la app
st.markdown('<div class="sub-header">📤 1. Sube tu imagen</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Arrastra o selecciona una foto de tus notas de grupo focal",
    type=['png', 'jpg', 'jpeg'],
    help="Puede ser foto de papel, pizarra, cuaderno, etc."
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagen subida")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        procesar = st.button("🚀 Procesar con OCR", type="primary", use_container_width=True)

    with col2:
        st.subheader("Vista previa")
        st.info("La imagen se procesará con DOTS.OCR-1.5 via HuggingFace API para extraer texto estructurado")

        if procesar:
            if not api_token:
                st.error("⚠️ Ingresa tu token de HuggingFace en la barra lateral para continuar.")
            else:
                try:
                    with st.spinner("🔍 Analizando imagen con IA..."):
                        resultado_api = query_ocr_api(image, api_token)

                        # La API puede devolver lista o dict
                        if isinstance(resultado_api, list) and len(resultado_api) > 0:
                            resultado_ocr = resultado_api[0].get('generated_text', str(resultado_api))
                        elif isinstance(resultado_api, dict):
                            if 'error' in resultado_api:
                                st.warning(f"⚠️ La API respondió con un error: {resultado_api['error']}")
                            resultado_ocr = resultado_api.get('generated_text', str(resultado_api))
                        else:
                            st.warning("⚠️ Respuesta inesperada de la API. Se mostrará el texto crudo.")
                            resultado_ocr = str(resultado_api)

                        st.session_state.texto_crudo = resultado_ocr

                        datos_json = extraer_json(resultado_ocr)

                        if datos_json.get('registros'):
                            df = pd.DataFrame(datos_json['registros'])
                        else:
                            df = pd.DataFrame({
                                'hora': [''],
                                'cita': [resultado_ocr[:500]],
                                'tema': ['Por clasificar'],
                                'subtema': [''],
                                'importancia': ['Media']
                            })

                        df['segmento'] = segmento
                        st.session_state.df_resultado = df
                        st.success("✅ ¡Procesamiento completado!")

                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
                    msg = str(e).lower()
                    if "token" in msg or "permisos" in msg or "inválido" in msg:
                        st.info("💡 Verifica que tu token de HuggingFace sea válido y tenga permisos de lectura.")
                    elif "cargando" in msg:
                        st.info("💡 El modelo tarda unos segundos en arrancar. Haz clic en 'Procesar con OCR' de nuevo.")
                    elif "límite" in msg:
                        st.info("💡 Espera unos minutos antes de volver a intentarlo.")
                    else:
                        st.info("💡 Intenta de nuevo. Si el problema persiste, verifica tu token de HuggingFace o usa una imagen más clara.")

# Mostrar resultados y análisis NLP
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado

    st.markdown('<div class="sub-header">📊 2. Tabla Extraída (Editable)</div>', unsafe_allow_html=True)

    st.markdown("**Edita directamente la tabla si el OCR necesita correcciones:**")
    df_editado = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "hora": st.column_config.TextColumn("Hora", width="small"),
            "cita": st.column_config.TextColumn("Cita/Lo que dijeron", width="large"),
            "tema": st.column_config.SelectboxColumn(
                "Tema",
                options=["Espacio", "Servicio", "Barrera", "Otro"],
                width="medium"
            ),
            "subtema": st.column_config.TextColumn("Subtema específico", width="medium"),
            "importancia": st.column_config.SelectboxColumn(
                "Importancia",
                options=["Alta", "Media", "Baja"],
                width="small"
            ),
            "segmento": st.column_config.TextColumn("Segmento", width="medium", disabled=True)
        }
    )

    st.markdown('<div class="sub-header">🤖 3. Análisis NLP y Sistematización</div>', unsafe_allow_html=True)

    col_nlp1, col_nlp2 = st.columns([1, 1])

    with col_nlp1:
        if st.button("🔍 Aplicar Análisis Automático", type="secondary", use_container_width=True):
            with st.spinner("Analizando con NLP..."):
                nlp_model = cargar_nlp()
                df_analizado = df_editado.copy()

                if analisis_tematico:
                    df_analizado['tema_detectado'] = df_analizado['cita'].apply(clasificar_tema)

                if extraer_keywords:
                    df_analizado['palabras_clave'] = df_analizado['cita'].apply(
                        lambda x: extraer_palabras_clave(x, nlp_model)
                    )

                if analizar_sentimiento:
                    df_analizado['prioridad_nlp'] = df_analizado['cita'].apply(detectar_prioridad)

                st.session_state.df_analizado = df_analizado
                st.success("Análisis NLP completado")

    with col_nlp2:
        if st.button("📈 Generar Visualizaciones", use_container_width=True):
            if 'df_analizado' in st.session_state:
                st.session_state.mostrar_viz = True
            else:
                st.warning("Primero aplica el análisis NLP")

    if 'df_analizado' in st.session_state:
        df_viz = st.session_state.df_analizado

        st.markdown("**Resultado del análisis:**")
        st.dataframe(df_viz, use_container_width=True)

        if st.session_state.get('mostrar_viz', False):
            st.markdown('<div class="sub-header">📈 4. Visualizaciones y Sistematización</div>', unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Distribución de Temas", "Nube de Palabras", "Matriz de Prioridades"])

            with tab1:
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.subheader("Distribución por Tema")
                    tema_col = 'tema_detectado' if 'tema_detectado' in df_viz.columns else 'tema'
                    st.bar_chart(df_viz[tema_col].value_counts())

                with col_chart2:
                    st.subheader("Distribución por Prioridad")
                    prio_col = 'prioridad_nlp' if 'prioridad_nlp' in df_viz.columns else 'importancia'
                    st.bar_chart(df_viz[prio_col].value_counts())

            with tab2:
                st.subheader("Nube de Palabras Clave")
                if 'palabras_clave' in df_viz.columns:
                    textos = df_viz['cita'].tolist()
                    wordcloud = generar_nube_palabras(textos)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Activa 'Extraer palabras clave' en la configuración")

            with tab3:
                st.subheader("Matriz de Consolidación")
                if 'tema_detectado' in df_viz.columns and 'prioridad_nlp' in df_viz.columns:
                    matriz = pd.crosstab(
                        df_viz['tema_detectado'],
                        df_viz['prioridad_nlp'],
                        margins=True
                    )
                    st.dataframe(matriz, use_container_width=True)

                    st.markdown("**🎯 Insights detectados:**")
                    tema_mas_frecuente = df_viz['tema_detectado'].mode()[0]
                    prioridad_dominante = df_viz['prioridad_nlp'].mode()[0]

                    st.markdown(f"""
                    - **Tema más mencionado:** {tema_mas_frecuente}
                    - **Nivel de prioridad predominante:** {prioridad_dominante}
                    - **Total de necesidades identificadas:** {len(df_viz)}
                    """)

                    alta_prio_espacio = len(df_viz[(df_viz['tema_detectado'] == 'Espacio') & (df_viz['prioridad_nlp'] == 'Alta')])
                    if alta_prio_espacio > 0:
                        st.warning(f"⚠️ Se detectaron {alta_prio_espacio} necesidades de ESPACIO con prioridad ALTA. Considerar intervención estructural.")

    # Exportación
    st.markdown('<div class="sub-header">💾 5. Exportar Resultados</div>', unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        df_export = st.session_state.get('df_analizado', df_editado)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Registros', index=False)

            if 'df_analizado' in st.session_state:
                if 'tema_detectado' in df_export.columns and 'prioridad_nlp' in df_export.columns:
                    resumen_tema = df_export.groupby(['tema_detectado', 'prioridad_nlp']).size().reset_index(name='conteo')
                    resumen_tema.to_excel(writer, sheet_name='Resumen_Temas', index=False)

                if 'palabras_clave' in df_export.columns:
                    todas_keywords = ', '.join(df_export['palabras_clave'].dropna())
                    pd.DataFrame({'palabras_clave_consolidadas': [todas_keywords]}).to_excel(
                        writer, sheet_name='Keywords', index=False
                    )

        st.download_button(
            label="📥 Descargar Excel completo",
            data=buffer.getvalue(),
            file_name=f"sistematizacion_{segmento.replace('/', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_exp2:
        if 'df_analizado' in st.session_state:
            json_str = st.session_state.df_analizado.to_json(orient='records', force_ascii=False, indent=2)
            st.download_button(
                label="📥 Descargar JSON",
                data=json_str,
                file_name=f"sistematizacion_{segmento.replace('/', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

# Instrucciones al final
with st.expander("📖 ¿Cómo usar esta aplicación?"):
    st.markdown("""
    ### Guía paso a paso:

    1. **Ingresa tu token**: En la barra lateral, pega tu token gratuito de HuggingFace
       (obténlo en huggingface.co/settings/tokens).

    2. **Selecciona el segmento**: Elige qué grupo poblacional estás analizando.

    3. **Sube la imagen**: Arrastra o selecciona tu foto en el área de carga.

    4. **Procesa con OCR**: Haz clic en "Procesar con OCR". El modelo DOTS.OCR-1.5
       extraerá el texto automáticamente via API.

    5. **Revisa y edita**: La tabla generada es editable. Corrige cualquier error
       del OCR directamente en las celdas.

    6. **Aplica NLP**: Haz clic en "Aplicar Análisis Automático" para:
       - Clasificar automáticamente en Espacio/Servicio/Barrera
       - Extraer palabras clave
       - Detectar nivel de prioridad

    7. **Visualiza**: Genera gráficos y la nube de palabras para identificar patrones.

    8. **Exporta**: Descarga el Excel con todo el análisis para tu informe final.

    ### Tips para mejores resultados:
    - Usa buena iluminación al tomar la foto
    - Evita ángulos extremos (foto lo más frontal posible)
    - Si el OCR no detecta bien, puedes escribir manualmente en la tabla
    - Procesa un grupo focal a la vez para mantener organizados los segmentos
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Desarrollado para sistematización de diagnóstico participativo - Biblioteca Departamental</div>", unsafe_allow_html=True)
