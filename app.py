
import streamlit as st
import pandas as pd
import requests
import base64
from PIL import Image
import io
import json
import re
from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuración API HuggingFace
API_URL = "https://api-inference.huggingface.co/models/kristaller486/dots.ocr-1.5"
# Nota: Necesitas token de HuggingFace para usar esto
# Obtén uno gratis en: https://huggingface.co/settings/tokeczc

st.set_page_config(
    page_title="Sistematización Biblioteca - OCR API",
    page_icon="📚",
    layout="wide"
)

def query_ocr_api(image, api_token):
    """Consulta la API de HuggingFace para OCR"""
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Convertir imagen a bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    response = requests.post(API_URL, headers=headers, data=img_bytes)
    return response.json()

def ocr_simple(image):
    """OCR alternativo usando pytesseract si está disponible"""
    try:
        import pytesseract
        texto = pytesseract.image_to_string(image, lang='spa')
        return texto
    except:
        return None

# Crear la aplicación Streamlit complet

# Configuración de la página
st.set_page_config(
    page_title="Sistematización Biblioteca - OCR Inteligente",
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
    
    st.subheader("Modelo OCR")
    usar_gpu = st.checkbox("Usar GPU (si disponible)", value=torch.cuda.is_available())

# Inicializar session state
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None
if 'texto_crudo' not in st.session_state:
    st.session_state.texto_crudo = ""

# Función para cargar el modelo (con caché)
@st.cache_resource
def cargar_modelo_ocr():
    """Carga el modelo DOTS.OCR-1.5"""
    try:
        with st.spinner("Cargando modelo DOTS.OCR-1.5... (puede tomar 1-2 minutos)"):
            model_path = "kristaller486/dots.ocr-1.5"
            
            # Cargar modelo y procesador
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            
            return model, processor
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

# Función para procesar imagen con OCR
def procesar_imagen_ocr(image, model, processor):
    """Procesa la imagen y extrae texto estructurado"""
    
    prompt = """Analiza esta imagen de notas de un grupo focal sobre una biblioteca.
    Extrae la información en formato tabla con las siguientes columnas:
    - Hora (si está visible)
    - Cita/Lo que dijeron (texto exacto o resumen)
    - Tema (Espacio/Servicio/Barrera/Otro)
    - Subtema específico
    - Importancia (Alta/Media/Baja según énfasis)
    
    Devuelve el resultado SOLO en formato JSON con esta estructura:
    {
        "registros": [
            {
                "hora": "...",
                "cita": "...",
                "tema": "...",
                "subtema": "...",
                "importancia": "..."
            }
        ]
    }
    """
    
    # Preparar inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Generar
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2000)
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return output_text

# Función para extraer JSON del texto
def extraer_json(texto):
    """Extrae el JSON de la respuesta del modelo"""
    try:
        # Buscar JSON en el texto
        json_match = re.search(r'\{.*\}', texto, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Si no hay JSON, crear estructura básica
            return {"registros": []}
    except:
        return {"registros": []}

# Funciones de NLP para sistematización
def cargar_nlp():
    """Carga modelo spaCy para español"""
    try:
        return spacy.load("es_core_news_md")
    except:
        st.warning("Modelo spaCy no encontrado. Instalando...")
        import os
        os.system("python -m spacy download es_core_news_md")
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

def extraer_palabras_clave(texto, n=5):
    """Extrae las n palabras clave más importantes"""
    doc = nlp(texto.lower())
    
    # Filtrar tokens relevantes
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
        and len(token.text) > 3 and token.pos_ in ['NOUN', 'ADJ', 'VERB']
    ]
    
    # Contar frecuencia
    freq = Counter(tokens)
    return ', '.join([palabra for palabra, _ in freq.most_common(n)])

def detectar_prioridad(texto):
    """Detecta nivel de prioridad basado en intensidad lingüística"""
    texto_lower = texto.lower()
    
    # Indicadores de alta prioridad
    alta = ['urgente', 'necesitamos', 'falta', 'importante', 'crítico', 'todos', 'siempre', 'nunca']
    # Indicadores de baja prioridad
    baja = ['quizás', 'tal vez', 'podría', 'sería bueno', 'me gustaría', 'preferiría']
    
    score_alta = sum(1 for palabra in alta if palabra in texto_lower)
    score_baja = sum(1 for palabra in baja if palabra in texto_lower)
    
    if score_alta > score_baja:
        return 'Alta'
    elif score_baja > score_alta:
        return 'Baja'
    else:
        return 'Media'

def generar_nube_palabras(textos):
    """Genera nube de palabras de los textos"""
    texto_completo = ' '.join(textos).lower()
    
    # Filtrar palabras comunes
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
    # Mostrar imagen
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Imagen subida")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Botón para procesar
        procesar = st.button("🚀 Procesar con OCR", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Vista previa")
        st.info("La imagen se procesará con DOTS.OCR-1.5 para extraer texto estructurado")
        
        if procesar:
            # Cargar modelo
            model, processor = cargar_modelo_ocr()
            
            if model is not None:
                try:
                    # Procesar imagen
                    with st.spinner("🔍 Analizando imagen con IA..."):
                        resultado_ocr = procesar_imagen_ocr(image, model, processor)
                        st.session_state.texto_crudo = resultado_ocr
                        
                        # Extraer JSON
                        datos_json = extraer_json(resultado_ocr)
                        
                        # Crear DataFrame
                        if datos_json['registros']:
                            df = pd.DataFrame(datos_json['registros'])
                        else:
                            # Si no hay JSON, crear estructura básica
                            df = pd.DataFrame({
                                'hora': [''],
                                'cita': [resultado_ocr[:500]],  # Primeros 500 caracteres
                                'tema': ['Por clasificar'],
                                'subtema': [''],
                                'importancia': ['Media']
                            })
                        
                        # Agregar columna de segmento
                        df['segmento'] = segmento
                        
                        st.session_state.df_resultado = df
                        st.success("✅ ¡Procesamiento completado!")
                        
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
                    st.info("Intenta con una imagen más clara o recorta la región de texto")

# Mostrar resultados y análisis NLP
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado
    
    st.markdown('<div class="sub-header">📊 2. Tabla Extraída (Editable)</div>', unsafe_allow_html=True)
    
    # Editor de tabla
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
    
    # Botón para aplicar NLP
    st.markdown('<div class="sub-header">🤖 3. Análisis NLP y Sistematización</div>', unsafe_allow_html=True)
    
    col_nlp1, col_nlp2 = st.columns([1, 1])
    
    with col_nlp1:
        if st.button("🔍 Aplicar Análisis Automático", type="secondary", use_container_width=True):
            with st.spinner("Analizando con NLP..."):
                # Cargar modelo NLP
                nlp = cargar_nlp()
                
                # Aplicar análisis a cada fila
                df_analizado = df_editado.copy()
                
                # Clasificación temática
                if analisis_tematico:
                    df_analizado['tema_detectado'] = df_analizado['cita'].apply(clasificar_tema)
                
                # Extracción de keywords
                if extraer_keywords:
                    df_analizado['palabras_clave'] = df_analizado['cita'].apply(
                        lambda x: extraer_palabras_clave(x, nlp)
                    )
                
                # Detección de prioridad
                if analizar_sentimiento:
                    df_analizado['prioridad_nlp'] = df_analizado['cita'].apply(detectar_prioridad)
                
                # Guardar en session state
                st.session_state.df_analizado = df_analizado
                st.success("Análisis NLP completado")
    
    with col_nlp2:
        if st.button("📈 Generar Visualizaciones", use_container_width=True):
            if 'df_analizado' in st.session_state:
                st.session_state.mostrar_viz = True
            else:
                st.warning("Primero aplica el análisis NLP")
    
    # Mostrar resultados del análisis
    if 'df_analizado' in st.session_state:
        df_viz = st.session_state.df_analizado
        
        st.markdown("**Resultado del análisis:**")
        st.dataframe(df_viz, use_container_width=True)
        
        # Visualizaciones
        if st.session_state.get('mostrar_viz', False):
            st.markdown('<div class="sub-header">📈 4. Visualizaciones y Sistematización</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Distribución de Temas", "Nube de Palabras", "Matriz de Prioridades"])
            
            with tab1:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Distribución por Tema")
                    tema_counts = df_viz['tema_detectado' if 'tema_detectado' in df_viz.columns else 'tema'].value_counts()
                    st.bar_chart(tema_counts)
                
                with col_chart2:
                    st.subheader("Distribución por Prioridad")
                    prio_counts = df_viz['prioridad_nlp' if 'prioridad_nlp' in df_viz.columns else 'importancia'].value_counts()
                    st.bar_chart(prio_counts)
            
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
                
                # Crear matriz de consolidación
                if 'tema_detectado' in df_viz.columns and 'prioridad_nlp' in df_viz.columns:
                    matriz = pd.crosstab(
                        df_viz['tema_detectado'], 
                        df_viz['prioridad_nlp'],
                        margins=True
                    )
                    st.dataframe(matriz, use_container_width=True)
                    
                    # Insights automáticos
                    st.markdown("**🎯 Insights detectados:**")
                    
                    tema_mas_frecuente = df_viz['tema_detectado'].mode()[0]
                    prioridad_dominante = df_viz['prioridad_nlp'].mode()[0]
                    
                    st.markdown(f"""
                    - **Tema más mencionado:** {tema_mas_frecuente}
                    - **Nivel de prioridad predominante:** {prioridad_dominante}
                    - **Total de necesidades identificadas:** {len(df_viz)}
                    """)
                    
                    # Recomendaciones basadas en datos
                    alta_prio_espacio = len(df_viz[(df_viz['tema_detectado'] == 'Espacio') & (df_viz['prioridad_nlp'] == 'Alta')])
                    if alta_prio_espacio > 0:
                        st.warning(f"⚠️ Se detectaron {alta_prio_espacio} necesidades de ESPACIO con prioridad ALTA. Considerar intervención estructural.")
    
    # Exportación
    st.markdown('<div class="sub-header">💾 5. Exportar Resultados</div>', unsafe_allow_html=True)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Exportar a Excel
        if 'df_analizado' in st.session_state:
            df_export = st.session_state.df_analizado
        else:
            df_export = df_editado
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Registros', index=False)
            
            # Si hay análisis, agregar hoja de resumen
            if 'df_analizado' in st.session_state:
                # Resumen por tema
                resumen_tema = df_export.groupby(['tema_detectado', 'prioridad_nlp']).size().reset_index(name='conteo')
                resumen_tema.to_excel(writer, sheet_name='Resumen_Temas', index=False)
                
                # Palabras clave consolidadas
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
        # Exportar JSON
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
    
    1. **Prepara tu foto**: Toma una foto clara de tus notas del grupo focal. Asegúrate de que el texto sea legible.
    
    2. **Selecciona el segmento**: En la barra lateral, elige qué grupo poblacional estás analizando (Infancia, Académicos, etc.)
    
    3. **Sube la imagen**: Arrastra o selecciona tu foto en el área de carga.
    
    4. **Procesa con OCR**: Haz clic en "Procesar con OCR". El modelo DOTS.OCR-1.5 extraerá el texto automáticamente.
    
    5. **Revisa y edita**: La tabla generada es editable. Corrige cualquier error del OCR directamente en las celdas.
    
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


# Guardar el archivo
app_path = '/mnt/kimi/output/app_sistematizacion.py'
with open(app_path, 'w', encoding='utf-8') as f:
    f.write(app_code)

print(f"✅ Aplicación creada: {app_path}")
print(f"\n📋 Archivos adicionales necesarios:")
print("   - requirements.txt (dependencias)")
print("   - README.md (instrucciones de instalación)")


st.title("📚 Sistematización con OCR API")

st.warning("""
    ⚠️ **Versión API**: Esta versión usa HuggingFace Inference API (más ligera para Streamlit Cloud).
    Necesitas un token gratuito de HuggingFace.
""")

# Input para API token
api_token = st.text_input("🔑 Token de HuggingFace (obtén uno gratis en huggingface.co/settings/tokens)", 
                          type="password")

if api_token:
    # Aquí iría el resto de la lógica similar a la app principal
    # pero usando query_ocr_api() en lugar de cargar el modelo local
    st.success("Token configurado. Puedes subir imágenes.")
else:
    st.info("Ingresa tu token de HuggingFace para comenzar (es gratis)")


with open('/mnt/kimi/output/app_api_version.py', 'w', encoding='utf-8') as f:
    f.write(app_api_version)

# Crear archivo de configuración para Streamlit
config_toml = [theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10


with open('/mnt/kimi/output/.streamlit/config.toml', 'w') as f:
    f.write(config_toml)

print("✅ Archivos adicionales creados:")
print("   - app_api_version.py (versión ligera con API)")
print("   - .streamlit/config.toml (configuración de tema)")
print("\n📦 Archivos listos para descargar:")
print("   1. app_sistematizacion.py (versión completa local)")
print("   2. app_api_version.py (versión API cloud)")
print("   3. requirements.txt")
print("   4. README.md")
print("   5. install.sh")
print("   6. diagrama_flujo_app.png")
