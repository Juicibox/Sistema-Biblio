import streamlit as st
import pandas as pd
from PIL import Image
import io
import requests

MODEL_ID = "kristaller486/dots.ocr-1.5"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OCR - dots.ocr-1.5",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Extracción de datos con dots.ocr-1.5")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    hf_token = st.text_input(
        "🔑 HuggingFace API Token",
        type="password",
        help="Obtén uno en: huggingface.co/settings/tokens"
    )
    st.caption("👉 [Obtener token gratis](https://huggingface.co/settings/tokens)")

if not hf_token:
    st.warning("⚠️ Ingresa tu HuggingFace API Token en la barra lateral para continuar.")
    st.stop()

# ─────────────────────────────────────────────
# OCR con dots.ocr-1.5
# ─────────────────────────────────────────────
def ocr_with_dots(image: Image.Image, token: str) -> str:
    """Envía la imagen al modelo dots.ocr-1.5 y devuelve el texto extraído.

    La API de HuggingFace Inference devuelve para modelos image-to-text:
      [{"generated_text": "..."}]  (lista con un dict)
    o en caso de error un dict con campo "error".
    """
    buf = io.BytesIO()
    # Preserve original format when known; fall back to PNG for in-memory images.
    fmt = image.format if image.format in ("PNG", "JPEG", "WEBP", "BMP", "TIFF") else "PNG"
    image.save(buf, format=fmt)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(API_URL, headers=headers, data=buf.getvalue(), timeout=60)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "")
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


def texto_a_tabla(texto: str) -> pd.DataFrame:
    """Convierte el texto OCR en un DataFrame: una fila por línea no vacía."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    if not lineas:
        return pd.DataFrame({"Texto extraído": []})
    return pd.DataFrame({"Texto extraído": lineas})

# ─────────────────────────────────────────────
# Subir imagen y procesar
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Sube una foto para extraer datos",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagen")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Resultado")
        if st.button("🔍 Extraer datos con OCR", type="primary", use_container_width=True):
            with st.spinner("Procesando con dots.ocr-1.5…"):
                try:
                    texto = ocr_with_dots(image, hf_token)
                    df = texto_a_tabla(texto)
                    st.session_state["df_ocr"] = df
                    st.success("✅ Procesamiento completado")
                except requests.Timeout:
                    st.error("⏳ El servicio OCR tardó demasiado. Intenta de nuevo en unos segundos.")
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status == 401:
                        st.error("❌ Token inválido. Verifica tu HuggingFace API Token.")
                    elif status == 503:
                        st.error("⏳ Modelo cargando. Espera unos segundos e intenta de nuevo.")
                    else:
                        st.error(f"Error HTTP {status}: {e}")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")

        if "df_ocr" in st.session_state and st.session_state["df_ocr"] is not None:
            st.dataframe(st.session_state["df_ocr"], use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Powered by kristaller486/dots.ocr-1.5 · HuggingFace Inference API</div>",
    unsafe_allow_html=True,
)
