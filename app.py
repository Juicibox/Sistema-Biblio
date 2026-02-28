import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import requests

API_URL = "https://dotsocr.xiaohongshu.com/run/predict"

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
# OCR con dots.ocr-1.5 (dotsocr.xiaohongshu.com)
# ─────────────────────────────────────────────
def ocr_with_dots(image: Image.Image) -> str:
    """Envía la imagen a dotsocr.xiaohongshu.com y devuelve el texto extraído.

    La API espera una petición JSON con la imagen codificada en base64:
      {"data": ["data:<mime>;base64,<encoded>"]}
    y devuelve:
      {"data": ["<texto extraído>"]}
    """
    buf = io.BytesIO()
    # Preserve original format when known; fall back to PNG for in-memory images.
    fmt = image.format if image.format in ("PNG", "JPEG", "WEBP", "BMP", "TIFF") else "PNG"
    _MIME = {"JPEG": "image/jpeg", "WEBP": "image/webp", "BMP": "image/bmp", "TIFF": "image/tiff"}
    mime = _MIME.get(fmt, "image/png")
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {"data": [f"data:{mime};base64,{b64}"]}
    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict) and "data" in result:
        data = result["data"]
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], str) else str(data[0])
    if isinstance(result, list) and result:
        return result[0] if isinstance(result[0], str) else str(result[0])
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
                    texto = ocr_with_dots(image)
                    df = texto_a_tabla(texto)
                    st.session_state["df_ocr"] = df
                    st.success("✅ Procesamiento completado")
                except requests.Timeout:
                    st.error("⏳ El servicio OCR tardó demasiado. Intenta de nuevo en unos segundos.")
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status == 410:
                        st.error(
                            "❌ El servicio OCR no está disponible (error 410). "
                            "Puedes usar la interfaz web directamente en: "
                            "https://dotsocr.xiaohongshu.com/"
                        )
                    elif status == 503:
                        st.error("⏳ Servicio no disponible. Espera unos segundos e intenta de nuevo.")
                    else:
                        st.error(f"Error HTTP {status}: {e}")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")

        if "df_ocr" in st.session_state and st.session_state["df_ocr"] is not None:
            st.dataframe(st.session_state["df_ocr"], use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Powered by dots.ocr-1.5 · <a href='https://dotsocr.xiaohongshu.com/' target='_blank'>dotsocr.xiaohongshu.com</a></div>",
    unsafe_allow_html=True,
)
