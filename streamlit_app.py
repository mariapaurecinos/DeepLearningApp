"""
Streamlit multi-model deployment demo (flat layout)
==================================================

Esta versión usa los nombres **exactos** que aparecen en tu carpeta:

```
├── model_image.keras          # CNN de imágenes (CIFAR-10, etc.)
├── model_text.keras           # Clasificador de texto (sentiment)
├── diabetes_regressor.keras   # Modelo de regresión (ej. Diabetes)
├── scaler.pkl                 # (opcional) StandardScaler u otro pre-procesador
├── text_tokenizer.pkl         # Tokenizer de Keras --> lo crearás tú
├── streamlit_app.py           # <— este archivo
└── requirements.txt
```

> Si entrenaste tu modelo de regresión con un `StandardScaler`, déjalo guardado en
> `scaler.pkl`.  Si no, simplemente elimina el fichero y el código lo ignorará.

Lanza con:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
import joblib

# --------------------------------------------------------------
#  Carga perezosa de TensorFlow para acelerar las páginas que
#  no lo necesitan (hasta que el usuario lo requiera).
# --------------------------------------------------------------

def _lazy_tf():
    import tensorflow as tf  # pylint: disable=import-error
    return tf

# ------------------ LOADERS CON CACHÉ ------------------------

@st.cache_resource(show_spinner=False)
def load_image_model():
    tf = _lazy_tf()
    return tf.keras.models.load_model("model_image.keras")

@st.cache_resource(show_spinner=False)
def load_text_model():
    tf = _lazy_tf()
    return tf.keras.models.load_model("model_text.keras")

@st.cache_resource(show_spinner=False)
def load_regression_model():
    tf = _lazy_tf()
    return tf.keras.models.load_model("diabetes_regressor.keras")

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    return joblib.load("text_tokenizer.pkl")

@st.cache_resource(show_spinner=False)
def load_scaler():
    path = Path("scaler.pkl")
    return joblib.load(path) if path.exists() else None

# ------------------ CONSTANTES A EDITAR ----------------------

IMAGE_SIZE: tuple[int, int] = (32, 32)  # tamaño de entrada del CNN
IMAGE_CLASS_NAMES: list[str] = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]  # ajusta al orden de tu modelo

TEXT_MAXLEN: int = 400  # longitud de secuencia usada en entrenamiento

FEATURE_NAMES: list[str] = [
    # pon aquí las columnas (en orden) del set de entrenamiento del modelo de regresión
    "age", "sex", "bmi", "average blood pressure", "total serum cholesterol", " low-density lipoproteins", "high-density lipoprotein", " total cholesterol / HDL", "serum triglycerides level", " blood sugar level",
]

# ------------------ CONFIGURACIÓN UI -------------------------

st.set_page_config(
    page_title="🧠 Multi-Model Demo",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🧠 Multi-Model Prediction Demo")

pagina = st.sidebar.selectbox(
    "Elige el modelo a probar:",
    ("Clasificación de Imagen", "Clasificación de Texto", "Regresión"),
)

# ------------------ PÁGINA: IMAGEN ---------------------------
import numpy as np
from typing import Tuple

def preprocess_images(
        images: np.ndarray,
        labels: np.ndarray | None = None,
        flatten: bool = True,
        one_hot: bool = False,
        num_classes: int | None = None
) -> Tuple[np.ndarray, np.ndarray | None]:

    images = images / 255.0

    if flatten:
        images = images.reshape(images.shape[0], -1)

    if labels is not None:
        if one_hot:
            if num_classes is None:
                num_classes = int(np.max(labels)) + 1
            labels = np.eye(num_classes, dtype='float32')[labels]
        return images, labels
    else:
        return images, None

if pagina == "Clasificación de Imagen":
    st.header("📷 Clasificación de Imagen")

    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        imagen = Image.open(archivo).convert("RGB").resize(IMAGE_SIZE)
        st.image(imagen, caption="Imagen cargada", use_container_width=False)

        # Convertir a array y expandir dimensión
        x = np.asarray(imagen, dtype="float32")
        x = np.expand_dims(x, axis=0)  # Para que tenga shape (1, H, W, 3)
        
        
        # Preprocesar imagen usando tu función
        x, _ = preprocess_images(x, flatten=True)  # Aquí NO aplanamos, porque el modelo espera (1, H, W, 3)

        with st.spinner("Generando predicción …"):
            pred = load_image_model().predict(x, verbose=0)
            idx = int(np.argmax(pred, axis=-1))
            clase = IMAGE_CLASS_NAMES[idx]
            conf = float(np.max(pred))

        st.success(f"**Predicción:** {clase}")
        st.caption(f"Confianza: {conf:.3f}")

# ------------------ PÁGINA: TEXTO ----------------------------

elif pagina == "Clasificación de Texto":
    st.header("📝 Sentiment Analysis (1 = positivo, 0 = negativo)")

    texto = st.text_area(
        "Ingresa una oración:",
        placeholder="Hoy fue un gran día para aprender Streamlit!",
    )

    if st.button("Predecir") and texto.strip():
        tok = load_tokenizer()
        seq = tok.texts_to_sequences([texto])
        tf = _lazy_tf()
        seq_pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=TEXT_MAXLEN, padding="post"
        )

        with st.spinner("Generando predicción …"):
            prob_pos = float(load_text_model().predict(seq_pad, verbose=0)[0][0])
            label = 1 if prob_pos >= 0.5 else 0
        prediccion_texto = "Positive review"
        if label == 1:
            prediccion_texto = prediccion_texto
        else:
            prediccion_texto = "Negative review"
        st.success(f"**Predicción:** {prediccion_texto}")
        st.caption(f"Probabilidad positiva: {prob_pos:.3f}")

# ------------------ PÁGINA: REGRESIÓN ------------------------
else:
    st.header("📈 Predicción de Regresión")
    st.write("Se utilizó el dataset Diabetes de skelearn")
    st.write("Introduce las variables numéricas del modelo:")

    with st.form("reg_form"):
        valores = []
        for f in FEATURE_NAMES:
            if f == "sex":
                sexo = st.selectbox("Sexo (F = Femenino, M = Masculino):", ["F", "M"])
                valores.append(1.0 if sexo == "M" else 0.0)
            elif f == "age":
                valor = st.number_input(f, min_value=0, max_value=120, step=1, value=0)
                valores.append(int(valor))
            else:
                valor = st.number_input(f, value=0.0, format="%.4f")
                valores.append(valor)
        enviar = st.form_submit_button("Predecir")

    if enviar:
        x = np.array(valores, dtype="float32").reshape(1, -1)
        scaler = load_scaler()
        if scaler is not None:
            x = scaler.transform(x)
        with st.spinner("Generando predicción …"):
            y = load_regression_model().predict(x)
        st.success(f"**Valor predicho:** {float(y[0]):.4f}")

#activar entorno virtual
#.\tfenv\Scripts\python.exe -c "import sys; print(sys.executable)"

#ejecutar:
#.\tfenv\Scripts\python.exe -m streamlit run streamlit_app.py
