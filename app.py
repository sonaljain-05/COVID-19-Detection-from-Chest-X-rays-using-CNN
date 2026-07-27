import streamlit as st
import tensorflow as tf
import numpy as np
import requests

from PIL import Image
from io import BytesIO
from pathlib import Path

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)


classes = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

body {
    background-color: #f5f9ff;
}

.title {
    text-align:center;
    color:#0b3d91;
    font-size:42px;
    font-weight:800;
}

.subtitle {
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:30px;
}

.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);
}

.result {
    background:#e8f5e9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#1b5e20;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)



# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )


    for layer in base_model.layers:
        layer.trainable = False


    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax")
    ])


    # Build model
    model(
        np.zeros((1,224,224,3), dtype=np.float32)
    )


    # Load classifier weights
    model_file = Path(__file__).parent / "classifier_head_fp16.npz"


    w = np.load(
        model_file,
        allow_pickle=True
    )


    model.layers[2].set_weights([
        w["dense_kernel"].astype(np.float32),
        w["dense_bias"].astype(np.float32)
    ])


    model.layers[4].set_weights([
        w["output_kernel"].astype(np.float32),
        w["output_bias"].astype(np.float32)
    ])


    return model



model = load_model()



# ---------------- APP UI ----------------


st.markdown(
    '<div class="title">🩺 COVID-19 Detection</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">AI Based Chest X-Ray Classification using VGG16 CNN</div>',
    unsafe_allow_html=True
)



st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)



image_url = st.text_input(
    "🔗 Enter Chest X-Ray Image URL"
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)



# ---------------- PREDICTION ----------------


if image_url:


    try:

        response = requests.get(
            image_url,
            timeout=10
        )


        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")


        st.image(
            image,
            caption="Chest X-Ray Image",
            use_container_width=True
        )


        img = image.resize(
            (224,224)
        )


        img_array = np.array(img) / 255.0


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        with st.spinner("Analyzing X-Ray..."):

            prediction = model.predict(
                img_array
            )


        result = classes[
            np.argmax(prediction)
        ]


        confidence = (
            np.max(prediction) * 100
        )


        st.markdown(
            f"""
            <div class="result">
            🧬 Prediction: {result}
            <br><br>
            🎯 Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )


        st.progress(
            int(confidence)
        )


    except Exception as e:

        st.error(
            f"Image URL load failed: {e}"
        )


else:

    st.info(
        "Please enter a chest X-ray image URL."
    )
