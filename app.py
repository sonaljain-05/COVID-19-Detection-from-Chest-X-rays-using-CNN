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


# ---------------- PAGE SETTINGS ----------------

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


# ---------------- STYLE ----------------

st.markdown("""
<style>

.title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:#0b3d91;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:25px;
}

.result {
    background:#e8f5e9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#1b5e20;
}

</style>
""", unsafe_allow_html=True)



# ---------------- MODEL LOAD ----------------

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


    model(
        np.zeros((1,224,224,3), dtype=np.float32)
    )


    # Correct file name (no space)
    model_file = Path(__file__).parent / "classifier_head_fp16 .npz"


    if not model_file.exists():
        st.error(
            "classifier_head_fp16.npz file not found."
        )
        st.stop()


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



# ---------------- UI ----------------

st.markdown(
    '<div class="title">🩺 COVID-19 Detection</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">AI Based Chest X-Ray Analysis using VGG16 CNN</div>',
    unsafe_allow_html=True
)



image_url = st.text_input(
    "🔗 Enter Chest X-Ray Image URL"
)



analyze_button = st.button(
    "🔍 Analyze X-Ray"
)



# ---------------- PREDICTION ----------------

if analyze_button:

    if image_url.strip() == "":
        
        st.warning(
            "Please enter an image URL first."
        )

    else:

        try:

            response = requests.get(
                image_url,
                timeout=15
            )


            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")


            st.image(
                image,
                caption="Chest X-Ray",
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
                f"Image loading error: {e}"
            )


else:

    st.info(
        "Enter X-Ray URL and click Analyze X-Ray."
    )
