import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)


# ---------- Custom CSS ----------
st.markdown("""
<style>

.main {
    background-color: #f7fbff;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #0b3d91;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 20px;
    border-radius: 15px;
    background: #e8f5e9;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1b5e20;
}

</style>
""", unsafe_allow_html=True)



classes = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]



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


    model(np.zeros((1,224,224,3), dtype=np.float32))


    model_file = Path(__file__).parent / "classifier_head_fp16 .npz"


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



# ---------- UI ----------

st.markdown(
    '<div class="title">🩺 COVID-19 Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Chest X-Ray Analysis using VGG16 CNN</div>',
    unsafe_allow_html=True
)



st.markdown('<div class="card">', unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg","jpeg","png"]
)


st.markdown('</div>', unsafe_allow_html=True)



if uploaded_file:


    image = Image.open(uploaded_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )


    img = image.resize((224,224))


    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    with st.spinner("Analyzing X-Ray..."):

        prediction = model.predict(img_array)


    result = classes[np.argmax(prediction)]

    confidence = np.max(prediction)*100



    st.markdown(
        f"""
        <div class="result">
        🧬 Result: {result}
        <br>
        🎯 Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )


    st.progress(
        int(confidence)
    )


else:

    st.info(
        "Please upload a chest X-ray image to start diagnosis."
    )
