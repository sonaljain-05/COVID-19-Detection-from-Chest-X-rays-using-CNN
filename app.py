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


classes = ["Covid", "Normal", "Viral Pneumonia"]


@st.cache_resource
def load_model():

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
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
    model(np.zeros((1,224,224,3), dtype=np.float32))


    # Load npz file
    model_file = Path(__file__).parent / "classifier_head_fp16 .npz"


    if not model_file.exists():
        st.error("classifier_head_fp16.npz file not found")
        st.stop()


    # Debug information
    st.write("File exists:", model_file.exists())
    st.write("File size:", model_file.stat().st_size)


    with open(model_file, "rb") as f:
        st.write("File header:", f.read(20))


    try:
        w = np.load(
            model_file,
            allow_pickle=True
        )

        st.write("NPZ Keys:", w.files)


    except Exception as e:
        st.error(f"NPZ loading error: {e}")
        st.stop()



    # Load Dense layer weights
    model.layers[2].set_weights([
        w["dense_kernel"].astype(np.float32),
        w["dense_bias"].astype(np.float32)
    ])


    # Load Output layer weights
    model.layers[4].set_weights([
        w["output_kernel"].astype(np.float32),
        w["output_bias"].astype(np.float32)
    ])


    return model



model = load_model()



st.title("🩺 COVID-19 Detection from Chest X-Ray")


uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")


    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )


    img = image.resize((224,224))


    img_array = np.array(img) / 255.0


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    prediction = model.predict(img_array)


    result = classes[np.argmax(prediction)]


    confidence = np.max(prediction) * 100


    st.success(
        f"Prediction: {result}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
