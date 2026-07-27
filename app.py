import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.title {
    text-align:center;
    font-size:40px;
    color:#0d6efd;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}
.result{
    padding:15px;
    border-radius:10px;
    background:#e9f7ef;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

classes = ["Covid", "Normal", "Viral Pneumonia"]

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    image = image.convert("RGB")
    image = image.resize((224,224))

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = np.max(prediction)*100

    return classes[index], confidence

# -----------------------------
# UI
# -----------------------------
st.markdown("<div class='title'>🩺 COVID-19 Detection</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Chest X-Ray Classification using Deep Learning</div>", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    st.write("")

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing Image..."):

            label, confidence = predict(image)

        st.success("Prediction Completed")

        st.markdown(f"""
        <div class='result'>
        Prediction : {label}<br><br>
        Confidence : {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        if label == "Covid":
            st.error("⚠️ Possible COVID-19 detected. Please consult a medical professional.")

        elif label == "Normal":
            st.success("✅ Chest X-Ray appears Normal.")

        else:
            st.warning("🫁 Viral Pneumonia detected.")
