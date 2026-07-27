import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
    model(np.zeros((1,224,224,3), dtype=np.float32))

    w = np.load("classifier_head_fp16.npz")

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
