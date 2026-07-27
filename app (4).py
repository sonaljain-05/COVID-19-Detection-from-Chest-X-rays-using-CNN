from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("best_model.h5")

classes = ["Covid", "Normal", "Viral Pneumonia"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return classes[index], round(confidence, 2)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    image = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result, confidence = predict(path)
            image = path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=image
    )


if __name__ == "__main__":
    app.run(debug=True)