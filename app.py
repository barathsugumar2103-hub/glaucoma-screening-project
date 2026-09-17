from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import joblib
import numpy as np
from skimage.feature import hog

app = Flask(__name__)

CORS(app)

MODEL_FILE = "glaucoma_model.pkl"

# Load trained model
model = joblib.load(MODEL_FILE)


def extract_features(image_bytes):

    # Convert uploaded image into OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


@app.route("/")
def home():

    return jsonify({
        "message": "Glaucoma Screening API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    image_bytes = file.read()

    features = extract_features(image_bytes)

    if features is None:
        return jsonify({"error": "Invalid image"}), 400

    prediction = model.predict([features])[0]

    # Get prototype model score
    model_score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]

        if len(probabilities) > 1:
            model_score = round(float(probabilities[1]) * 100, 2)

    if prediction == 1:

        result = "Possible Glaucoma"

        stage = "Stage estimation unavailable"

        risk_factors = [
            "Increasing age can be associated with glaucoma risk.",
            "Family history of glaucoma may increase risk.",
            "High eye pressure is an important clinical risk factor.",
            "Some medical conditions can be associated with increased risk."
        ]

        about = (
            "The prototype model detected features associated with "
            "the glaucoma class in the uploaded image."
        )

        next_steps = [
            "Consider professional eye examination.",
            "An eye specialist can perform appropriate glaucoma tests.",
            "Do not use this prototype result as a medical diagnosis."
        ]

    else:

        result = "Likely Normal"

        stage = "No glaucoma stage estimated"

        risk_factors = [
            "A normal prototype result does not rule out glaucoma.",
            "Regular eye examinations can help detect eye problems early."
        ]

        about = (
            "The prototype model classified this image as belonging "
            "to the normal class."
        )

        next_steps = [
            "Continue routine eye examinations.",
            "Seek professional evaluation if you have eye-related concerns.",
            "Do not use this prototype result as a medical diagnosis."
        ]

    return jsonify({
        "result": result,
        "model_score": model_score,
        "stage": stage,
        "risk_factors": risk_factors,
        "about": about,
        "next_steps": next_steps,
        "disclaimer": (
            "This is a college-project screening prototype and "
            "not a medical diagnostic tool."
        )
    })
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
