from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import joblib
import numpy as np
from skimage.feature import hog

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load trained glaucoma model
# --------------------------------------------------

MODEL_FILE = "glaucoma_model.pkl"

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features(image_bytes):

    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG feature extraction
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/")
def home():

    return jsonify({
        "message": "Glaucoma Screening API is running - V3"
    })


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check image
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected"
        }), 400

    # Read image
    image_bytes = file.read()

    # Extract HOG features
    features = extract_features(image_bytes)

    if features is None:

        return jsonify({
            "error": "Invalid image"
        }), 400

    # --------------------------------------------------
    # Glaucoma prediction
    # --------------------------------------------------

    prediction = model.predict([features])[0]

    # --------------------------------------------------
    # Model confidence / estimate
    # --------------------------------------------------

    model_score = None

    try:

        probabilities = model.predict_proba([features])[0]

        if len(probabilities) > 1:

            # Probability associated with glaucoma class
            model_score = round(
                float(probabilities[1]) * 100,
                2
            )

    except Exception as error:

        print("Confidence calculation error:", error)


    # --------------------------------------------------
    # Result information
    # --------------------------------------------------

    if prediction == 1:

        result = "Possible Glaucoma"

        risk_factors = [
            "Family history may be associated with increased glaucoma risk.",
            "High eye pressure is an important clinical risk factor.",
            "Increasing age can be associated with glaucoma risk.",
            "Some medical conditions may be associated with increased risk."
        ]

        about = (
            "The prototype AI model classified this image "
            "as belonging to the glaucoma class."
        )

        next_steps = [
            "Consider having a professional eye examination.",
            "An eye specialist can perform appropriate glaucoma tests.",
            "Do not use this prototype result as a medical diagnosis."
        )

    else:

        result = "Likely Normal"

        risk_factors = [
            "A normal prototype result does not rule out glaucoma.",
            "Regular eye examinations can help detect eye problems early."
        ]

        about = (
            "The prototype AI model classified this image "
            "as belonging to the normal class."
        )

        next_steps = [
            "Continue routine eye examinations.",
            "Seek professional evaluation if you have eye-related concerns.",
            "Do not use this prototype result as a medical diagnosis."
        ]


    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return jsonify({

        "result": result,

        "model_score": model_score,

        "risk_factors": risk_factors,

        "about": about,

        "next_steps": next_steps,

        "disclaimer": (
            "This is an educational AI prototype and "
            "not a medical diagnostic system."
        )
    })


# --------------------------------------------------
# Run locally
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
