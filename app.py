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

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected"
        }), 400

    image_bytes = file.read()

    features = extract_features(image_bytes)

    if features is None:

        return jsonify({
            "error": "Invalid image"
        }), 400

    # Make prediction
    prediction = model.predict([features])[0]

    # --------------------------------
    # POSSIBLE GLAUCOMA
    # --------------------------------

    if prediction == 1:

        result = "Possible Glaucoma"

        risk_level = "High"

        stage = "Stage estimation unavailable"

        risk_factors = [
            "Family history of glaucoma",
            "Increased eye pressure",
            "Older age",
            "Certain eye conditions",
            "Some medical conditions or medications"
        ]

        warning_signs = [
            "Early glaucoma may have no noticeable symptoms",
            "Changes in peripheral vision can occur as glaucoma progresses",
            "Some forms of glaucoma can cause eye pain or blurred vision"
        ]

        next_steps = [
            "Consider a comprehensive eye examination",
            "An eye-care professional can measure eye pressure",
            "The optic nerve can be examined",
            "Visual field testing may be performed"
        ]

        about = (
            "Glaucoma is a group of eye diseases that can damage "
            "the optic nerve and may lead to vision loss."
        )

    # --------------------------------
    # LIKELY NORMAL
    # --------------------------------

    else:

        result = "Likely Normal"

        risk_level = "Lower screening concern"

        stage = "Not applicable"

        risk_factors = [
            "Family history of glaucoma",
            "Increased eye pressure",
            "Older age",
            "Certain eye conditions"
        ]

        warning_signs = [
            "Early glaucoma may have no noticeable symptoms"
        ]

        next_steps = [
            "Continue routine eye examinations",
            "Seek professional evaluation if you have eye-related concerns"
        ]

        about = (
            "The prototype did not detect image patterns associated "
            "with the glaucoma class."
        )

    # --------------------------------
    # MODEL SCORE
    # --------------------------------

    model_score = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba([features])[0]

        glaucoma_probability = probabilities[1]

        model_score = round(glaucoma_probability * 100, 2)

    # --------------------------------
    # SEND RESPONSE
    # --------------------------------

    return jsonify({

        "result": result,

        "model_score": model_score,

        "risk_level": risk_level,

        "stage": stage,

        "risk_factors": risk_factors,

        "warning_signs": warning_signs,

        "next_steps": next_steps,

        "about": about,

        "model_information": {
            "algorithm": "Support Vector Machine (SVM)",
            "features": "Histogram of Oriented Gradients (HOG)",
            "image_size": "128 x 128 pixels"
        },

        "disclaimer": (
            "This is a college-project AI screening prototype. "
            "The result is not a medical diagnosis and should not "
            "be used to make medical decisions."
        )

    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
