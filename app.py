from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "glaucoma_model.pkl"
)

try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully.")

except Exception as e:
    print("Error loading model:", e)
    model = None


# --------------------------------------------------
# Extract image features
# --------------------------------------------------

def extract_features(image_bytes):

    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    # Same image size used during training
    image = cv2.resize(image, (64, 64))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    gray = gray.astype(np.float32) / 255.0

    # Convert image into a single feature vector
    features = gray.flatten()

    return features


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Glaucoma Screening API is running",
        "status": "online"
    })


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return jsonify({
            "error": "Model could not be loaded."
        }), 500

    # Check whether an image was uploaded
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # Read image
        image_bytes = file.read()

        # Extract features
        features = extract_features(image_bytes)

        # Get trained class centers
        glaucoma_center = np.array(
            model["glaucoma_center"],
            dtype=np.float32
        )

        normal_center = np.array(
            model["normal_center"],
            dtype=np.float32
        )

        # Calculate distance from each class
        glaucoma_distance = np.linalg.norm(
            features - glaucoma_center
        )

        normal_distance = np.linalg.norm(
            features - normal_center
        )

        # Avoid division by zero
        total_distance = (
            normal_distance + glaucoma_distance
        )

        if total_distance == 0:

            glaucoma_score = 50.0

        else:

            glaucoma_score = (
                normal_distance / total_distance
            ) * 100

        glaucoma_score = round(
            float(glaucoma_score), 2
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        if glaucoma_distance < normal_distance:

            result = "Possible Glaucoma"

            about = (
                "The model detected image patterns "
                "that were more similar to the glaucoma "
                "examples in its training data."
            )

        else:

            result = "Likely Normal"

            about = (
                "The model detected image patterns "
                "that were more similar to the normal "
                "examples in its training data."
            )

        # --------------------------------------------------
        # Stage information
        # --------------------------------------------------

        stage = (
            "Stage estimation unavailable. "
            "This model was trained only for "
            "glaucoma/normal classification."
        )

        # --------------------------------------------------
        # Educational risk factors
        # --------------------------------------------------

        risk_factors = [

            "Age and family history can affect glaucoma risk.",

            "Eye pressure is an important factor.",

            "Regular eye examinations can help detect "
            "eye problems."

        ]

        # --------------------------------------------------
        # Next steps
        # --------------------------------------------------

        next_steps = [

            "This result does not rule out glaucoma.",

            "Continue regular eye examinations when recommended.",

            "Seek professional evaluation if you have "
            "eye-related concerns."

        ]

        # --------------------------------------------------
        # Model information
        # --------------------------------------------------

        model_information = {

            "model_type": "Nearest-Centroid Image Classifier",

            "test_accuracy": 58.82,

            "test_images": 17,

            "training_images": 70

        }

        # --------------------------------------------------
        # Final response
        # --------------------------------------------------

        return jsonify({

            "result": result,

            "model_score": glaucoma_score,

            "stage": stage,

            "risk_factors": risk_factors,

            "about": about,

            "next_steps": next_steps,

            "model_information": model_information,

            "disclaimer": (
                "This system is a college project prototype "
                "and is not a medical diagnostic tool. "
                "The percentage shown is a model score, "
                "not a medical probability or diagnosis."
            )

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


# --------------------------------------------------
# Run the application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
