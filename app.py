from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

MODEL_FILE = "glaucoma_model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "glaucoma_model.pkl not found. Make sure the model file is in the same folder as app.py."
    )

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

glaucoma_center = np.asarray(model["glaucoma_center"], dtype=np.float32)
normal_center = np.asarray(model["normal_center"], dtype=np.float32)

EXPECTED_FEATURE_SIZE = len(glaucoma_center)

# The saved college-project model contains an image_size of (64, 64)
# and 4096 features (64 x 64). It was trained using flattened,
# normalized grayscale images, not HOG features.
saved_size = model.get("image_size", (64, 64))

try:
    saved_size = tuple(saved_size)
except Exception:
    saved_size = (64, 64)

if len(saved_size) != 2 or saved_size[0] != saved_size[1]:
    saved_size = (64, 64)

image_size = (int(saved_size[0]), int(saved_size[1]))

if image_size[0] * image_size[1] != EXPECTED_FEATURE_SIZE:
    raise ValueError(
        f"Model feature size mismatch. The saved model expects "
        f"{EXPECTED_FEATURE_SIZE} features, but image_size {image_size} "
        f"produces {image_size[0] * image_size[1]} features."
    )

print("Model loaded successfully.")
print("Model feature size:", EXPECTED_FEATURE_SIZE)
print("Using image size:", image_size)
print("Feature extractor: 64x64 normalized grayscale pixels")


def extract_features(image):
    """
    Reproduce the feature format used by the saved model:
    1. Resize to the model's saved image size.
    2. Convert to grayscale.
    3. Normalize pixel values to 0..1.
    4. Flatten to a 1-D vector.
    """
    resized = cv2.resize(image, image_size)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features = gray.astype(np.float32) / 255.0
    features = features.flatten()

    if len(features) != EXPECTED_FEATURE_SIZE:
        raise ValueError(
            f"Feature size mismatch. Model expects {EXPECTED_FEATURE_SIZE}, "
            f"but extractor produced {len(features)}."
        )

    return features


def predict_image(features):
    features = np.asarray(features, dtype=np.float32)

    glaucoma_distance = np.linalg.norm(
        features - glaucoma_center
    )

    normal_distance = np.linalg.norm(
        features - normal_center
    )

    total_distance = glaucoma_distance + normal_distance

    if total_distance > 0:
        glaucoma_score = (
            normal_distance / total_distance
        ) * 100.0
    else:
        glaucoma_score = 50.0

    if glaucoma_distance < normal_distance:
        result = "Possible Glaucoma"
    else:
        result = "Likely Normal"

    return result, round(float(glaucoma_score), 2)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Glaucoma Screening API is running",
        "status": "online"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No image selected."}), 400

        image_bytes = file.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({
                "error": "Unable to read the uploaded image."
            }), 400

        features = extract_features(image)

        result, score = predict_image(features)

        risk_factors = [
            "Increasing age can increase glaucoma risk.",
            "Family history of glaucoma can increase risk.",
            "High eye pressure is an important risk factor.",
            "Some eye conditions and medications may affect glaucoma risk."
        ]

        if result == "Possible Glaucoma":
            about = (
                "The model detected image patterns that were more similar "
                "to the glaucoma examples in its training data."
            )
        else:
            about = (
                "The model detected image patterns that were more similar "
                "to the normal examples in its training data."
            )

        next_steps = [
            "This is an experimental AI screening result.",
            "A qualified eye-care professional should evaluate the eyes.",
            "A complete eye examination may include eye-pressure measurement and optic-nerve assessment."
        ]

        return jsonify({
            "result": result,

            # This is a model-derived screening score, not a medical probability.
            "screening_percentage": score,
            "model_score": score,

            "risk_factors": risk_factors,
            "about": about,
            "next_steps": next_steps,

            "model_information": {
                "model_type": "Nearest-Centroid Classifier",
                "feature_extractor": "64x64 normalized grayscale pixels",
                "feature_size": EXPECTED_FEATURE_SIZE,
                "image_size": list(image_size),
                "training_images": 70,
                "test_images": 16,
                "test_accuracy": 75.0
            },

            "disclaimer": (
                "This system is a college project prototype and is not "
                "a medical diagnostic tool. The screening percentage is "
                "a model score, not a medical probability or diagnosis."
            )
        })

    except Exception as e:
        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
