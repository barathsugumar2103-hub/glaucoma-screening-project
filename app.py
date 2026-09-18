```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

MODEL_FILE = "glaucoma_model.pkl"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "glaucoma_model.pkl not found. "
        "Make sure the model file is in the same folder as app.py."
    )

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

glaucoma_center = np.array(model["glaucoma_center"], dtype=np.float32)
normal_center = np.array(model["normal_center"], dtype=np.float32)
image_size = tuple(model["image_size"])

print("Model loaded successfully.")
print("Model type:", model.get("model_type", "Unknown"))
print("Feature size:", len(glaucoma_center))


# --------------------------------------------------
# Feature Extraction
# Must exactly match train.py
# --------------------------------------------------

def extract_features(image):

    image = cv2.resize(image, image_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    gray = gray.astype(np.float32) / 255.0

    # Sobel gradients
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(gx, gy)

    angle = cv2.phase(
        gx,
        gy,
        angleInDegrees=True
    )

    # HOG settings
    cell_size = 8
    block_size = 2
    bins = 9

    cells_y = image.shape[0] // cell_size
    cells_x = image.shape[1] // cell_size

    histogram = np.zeros(
        (cells_y, cells_x, bins),
        dtype=np.float32
    )

    bin_width = 180.0 / bins

    # Build HOG histogram
    for cy in range(cells_y):

        for cx in range(cells_x):

            y1 = cy * cell_size
            y2 = y1 + cell_size

            x1 = cx * cell_size
            x2 = x1 + cell_size

            cell_magnitude = magnitude[y1:y2, x1:x2]
            cell_angle = angle[y1:y2, x1:x2]

            cell_angle = cell_angle % 180

            bin_index = (
                cell_angle / bin_width
            ).astype(np.int32)

            bin_index = np.clip(
                bin_index,
                0,
                bins - 1
            )

            for b in range(bins):

                histogram[cy, cx, b] = np.sum(
                    cell_magnitude[bin_index == b]
                )

    # Block normalization
    features = []

    for y in range(cells_y - block_size + 1):

        for x in range(cells_x - block_size + 1):

            block = histogram[
                y:y + block_size,
                x:x + block_size
            ].flatten()

            norm = np.sqrt(
                np.sum(block ** 2) + 1e-6
            )

            block = block / norm

            features.extend(block)

    return np.array(
        features,
        dtype=np.float32
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(features):

    features = np.array(
        features,
        dtype=np.float32
    )

    norm = np.linalg.norm(features)

    if norm > 0:
        features = features / norm

    glaucoma_distance = np.linalg.norm(
        features - glaucoma_center
    )

    normal_distance = np.linalg.norm(
        features - normal_center
    )

    # Convert distance into a model score.
    # This is NOT medical probability.
    total_distance = (
        glaucoma_distance +
        normal_distance
    )

    if total_distance > 0:
        glaucoma_score = (
            normal_distance /
            total_distance
        ) * 100
    else:
        glaucoma_score = 50.0

    if glaucoma_distance < normal_distance:

        result = "Possible Glaucoma"

    else:

        result = "Likely Normal"

    return result, round(
        float(glaucoma_score),
        2
    )


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Glaucoma Screening API is running",
        "status": "online"
    })


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400

        # Read uploaded image
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

        # Extract HOG features
        features = extract_features(image)

        # Safety check
        if len(features) != len(glaucoma_center):

            return jsonify({
                "error": "Feature size mismatch.",
                "model_feature_size": len(glaucoma_center),
                "image_feature_size": len(features)
            }), 500

        # Predict
        result, score = predict_image(features)

        # Stage cannot be estimated from binary classifier
        stage = (
            "Stage estimation unavailable. "
            "This model was trained only for "
            "glaucoma/normal classification."
        )

        # Educational information only
        risk_factors = [
            "Increasing age can increase glaucoma risk.",
            "Family history of glaucoma can increase risk.",
            "High eye pressure is an important risk factor.",
            "Some eye conditions and medications may affect glaucoma risk."
        ]

        if result == "Possible Glaucoma":

            about = (
                "The model detected image patterns "
                "that were more similar to the "
                "glaucoma examples in its training data."
            )

        else:

            about = (
                "The model detected image patterns "
                "that were more similar to the "
                "normal examples in its training data."
            )

        next_steps = [
            "This is an experimental AI screening result.",
            "A qualified eye-care professional should evaluate the eyes.",
            "A complete eye examination may include eye-pressure measurement and optic-nerve assessment."
        ]

        return jsonify({

            "result": result,

            "model_score": score,

            "stage": stage,

            "risk_factors": risk_factors,

            "about": about,

            "next_steps": next_steps,

            "model_information": {
                "model_type": model.get(
                    "model_type",
                    "HOG Nearest-Centroid Classifier"
                ),
                "training_images": 70,
                "test_images": 16,
                "test_accuracy": 75.0
            },

            "disclaimer": (
                "This system is a college project prototype "
                "and is not a medical diagnostic tool. "
                "The percentage shown is a model score, "
                "not a medical probability or diagnosis."
            )

        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
```
