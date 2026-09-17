from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import joblib
import numpy as np

from skimage.feature import hog


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)

CORS(app)


# ==================================================
# LOAD AI MODEL
# ==================================================

MODEL_FILE = "glaucoma_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    print("AI model loaded successfully.")

except Exception as e:
    model = None
    print("Error loading AI model:", e)


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (128, 128)
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ==================================================
# HOME ROUTE
# ==================================================

@app.route("/")
def home():

    return jsonify({

        "message":
        "Glaucoma Screening API is running - V2"

    })


# ==================================================
# PREDICTION ROUTE
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ----------------------------------------------
    # CHECK IMAGE
    # ----------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "error":
            "No image uploaded"

        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({

            "error":
            "No image selected"

        }), 400


    # ----------------------------------------------
    # READ IMAGE
    # ----------------------------------------------

    image_bytes = file.read()


    # ----------------------------------------------
    # EXTRACT FEATURES
    # ----------------------------------------------

    features = extract_features(
        image_bytes
    )


    if features is None:

        return jsonify({

            "error":
            "Invalid image"

        }), 400


    # ----------------------------------------------
    # CHECK MODEL
    # ----------------------------------------------

    if model is None:

        return jsonify({

            "error":
            "AI model could not be loaded"

        }), 500


    # ----------------------------------------------
    # AI PREDICTION
    # ----------------------------------------------

    prediction = model.predict(
        [features]
    )[0]


    # ==================================================
    # MODEL SCORE
    # ==================================================

    model_score = None

    try:

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                [features]
            )[0]


            if len(probabilities) > 1:

                model_score = round(
                    float(probabilities[1]) * 100,
                    2
                )

    except Exception as e:

        print(
            "Model score error:",
            e
        )

        model_score = None


    # ==================================================
    # GLAUCOMA RESULT
    # ==================================================

    if prediction == 1:

        result = "Possible Glaucoma"


        stage = (
            "Stage estimation unavailable."
        )


        risk_factors = [

            "Increasing age can be associated "
            "with glaucoma risk.",

            "Family history of glaucoma may "
            "increase risk.",

            "High eye pressure is an important "
            "clinical risk factor.",

            "Some medical conditions can be "
            "associated with increased risk."

        ]


        about = (

            "The prototype AI model detected "
            "image features associated with "
            "the glaucoma class."

        )


        next_steps = [

            "Consider having a professional "
            "eye examination.",

            "An eye specialist can perform "
            "appropriate glaucoma tests.",

            "Do not use this prototype result "
            "as a medical diagnosis."

        ]


    # ==================================================
    # NORMAL RESULT
    # ==================================================

    else:

        result = "Likely Normal"


        stage = (
            "No glaucoma stage estimated."
        )


        risk_factors = [

            "A normal prototype result does "
            "not rule out glaucoma.",

            "Regular eye examinations can help "
            "detect eye problems early."

        ]


        about = (

            "The prototype AI model classified "
            "this image as belonging to the "
            "normal class."

        )


        next_steps = [

            "Continue routine eye examinations.",

            "Seek professional evaluation if "
            "you have eye-related concerns.",

            "Do not use this prototype result "
            "as a medical diagnosis."

        ]


    # ==================================================
    # COMPLETE RESPONSE
    # ==================================================

    response = {

        "result": result,

        "model_score": model_score,

        "stage": stage,

        "risk_factors": risk_factors,

        "about": about,

        "next_steps": next_steps,

        "disclaimer": (

            "This is a college-project AI "
            "screening prototype and not a "
            "medical diagnostic tool."

        )

    }


    print(
        "Sending response:",
        response
    )


    return jsonify(response)


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
