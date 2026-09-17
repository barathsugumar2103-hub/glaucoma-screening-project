const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");

const resultBox = document.getElementById("resultBox");
const emptyState = document.getElementById("emptyState");
const resultText = document.getElementById("resultText");

const modelScore = document.getElementById("modelScore");
const stageText = document.getElementById("stageText");
const riskFactors = document.getElementById("riskFactors");
const aboutText = document.getElementById("aboutText");
const nextSteps = document.getElementById("nextSteps");


// ========================================
// IMAGE PREVIEW
// ========================================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {

        preview.src = URL.createObjectURL(file);

        preview.style.display = "block";

        resultBox.style.display = "none";

        if (emptyState) {
            emptyState.style.display = "block";
        }

    }

});


// ========================================
// DISPLAY LIST ITEMS
// ========================================

function fillList(element, items) {

    if (!element) {
        return;
    }

    element.innerHTML = "";

    if (!Array.isArray(items) || items.length === 0) {

        const li = document.createElement("li");

        li.textContent = "No information available.";

        element.appendChild(li);

        return;
    }

    items.forEach(function (item) {

        const li = document.createElement("li");

        li.textContent = item;

        element.appendChild(li);

    });

}


// ========================================
// ANALYZE IMAGE
// ========================================

async function analyzeImage() {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please select a retinal image first.");

        return;

    }


    // Disable button while analyzing

    analyzeButton.disabled = true;

    analyzeButton.textContent = "◌ ANALYZING...";


    // Create form data

    const formData = new FormData();

    formData.append("image", file);


    try {

        // ========================================
        // CONNECT TO FLASK RENDER API
        // ========================================

        const response = await fetch(
            "https://glaucoma-screening-project-die1.onrender.com/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        // ========================================
        // ERROR CHECK
        // ========================================

        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Analysis failed."
            );

        }


        // ========================================
        // SCREENING RESULT
        // ========================================

        resultText.textContent =
            data.result || "No result returned";


        // ========================================
        // MODEL SCORE
        // ========================================

        if (
            data.model_score !== null &&
            data.model_score !== undefined
        ) {

            modelScore.textContent =
                data.model_score + "%";

        } else {

            modelScore.textContent =
                "Unavailable";

        }


        // ========================================
        // STAGE
        // ========================================

        stageText.textContent =
            data.stage ||
            "Stage estimation unavailable.";


        // ========================================
        // ABOUT RESULT
        // ========================================

        aboutText.textContent =
            data.about ||
            "No additional information available.";


        // ========================================
        // RISK FACTORS
        // ========================================

        fillList(
            riskFactors,
            data.risk_factors
        );


        // ========================================
        // NEXT STEPS
        // ========================================

        fillList(
            nextSteps,
            data.next_steps
        );


        // ========================================
        // SHOW REPORT
        // ========================================

        if (emptyState) {
            emptyState.style.display = "none";
        }

        resultBox.style.display = "block";


        // Scroll to report on mobile

        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    } catch (error) {

        console.error(error);


        // ========================================
        // CONNECTION ERROR
        // ========================================

        resultText.textContent =
            "Unable to connect to the AI server.";


        modelScore.textContent = "—";

        stageText.textContent =
            "Unavailable";

        aboutText.textContent =
            "The AI server could not return a screening report.";


        fillList(
            riskFactors,
            []
        );

        fillList(
            nextSteps,
            [] 
        );


        if (emptyState) {
            emptyState.style.display = "none";
        }

        resultBox.style.display = "block";

    }


    // ========================================
    // RESET BUTTON
    // ========================================

    analyzeButton.disabled = false;

    analyzeButton.textContent =
        "✦ ANALYZE IMAGE";

}
