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


// ------------------------------------------
// Image preview
// ------------------------------------------

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    resultBox.style.display = "none";

    if (emptyState) {
        emptyState.style.display = "block";
    }
});


// ------------------------------------------
// Create list items
// ------------------------------------------

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


// ------------------------------------------
// Analyze image
// ------------------------------------------

async function analyzeImage() {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please select a retinal image first.");

        return;
    }


    analyzeButton.disabled = true;

    analyzeButton.textContent = "◌ ANALYZING...";


    const formData = new FormData();

    formData.append("image", file);


    try {

        const response = await fetch(
            "https://glaucoma-screening-project-die1.onrender.com/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Analysis failed."
            );
        }


        // ------------------------------------------
        // Screening result
        // ------------------------------------------

        resultText.textContent =
            data.result || "No result returned";


        // ------------------------------------------
        // Model score
        // ------------------------------------------

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


        // ------------------------------------------
        // Stage
        // ------------------------------------------

        stageText.textContent =
            data.stage ||
            "Stage estimation unavailable.";


        // ------------------------------------------
        // Risk factors
        // ------------------------------------------

        fillList(
            riskFactors,
            data.risk_factors
        );


        // ------------------------------------------
        // About result
        // ------------------------------------------

        aboutText.textContent =
            data.about ||
            "No additional information available.";


        // ------------------------------------------
        // Next steps
        // ------------------------------------------

        fillList(
            nextSteps,
            data.next_steps
        );


        // Hide empty state
        if (emptyState) {

            emptyState.style.display = "none";

        }


        // Show result
        resultBox.style.display = "block";


        // Scroll to result
        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    } catch (error) {

        console.error(error);


        resultText.textContent =
            "Unable to connect to the AI server.";


        modelScore.textContent =
            "—";


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


    analyzeButton.disabled = false;

    analyzeButton.textContent =
        "✦ ANALYZE IMAGE";
}
