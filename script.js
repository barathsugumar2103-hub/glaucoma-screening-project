// --------------------------------------------------
// Glaucoma Detection System
// Frontend JavaScript
// --------------------------------------------------


// Backend URL

const API_URL =
    "https://glaucoma-screening-project-die1.onrender.com/predict";


// --------------------------------------------------
// Get HTML elements
// --------------------------------------------------

const imageInput =
    document.getElementById("imageInput");

const preview =
    document.getElementById("preview");

const analyzeButton =
    document.getElementById("analyzeButton");

const resultBox =
    document.getElementById("resultBox");

const resultText =
    document.getElementById("resultText");

const modelScore =
    document.getElementById("modelScore");

const riskFactors =
    document.getElementById("riskFactors");

const aboutText =
    document.getElementById("aboutText");

const nextSteps =
    document.getElementById("nextSteps");

const emptyState =
    document.getElementById("emptyState");


// --------------------------------------------------
// Store selected file
// --------------------------------------------------

let selectedFile = null;


// --------------------------------------------------
// Image selection
// --------------------------------------------------

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];


    if (!file) {

        selectedFile = null;

        analyzeButton.disabled = true;

        preview.style.display = "none";

        return;
    }


    selectedFile = file;


    // Show image preview

    const imageURL =
        URL.createObjectURL(file);

    preview.src = imageURL;

    preview.style.display = "block";


    // Enable analyze button

    analyzeButton.disabled = false;


    // Hide old result

    resultBox.style.display = "none";

    emptyState.style.display = "block";

});


// --------------------------------------------------
// Analyze image
// --------------------------------------------------

analyzeButton.addEventListener(
    "click",
    async function () {


        if (!selectedFile) {

            alert("Please select an image first.");

            return;
        }


        // Change button state

        analyzeButton.disabled = true;

        analyzeButton.textContent =
            "Analyzing...";


        // Create form data

        const formData = new FormData();

        formData.append(
            "image",
            selectedFile
        );


        try {


            // Send image to Render

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            // Read response

            const data =
                await response.json();


            // Check backend error

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Server error occurred."
                );
            }


            // --------------------------------------------------
            // Display result
            // --------------------------------------------------

            resultText.textContent =
                data.result || "Unknown";


            // --------------------------------------------------
            // Display model confidence
            // --------------------------------------------------

            if (
                data.model_score !== null &&
                data.model_score !== undefined
            ) {

                modelScore.textContent =
                    data.model_score + "%";

            } else {

                modelScore.textContent =
                    "Not available";
            }


            // --------------------------------------------------
            // Risk factors
            // --------------------------------------------------

            riskFactors.innerHTML = "";


            if (
                Array.isArray(data.risk_factors) &&
                data.risk_factors.length > 0
            ) {


                data.risk_factors.forEach(
                    function (factor) {

                        const li =
                            document.createElement("li");

                        li.textContent = factor;

                        riskFactors.appendChild(li);

                    }
                );


            } else {

                const li =
                    document.createElement("li");

                li.textContent =
                    "No information available.";

                riskFactors.appendChild(li);
            }


            // --------------------------------------------------
            // About result
            // --------------------------------------------------

            aboutText.textContent =
                data.about ||
                "No additional information available.";


            // --------------------------------------------------
            // Next steps
            // --------------------------------------------------

            nextSteps.innerHTML = "";


            if (
                Array.isArray(data.next_steps) &&
                data.next_steps.length > 0
            ) {


                data.next_steps.forEach(
                    function (step) {

                        const li =
                            document.createElement("li");

                        li.textContent = step;

                        nextSteps.appendChild(li);

                    }
                );


            } else {

                const li =
                    document.createElement("li");

                li.textContent =
                    "No information available.";

                nextSteps.appendChild(li);
            }


            // --------------------------------------------------
            // Show report
            // --------------------------------------------------

            emptyState.style.display =
                "none";

            resultBox.style.display =
                "block";


        }

        catch (error) {


            console.error(
                "Analysis error:",
                error
            );


            alert(
                "Unable to analyze the image.\n\n" +
                error.message
            );

        }


        finally {


            // Restore button

            analyzeButton.disabled =
                false;

            analyzeButton.textContent =
                "Analyze Image";

        }

    }
);
