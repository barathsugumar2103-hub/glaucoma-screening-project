```javascript
const API_URL =
    "https://glaucoma-screening-project-die1.onrender.com/predict";


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

const stageText =
    document.getElementById("stageText");

const riskFactors =
    document.getElementById("riskFactors");

const aboutText =
    document.getElementById("aboutText");

const nextSteps =
    document.getElementById("nextSteps");


// --------------------------------------------------
// Image Selection
// --------------------------------------------------

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (!file) {

            preview.style.display =
                "none";

            analyzeButton.disabled =
                true;

            return;
        }

        preview.src =
            URL.createObjectURL(file);

        preview.style.display =
            "block";

        analyzeButton.disabled =
            false;

        resultBox.style.display =
            "none";
    }
);


// --------------------------------------------------
// Analyze Image
// --------------------------------------------------

analyzeButton.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];

        if (!file) {

            alert(
                "Please select an eye image first."
            );

            return;
        }

        analyzeButton.disabled =
            true;

        analyzeButton.textContent =
            "Analyzing...";


        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );


        try {

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );
            }


            // Main result
            resultText.textContent =
                data.result;


            // Model score
            if (
                data.model_score !==
                undefined
            ) {

                modelScore.textContent =
                    data.model_score.toFixed(2)
                    + "%";

            } else {

                modelScore.textContent =
                    "N/A";
            }


            // Stage
            stageText.textContent =
                data.stage ||
                "Not available";


            // About
            aboutText.textContent =
                data.about ||
                "No additional information available.";


            // Risk factors
            riskFactors.innerHTML =
                "";


            if (
                Array.isArray(
                    data.risk_factors
                )
            ) {

                data.risk_factors.forEach(
                    function (factor) {

                        const li =
                            document.createElement(
                                "li"
                            );

                        li.textContent =
                            factor;

                        riskFactors.appendChild(
                            li
                        );
                    }
                );
            }


            // Next steps
            nextSteps.innerHTML =
                "";


            if (
                Array.isArray(
                    data.next_steps
                )
            ) {

                data.next_steps.forEach(
                    function (step) {

                        const li =
                            document.createElement(
                                "li"
                            );

                        li.textContent =
                            step;

                        nextSteps.appendChild(
                            li
                        );
                    }
                );
            }


            // Show result
            resultBox.style.display =
                "block";


            // Scroll to result
            resultBox.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });


        } catch (error) {

            console.error(error);


            resultBox.style.display =
                "block";


            resultText.textContent =
                "Error";


            modelScore.textContent =
                "N/A";


            stageText.textContent =
                "Unable to analyze";


            aboutText.textContent =
                "Something went wrong while connecting to the AI screening server.";


            riskFactors.innerHTML =
                "";


            nextSteps.innerHTML =
                "";


            const li =
                document.createElement(
                    "li"
                );

            li.textContent =
                error.message;


            nextSteps.appendChild(
                li
            );


        } finally {

            analyzeButton.disabled =
                false;

            analyzeButton.textContent =
                "Analyze Image";
        }
    }
);
```
