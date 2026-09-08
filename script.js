const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        resultBox.style.display = "none";
    }

});

async function analyzeImage() {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    const formData = new FormData();
    formData.append("image", file);

    try {

        const response = await fetch(
            "YOUR_RENDER_URL/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (data.error) {
            resultText.textContent = data.error;
        } else {
            resultText.textContent = data.result;
        }

        resultBox.style.display = "block";

    } catch (error) {

        resultText.textContent =
            "Unable to connect to the AI server.";

        resultBox.style.display = "block";

        console.error(error);
    }

    analyzeButton.disabled = false;
    analyzeButton.textContent = "🔍 Analyze Image";
}
