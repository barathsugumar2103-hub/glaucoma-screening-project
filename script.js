const API_URL = "https://glaucoma-screening-project-die1.onrender.com/predict";

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const screeningPercentage = document.getElementById("screeningPercentage");
const testAccuracy = document.getElementById("testAccuracy");
const riskFactors = document.getElementById("riskFactors");
const aboutText = document.getElementById("aboutText");
const nextSteps = document.getElementById("nextSteps");

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    preview.removeAttribute("src");
    preview.style.display = "none";
    analyzeButton.disabled = true;
    return;
  }
  if (!file.type.startsWith("image/")) {
    alert("Please choose an image file.");
    imageInput.value = "";
    preview.style.display = "none";
    analyzeButton.disabled = true;
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  analyzeButton.disabled = false;
  resultBox.style.display = "none";
});

analyzeButton.addEventListener("click", async () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);

    const percentage = typeof data.screening_percentage === "number"
      ? data.screening_percentage
      : data.model_score;

    resultText.textContent = data.result || "No result returned";
    screeningPercentage.textContent = typeof percentage === "number"
      ? `${percentage.toFixed(2)}%`
      : "N/A";
    testAccuracy.textContent = data.model_information && typeof data.model_information.test_accuracy === "number"\n      ? `${data.model_information.test_accuracy.toFixed(1)}%`\n      : "N/A";
    aboutText.textContent = data.about || "No additional information available.";

    riskFactors.replaceChildren();
    (Array.isArray(data.risk_factors) ? data.risk_factors : []).forEach((factor) => {
      const li = document.createElement("li");
      li.textContent = factor;
      riskFactors.appendChild(li);
    });

    nextSteps.replaceChildren();
    (Array.isArray(data.next_steps) ? data.next_steps : []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      nextSteps.appendChild(li);
    });

    resultBox.style.display = "block";
    resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error("Screening request failed:", error);
    resultBox.style.display = "block";
    resultText.textContent = "Could not analyze image";
    screeningPercentage.textContent = "N/A";
    testAccuracy.textContent = "N/A";
    aboutText.textContent = "The request did not complete.";
    riskFactors.replaceChildren();
    nextSteps.replaceChildren();
    const li = document.createElement("li");
    li.textContent = error.message || "Unknown error";
    nextSteps.appendChild(li);
  } finally {
    analyzeButton.disabled = !imageInput.files || !imageInput.files[0];
    analyzeButton.textContent = "⌕  Analyze Image";
  }
});