async function handlePredict() {
    const textInput = document.getElementById("textInput");
    const imageInput = document.getElementById("imageInput");
    const text = textInput.value.trim();
    const image = imageInput.files[0];
    const resultDiv = document.getElementById("result");
    const spinner = document.getElementById("loadingSpinner");
    const downloadBtn = document.getElementById("downloadJson");

    resultDiv.innerHTML = "";
    spinner.style.display = "none";
    textInput.style.border = "";
    imageInput.style.border = "";

    if (!text || !image) {
        if (!text) textInput.style.border = "2px solid #f87171";
        if (!image) imageInput.style.border = "2px solid #f87171";
        resultDiv.innerHTML = "<p style='color: #f87171;'>Please enter text and upload an image.</p>";
        downloadBtn.style.display = "none";
        return;
    }

    const formData = new FormData();
    formData.append("text", text);
    formData.append("image", image);

    spinner.style.display = "block";
    downloadBtn.style.display = "none";

    const startTime = performance.now();

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const endTime = performance.now();
        const predictionTime = ((endTime - startTime) / 1000).toFixed(2);

        const data = await response.json();

        if (!response.ok) {
            spinner.style.display = "none";
            resultDiv.innerHTML = `<p style='color: #f87171;'>Error: ${data.error}</p>`;
            return;
        }

        const label = data.label;
        const [fake, real] = data.softmax;
        const fakePercent = (fake * 100).toFixed(2);
        const realPercent = (real * 100).toFixed(2);

        const verdict = label === "fake"
            ? `🛑 <span class="glow-red">Fake News Detected</span>`
            : `✅ <span class="glow-green">News Appears Real</span>`;

        resultDiv.innerHTML = `
            <h3>${verdict}</h3>
            <p>Fake: ${fakePercent}%</p>
            <p>Real: ${realPercent}%</p>
            <p>🕒 Prediction Time: ${predictionTime} seconds</p>
            <canvas id="resultChart" width="180" height="180"></canvas>
        `;

        const ctx = document.getElementById("resultChart").getContext("2d");
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Fake', 'Real'],
                datasets: [{
                    data: [fakePercent, realPercent],
                    backgroundColor: ['#ef4444', '#22c55e']
                }]
            },
            options: {
                responsive: false,
                cutout: '50%',
                plugins: {
                    legend: {
                        labels: {
                            color: "#ffffff"
                        }
                    }
                }
            }
        });

        window.latestJson = {
            text_input: text,
            predicted_label: label,
            softmax: {
                fake: fakePercent + "%",
                real: realPercent + "%"
            },
            prediction_time_seconds: predictionTime
        };

        downloadBtn.style.display = "inline-block";

    } catch (error) {
        resultDiv.innerHTML = `<p style='color: #f87171;'>Request failed: ${error.message}</p>`;
    }

    spinner.style.display = "none";
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

function handleReset() {
    const textInput = document.getElementById("textInput");
    const imageInput = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const spinner = document.getElementById("loadingSpinner");
    const downloadBtn = document.getElementById("downloadJson");
    const preview = document.getElementById("imagePreview");

    textInput.value = "";
    imageInput.value = "";
    resultDiv.innerHTML = "";
    spinner.style.display = "none";
    downloadBtn.style.display = "none";
    textInput.style.border = "";
    imageInput.style.border = "";
    preview.src = "";
    preview.style.display = "none";
}

function downloadJSON() {
    if (!window.latestJson) return;
    const blob = new Blob([JSON.stringify(window.latestJson, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "detection_result.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Image preview handler
document.getElementById("imageInput").addEventListener("change", function () {
    const preview = document.getElementById("imagePreview");
    const file = this.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
        preview.style.display = "none";
    }
});
