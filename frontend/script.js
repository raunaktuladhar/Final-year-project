document.getElementById("profile-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append("screen_name", document.getElementById("screen_name").value.trim());
    formData.append("description", document.getElementById("description").value.trim());
    formData.append("profile-pic", document.getElementById("profile-pic").files[0]);
    formData.append("followers_count", document.getElementById("followers_count").value || 0);
    formData.append("friends_count", document.getElementById("friends_count").value || 0);
    formData.append("statuses_count", document.getElementById("statuses_count").value || 0);
    formData.append("verified", document.getElementById("verified").value);

    if (!formData.get("screen_name")) {
        alert("Twitter handle is required.");
        return;
    }

    document.getElementById("result-section").style.display = "block";
    document.getElementById("prediction-output").innerText = "Analyzing...";
    document.getElementById("prediction-output").classList.add("loading");

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        });
        const result = await response.json();
        document.getElementById("prediction-output").classList.remove("loading");
        document.getElementById("prediction-output").classList.add(result.prediction);
        document.getElementById("prediction-output").innerText = 
            `Prediction: This profile is ${result.prediction.toUpperCase()} ${result.prediction === "real" ? "✅" : "❌"} (confidence: ${result.confidence.toFixed(2)}%)`;

        const ctx = document.getElementById("confidence-chart").getContext("2d");
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Real", "Fake"],
                datasets: [{
                    label: "Prediction Confidence",
                    data: [result.prediction === "real" ? result.confidence : 100 - result.confidence, 
                           result.prediction === "fake" ? result.confidence : 100 - result.confidence],
                    backgroundColor: ["#16a34a", "#dc2626"],
                    borderColor: ["#14532d", "#991b1b"],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true, max: 100, title: { display: true, text: "Confidence (%)" } },
                    x: { title: { display: true, text: "Prediction" } }
                },
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: "Profile Prediction Confidence" }
                }
            }
        });
    } catch (error) {
        document.getElementById("prediction-output").classList.remove("loading");
        document.getElementById("prediction-output").classList.add("error");
        document.getElementById("prediction-output").innerText = "Error: Failed to analyze profile.";
        console.error("API error:", error);
    }
});