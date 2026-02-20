document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const btnText = predictBtn.querySelector(".btn-text");
    const btnLoading = predictBtn.querySelector(".btn-loading");
    const resultCard = document.getElementById("result-card");

    const dateInput = document.getElementById("date_of_reservation");
    dateInput.value = new Date().toISOString().split("T")[0];

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!validateForm()) return;
        setLoading(true);

        try {
            const formData = new FormData(form);
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (data.success) {
                showResult(data);
            } else {
                showError(data.error || "An unknown error occurred.");
            }
        } catch (err) {
            showError("Network error. Please make sure the server is running.");
        } finally {
            setLoading(false);
        }
    });

    function validateForm() {
        if (!form.checkValidity()) {
            form.reportValidity();
            return false;
        }

        const leadTime = parseInt(document.getElementById("lead_time").value);
        if (leadTime < 0) {
            alert("Lead time cannot be negative.");
            return false;
        }

        const avgPrice = parseFloat(document.getElementById("average_price").value);
        if (avgPrice < 0) {
            alert("Average price cannot be negative.");
            return false;
        }

        return true;
    }

    function setLoading(isLoading) {
        predictBtn.disabled = isLoading;
        btnText.classList.toggle("hidden", isLoading);
        btnLoading.classList.toggle("hidden", !isLoading);
    }

    function showResult(data) {
        resultCard.classList.remove("hidden");
        document.getElementById("result-content").classList.remove("hidden");
        document.getElementById("error-content").classList.add("hidden");

        const badge = document.getElementById("prediction-badge");
        badge.textContent = data.prediction;
        badge.className = "prediction-badge " +
            (data.prediction === "Canceled" ? "canceled" : "not-canceled");

        setTimeout(() => {
            document.getElementById("bar-not-cancel").style.width =
                data.not_cancel_probability + "%";
            document.getElementById("bar-cancel").style.width =
                data.cancel_probability + "%";
        }, 100);

        document.getElementById("prob-not-cancel").textContent =
            data.not_cancel_probability.toFixed(1) + "%";
        document.getElementById("prob-cancel").textContent =
            data.cancel_probability.toFixed(1) + "%";

        document.getElementById("confidence-text").textContent =
            `Model confidence: ${data.confidence.toFixed(1)}%`;

        resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    function showError(message) {
        resultCard.classList.remove("hidden");
        document.getElementById("result-content").classList.add("hidden");
        document.getElementById("error-content").classList.remove("hidden");
        document.getElementById("error-message").textContent = message;
        resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }
});
