document.addEventListener("DOMContentLoaded", function () {
  const textInput = document.getElementById("text-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-btn");
  const pasteBtn = document.getElementById("paste-btn");
  const resultContainer = document.getElementById("result-container");
  const loader = document.getElementById("loader");
  const results = document.getElementById("results");
  const errorMessage = document.getElementById("error-message");
  const sentimentLabel = document.getElementById("sentiment-label");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceFill = document.getElementById("confidence-fill");
  const sentimentIcon = document.getElementById("sentiment-icon");
  const sentimentIconFace = document.getElementById("sentiment-icon-face");
  const factorsList = document.getElementById("factors-list");

  // Function to analyze sentiment
  function analyzeSentiment() {
    const text = textInput.value.trim();

    if (!text) {
      showError("Please enter some text to analyze");
      return;
    }

    // Show loader, hide results and error
    loader.classList.remove("hidden");
    results.classList.add("hidden");
    errorMessage.classList.add("hidden");

    // Make API request
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text }),
    })
      .then((response) => response.json())
      .then((data) => {
        loader.classList.add("hidden");

        if (data.error) {
          showError(data.error);
          return;
        }

        // Show results
        results.classList.remove("hidden");

        // Update sentiment label
        sentimentLabel.textContent = data.sentiment.toUpperCase();
        sentimentLabel.style.color =
          data.sentiment === "positive"
            ? "var(--accent-positive)"
            : "var(--accent-negative)";

        // Update confidence
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceFill.style.backgroundColor =
          data.sentiment === "positive"
            ? "var(--accent-positive)"
            : "var(--accent-negative)";

        // Update sentiment icon
        sentimentIconFace.className =
          data.sentiment === "positive" ? "far fa-smile" : "far fa-frown";
        sentimentIcon.style.color =
          data.sentiment === "positive"
            ? "var(--accent-positive)"
            : "var(--accent-negative)";

        // Update key factors
        factorsList.innerHTML = "";
        if (data.top_contributions && data.top_contributions.length > 0) {
          data.top_contributions.forEach((contribution) => {
            const term = contribution[0];
            const value = contribution[1];
            const isPositive = value > 0;

            const factorItem = document.createElement("div");
            factorItem.className = "factor-item";

            const factorTerm = document.createElement("span");
            factorTerm.className = "factor-term";
            factorTerm.textContent = term;

            const factorValue = document.createElement("span");
            factorValue.className = `factor-value ${
              isPositive ? "positive" : "negative"
            }`;
            factorValue.textContent = isPositive
              ? `+${value.toFixed(2)}`
              : value.toFixed(2);

            factorItem.appendChild(factorTerm);
            factorItem.appendChild(factorValue);
            factorsList.appendChild(factorItem);
          });
        } else {
          factorsList.innerHTML = "<p>No significant factors found</p>";
        }
      })
      .catch((error) => {
        loader.classList.add("hidden");
        showError(
          "An error occurred while analyzing the text. Please try again."
        );
        console.error("Error:", error);
      });
  }

  // Function to show error message
  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
    results.classList.add("hidden");
  }

  // Event listeners
  analyzeBtn.addEventListener("click", analyzeSentiment);

  clearBtn.addEventListener("click", function () {
    textInput.value = "";
    results.classList.add("hidden");
    errorMessage.classList.add("hidden");
  });

  pasteBtn.addEventListener("click", async function () {
    try {
      const text = await navigator.clipboard.readText();
      textInput.value = text;
    } catch (err) {
      showError("Failed to read from clipboard. Please paste manually.");
    }
  });

  // Allow Enter key to submit
  textInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && event.ctrlKey) {
      analyzeSentiment();
    }
  });

  // Create confidence-fill element if it doesn't exist
  if (!document.getElementById("confidence-fill")) {
    const fill = document.createElement("div");
    fill.id = "confidence-fill";
    fill.className = "confidence-fill";
    document.querySelector(".confidence-bar").appendChild(fill);
  }
});
