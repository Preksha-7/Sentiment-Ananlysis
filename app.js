document
  .getElementById("analyzeBtn")
  .addEventListener("click", async function () {
    const reviewText = document.getElementById("review").value;
    const response = await fetch(
      "https://your-backend-url.herokuapp.com/predict",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          review: reviewText,
        }),
      }
    );

    const result = await response.json();
    document.getElementById(
      "result"
    ).innerText = `Sentiment: ${result.sentiment}`;
  });
