async function submitQuestion() {
  const input = document.getElementById("question");
  const question = input.value.trim();
  const answerDisplay = document.getElementById("answer");
  const loader = document.getElementById("loader");
  const responseBox = answerDisplay.closest(".response-box");

  if (!question) {
    answerDisplay.innerText = "Please enter a question.";
    return;
  }
  loader.classList.remove("hidden");
  answerDisplay.innerText = "";
  answerDisplay.classList.remove("fade-in");

  try {
    const res = await fetch("http://127.0.0.1:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    console.log("Received response:", data);

    if (data.answer) {
      answerDisplay.innerText = data.answer;
    } else {
      answerDisplay.innerText = "No answer returned.";
    }

    // Animate answer appearance
    void answerDisplay.offsetWidth;
    answerDisplay.classList.add("fade-in");

  } catch (error) {
    console.error("Error during fetch:", error);
    answerDisplay.innerText = "Oops! Something went wrong. Please try again.";
  } finally {
    loader.classList.add("hidden");
    input.value ="";
  }
}