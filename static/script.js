async function submitQuestion() {
      const input = document.getElementById("question");
      const question = input.value;
      const answerDisplay = document.getElementById("answer");

      answerDisplay.innerText = "Thinking...";

      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      
      const data = await res.json();
      document.getElementById("answer").innerText = data.answer;

       input.value = "";
}