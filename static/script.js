async function submitQuestion() {
    const input = document.getElementById("question");
    const question = input.value;
    const answerDisplay = document.getElementById("answer");

    answerDisplay.innerText = "Thinking...";

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
    } catch (error) {
        console.error("Error during fetch:", error);
        answerDisplay.innerText = "Failed to get an answer.";
    }

    input.value = "";
}
