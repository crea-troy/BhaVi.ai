// frontend/script.js
function askAgent() {
  const input = document.getElementById("userInput").value;
  const responseArea = document.getElementById("responseArea");

  responseArea.textContent = "Thinking...";

  // Later: Connect this to your Flask backend
  fetch("https://your-backend-api.com/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: input })
  })
  .then(response => response.json())
  .then(data => {
    responseArea.textContent = data.response;
  })
  .catch(err => {
    responseArea.textContent = "Error: " + err.message;
  });
}

