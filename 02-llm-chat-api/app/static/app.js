const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messagesContainer = document.getElementById("messages");

const history = [];


function addMessage(role, content) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;

    message.appendChild(bubble);
    messagesContainer.appendChild(message);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();

    if (!message) {
        return;
    }

    addMessage("user", message);

    input.value = "";
    input.disabled = true;

    try {
        const response = await fetch("/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                message: message,
                history: history,
            }),
        });

        if (!response.ok) {
            throw new Error("Request failed");
        }

        const data = await response.json();

        addMessage("assistant", data.answer);

        history.push({
            role: "user",
            content: message,
        });

        history.push({
            role: "assistant",
            content: data.answer,
        });

    } catch (error) {
        addMessage(
            "assistant",
            "Sorry, something went wrong."
        );

    } finally {
        input.disabled = false;
        input.focus();
    }
});