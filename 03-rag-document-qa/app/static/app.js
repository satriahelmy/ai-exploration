const documentList = document.getElementById("document-list");

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");


uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/documents", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        alert(error.detail || "Failed to upload document.");
        return;
    }

    fileInput.value = "";

    await loadDocuments();
});

const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question-input");
const chatMessages = document.getElementById("chat-messages");


function addMessage(role, text) {
    const message = document.createElement("div");

    message.className = `message ${role}`;

    const paragraph = document.createElement("p");
    paragraph.textContent = text;

    message.appendChild(paragraph);
    chatMessages.appendChild(message);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


questionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    addMessage("user", question);

    questionInput.value = "";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question: question,
            }),
        });

        if (!response.ok) {
            const error = await response.json();

            addMessage(
                "assistant",
                error.detail || "Failed to generate an answer."
            );

            return;
        }

        const data = await response.json();

        addMessage(
            "assistant",
            data.answer
        );

    } catch (error) {
        addMessage(
            "assistant",
            "Unable to connect to the server."
        );
    }
});

async function loadDocuments() {
    try {
        const response = await fetch("/documents");

        if (!response.ok) {
            throw new Error("Failed to load documents.");
        }

        const documents = await response.json();

        documentList.innerHTML = "";

        if (documents.length === 0) {
            documentList.innerHTML = "<p>No documents loaded.</p>";
            return;
        }

        documents.forEach((doc) => {
            const item = document.createElement("div");

            item.className = "document-item";

            item.innerHTML = `
                <span>${doc.filename}</span>
                <button>Delete</button>
            `;

            const deleteButton = item.querySelector("button");

            deleteButton.addEventListener("click", async () => {
                await deleteDocument(doc.document_id);
            });

            documentList.appendChild(item);
        });

    } catch (error) {
        documentList.innerHTML =
            "<p>Failed to load documents.</p>";
    }
}


async function deleteDocument(documentId) {
    const response = await fetch(
        `/documents/${documentId}`,
        {
            method: "DELETE",
        }
    );

    if (!response.ok) {
        alert("Failed to delete document.");
        return;
    }

    await loadDocuments();
}


loadDocuments();