class RAGService:
    def __init__(
        self,
        retrieval_service,
        llm_client,
    ):
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client

    def ask(self, query):
        results = self.retrieval_service.retrieve(query)

        context_parts = []

        for result in results:
            payload = result.payload

            context_parts.append(
                f"[{payload['filename']} - Page {payload['page']}]\n"
                f"{payload['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a question-answering assistant.

Answer the question using only the provided context.

Rules:
- Do not use outside knowledge.
- If the answer is not supported by the context, say:
  "I don't know based on the provided context."
- Do not make up information.
- Keep the answer concise and factual.
- Cite the document and page that support the answer.
- Only cite sources that actually support the answer.

Context:
{context}

Question:
{query}

Answer:
"""

        return self.llm_client.generate(prompt)