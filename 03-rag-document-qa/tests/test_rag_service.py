from unittest.mock import MagicMock

from app.services.rag_service import RAGService


def test_rag_service_returns_llm_answer():
    retrieval_service = MagicMock()
    llm_client = MagicMock()

    result = MagicMock()
    result.payload = {
        "filename": "employee_handbook.pdf",
        "page": 2,
        "text": "Employees receive 18 working days of paid annual leave.",
    }

    retrieval_service.retrieve.return_value = [result]

    llm_client.generate.return_value = (
        "Employees receive 18 working days of paid annual leave. "
        "[employee_handbook.pdf - Page 2]"
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )

    answer = rag_service.ask(
        "How many days of annual leave do employees receive?"
    )

    assert "18 working days" in answer
    assert "employee_handbook.pdf" in answer

def test_rag_service_calls_retrieval_with_question():
    retrieval_service = MagicMock()
    llm_client = MagicMock()

    retrieval_service.retrieve.return_value = []

    llm_client.generate.return_value = (
        "I don't know based on the provided context."
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
    )

    question = "Who is the CEO?"

    rag_service.ask(question)

    retrieval_service.retrieve.assert_called_once_with(
        question
    )

    llm_client.generate.assert_called_once()