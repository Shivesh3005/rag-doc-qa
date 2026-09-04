from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import Document
from app.core.config import settings

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question using only the provided context excerpts.
Cite sources using [Source N] notation matching the excerpt numbers given.
If the context does not contain enough information to answer, say so explicitly instead of guessing."""


def format_context(documents: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}] (file: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def get_llm():
    return ChatGroq(
        model=settings.chat_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def answer_question(question: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    context = format_context(retrieved_docs)
    llm = get_llm()
    user_message = f"Context excerpts:\n\n{context}\n\nQuestion: {question}"
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "snippet": doc.page_content[:200],
        }
        for doc in retrieved_docs
    ]
    return {"answer": response.content, "sources": sources}
