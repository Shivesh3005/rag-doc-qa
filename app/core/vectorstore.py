import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from app.core.config import settings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vectorstore(documents: List[Document], collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return store


def load_vectorstore(collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def collection_exists(collection_name: str) -> bool:
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    return os.path.isdir(persist_dir)
