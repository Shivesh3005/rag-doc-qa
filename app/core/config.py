from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    chroma_persist_dir: str = "./chroma_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chat_model: str = "openai/gpt-oss-120b"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
