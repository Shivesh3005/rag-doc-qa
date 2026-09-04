from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="RAG Document Q&A Assistant")
app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
