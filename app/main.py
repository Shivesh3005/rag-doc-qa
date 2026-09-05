from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router

app = FastAPI(title="RAG Document Q&A Assistant")
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
