import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.loader import process_file
from app.core.vectorstore import build_vectorstore, load_vectorstore, collection_exists
from app.core.rag_chain import answer_question

router = APIRouter()
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    collection_id: str
    question: str


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    collection_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{collection_id}{ext}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = process_file(file_path)
        build_vectorstore(chunks, collection_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"collection_id": collection_id, "filename": file.filename, "chunks": len(chunks)}


@router.post("/query")
async def query_document(request: QueryRequest):
    if not collection_exists(request.collection_id):
        raise HTTPException(status_code=404, detail="Collection not found")

    store = load_vectorstore(request.collection_id)
    retriever_docs = store.similarity_search(request.question, k=4)

    if not retriever_docs:
        raise HTTPException(status_code=404, detail="No relevant content found")

    result = answer_question(request.question, retriever_docs)
    return result
