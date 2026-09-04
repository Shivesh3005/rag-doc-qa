# RAG Document Q&A Assistant

A retrieval-augmented generation system that lets users upload documents (PDF, DOCX, TXT) and ask questions, receiving grounded, citation-backed answers.

## Architecture

- **API layer**: FastAPI, handling file upload and query endpoints
- **Document processing**: LangChain loaders + recursive chunking
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`), run locally, no external API cost
- **Vector store**: ChromaDB, persisted per document collection
- **LLM**: Groq-hosted `openai/gpt-oss-120b` for fast, low-latency answer generation with source citations

## Features

- Upload PDF, DOCX, or TXT documents
- Automatic chunking and embedding into a per-document vector collection
- Ask natural-language questions and get answers grounded in the uploaded content
- Every answer includes source citations with page numbers and text snippets

## Setup

```bash
git clone <repo-url>
cd rag-doc-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

Get a free key at https://console.groq.com/keys

## Running

```bash
uvicorn app.main:app --port 8000
```

## API

### Upload a document

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@document.pdf"
```

Returns a `collection_id` used for querying.

### Query a document

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"collection_id": "<id>", "question": "your question"}'
```

Returns an answer with cited sources.

## Tech Stack

Python, FastAPI, LangChain, ChromaDB, Sentence-Transformers, Groq
