# AI Document Chat Web App

Full-stack AI document chat application. Upload PDFs or text files, then ask questions about them using Retrieval-Augmented Generation (RAG). Includes JWT authentication, per-user document storage, chat history, and a FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## Overview

This project implements a full-stack document chat application with:

1. **User authentication** — register and login with JWT tokens
2. **Document upload** — PDF or text files, per-user isolation
3. **Text extraction** — PyPDF2 for PDFs, direct decode for text
4. **RAG pipeline** — chunking, TF-IDF vectorization, cosine similarity retrieval
5. **LLM answers** — with context-only responses (no hallucination beyond sources)
6. **Chat history** — every Q&A saved per user
7. **Interactive API** — Swagger UI auto-generated

If OpenAI is configured, the `/chat` endpoint uses real LLM inference. Otherwise it falls back to a template answer built from the retrieved context, keeping the pipeline fully functional.

## Problem Statement

Users need a fast way to extract answers from large PDFs without reading every page:

- Manual reading is slow
- Ctrl+F only matches exact strings, not semantic meaning
- Generic LLM chatbots don't know about your private documents
- Cloud-based tools raise privacy concerns

## Solution

A per-user RAG system that:

- Isolates documents per user via JWT auth
- Chunks and indexes text on demand
- Retrieves top-K relevant chunks for each question
- Answers only from retrieved context (grounded, with source previews)
- Saves full chat history per user
- Falls back gracefully when the LLM is unavailable

## Architecture

User (JWT)
    |
    v
+-----------+   +-------------+
| Register  |   |   Upload    |  POST /upload
|  Login    |   |  PDF/TXT    |
+-----+-----+   +------+------+
      |                |
      v                v
   SQLite  <------ Extract text (PyPDF2)
      |                |
      |                v
      |          Chunk into 400-char segments
      |                |
      |                v
      |         TF-IDF vectorize (or ChromaDB)
      |                |
      +--------+-------+
               |
               v
        POST /chat  <--- User question
               |
               v
        Cosine similarity -> top-K chunks
               |
               v
        LLM answer (OpenAI or template)
               |
               v
        Response + sources + saved to history

## Tech Stack

| Category | Technologies |
|----------|-------------|
| API | FastAPI, Uvicorn, Pydantic |
| Auth | JWT (python-jose), passlib + bcrypt |
| Storage | SQLite (users, documents, chat history) |
| RAG | Scikit-learn TF-IDF + cosine similarity |
| PDF Parsing | PyPDF2 |
| Optional LLM | OpenAI via langchain-openai |
| Testing | pytest, httpx |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

## Project Structure

ai-document-chat/
├── src/
│   ├── __init__.py
│   ├── auth.py                # JWT + password hashing
│   ├── storage.py             # SQLite users/documents/chat
│   └── rag.py                 # Chunking + TF-IDF retrieval + answer
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI endpoints
├── tests/
│   ├── __init__.py
│   └── test_api.py            # pytest tests
├── data/
│   ├── documents/             # Uploaded files (git-ignored)
│   └── .gitkeep
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-optional.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md

## Quick Start

### 1. Clone

git clone https://github.com/sumit966/ai-document-chat.git
cd ai-document-chat

### 2. Virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. (Optional) Install PDF + LLM stack

pip install -r requirements-optional.txt

If this fails on Python 3.14, skip it. The API works with basic text files and template answers.

### 5. (Optional) Configure OpenAI

Copy .env.example to .env and set:

JWT_SECRET=change-me
OPENAI_API_KEY=sk-your-key-here

If you skip this, the /chat endpoint returns a template answer from context.

### 6. Start the API

uvicorn api.main:app --reload

API runs at http://localhost:8000

### 7. Open Swagger UI

http://localhost:8000/docs

## API Usage

### POST /register

Request:
{
  "username": "sumit",
  "password": "securepass123"
}

Response:
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}

### POST /login

Same payload as register. Returns a fresh JWT.

### POST /upload

Multipart form upload. Requires Authorization header.

curl -X POST "http://localhost:8000/upload" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@document.pdf"

Response:
{
  "id": 1,
  "filename": "document.pdf",
  "chars": 4523
}

### POST /chat

Request:
{
  "question": "What is the main topic?",
  "top_k": 3
}

Response:
{
  "question": "What is the main topic?",
  "answer": "Based on your documents, here is what I found: ...",
  "sources": [
    {
      "filename": "document.pdf",
      "relevance": 0.84,
      "preview": "This document discusses..."
    }
  ]
}

### GET /documents

List all uploaded documents for the current user.

### GET /history

Returns the last 50 chat Q&A pairs for the current user.

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | API info |
| /health | GET | Health check |
| /docs | GET | Swagger UI |
| /redoc | GET | Alternative docs |

### Full Usage Flow

1. POST /register (or /login) -> get token
2. In Swagger, click the lock icon and paste the token
3. POST /upload with a PDF or text file
4. POST /chat with a question
5. GET /history to see the conversation

## How the RAG Works

1. **Chunking** — text is split into ~400-character chunks on sentence boundaries with 50-char overlap
2. **Vectorization** — TfidfVectorizer converts each chunk to a sparse vector
3. **Retrieval** — the question is vectorized and compared to all chunks via cosine similarity
4. **Top-K** — the K most similar chunks are returned with relevance scores
5. **Answer** — the LLM (or template) generates an answer using only those chunks as context

Why TF-IDF fallback:
- No GPU, no embeddings download, no external API
- Fast (milliseconds even for large documents)
- Works fully offline — great for CI and demos

Upgrade path: swap TF-IDF for Sentence Transformers + ChromaDB by installing optional deps.

## Authentication

- Passwords hashed with bcrypt (via passlib)
- JWT tokens signed with HS256
- Tokens expire after 60 minutes (configurable)
- Every protected endpoint requires a Bearer token
- Documents and chat history are scoped per user_id

## Testing

pytest tests/ -v

Tests cover:
- Register and login flow
- Upload a text document
- Chat returns answer + sources
- Chat without token returns 401

## Docker

Build:
docker build -t ai-document-chat .

Run:
docker run -p 8000:8000 ai-document-chat

## CI/CD Pipeline

Every push to main triggers GitHub Actions:

1. Install Python 3.11 + dependencies
2. Run pytest test suite (uses in-memory SQLite, no API key needed)

See .github/workflows/ci.yml.

## Key Learnings

- JWT auth + per-user storage = clean multi-tenant architecture
- Chunking strategy matters more than embedding model for small docs
- TF-IDF is a surprisingly strong baseline for document Q&A
- Cosine similarity over TF-IDF vectors scales to thousands of chunks on CPU
- Grounding answers in retrieved context (with source previews) reduces hallucination
- FastAPI dependency injection makes auth clean and testable
- SQLite is production-ready for small-scale multi-user apps

## Future Improvements

- Swap TF-IDF for Sentence Transformers + ChromaDB (already supported via optional deps)
- Add PDF preview thumbnail in responses
- Streaming answers via Server-Sent Events
- Document folders and tags
- Team workspaces (shared documents)
- Rate limiting per user
- Vector index persistence (save embeddings between requests)
- Deploy to GCP Cloud Run with Cloud SQL
- Full React frontend with chat UI

## License

MIT License - see LICENSE file.

## Author

Sumit Raj
- M.Tech Applied AI & ML @ VNIT Nagpur
- Ex-Software Engineer Intern @ Salesforce
- GitHub: https://github.com/sumit966
- LinkedIn: https://www.linkedin.com/in/er-sumit-raj-/
- Portfolio: https://sumit966-github-io.vercel.app
- Email: info.sr0909@gmail.com

If you found this project useful, please consider giving it a star!
