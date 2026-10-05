"""FastAPI service for AI Document Chat."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from auth import hash_password, verify_password, create_access_token, decode_token
from storage import (
    init_db, create_user, get_user, save_document, list_documents,
    get_documents_content, save_chat, get_chat_history
)
from rag import SimpleRAG, simple_answer


app = FastAPI(
    title="AI Document Chat API",
    description="Upload PDFs and chat with your documents using RAG + JWT auth",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(3, ge=1, le=10)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing token")
    username = decode_token(creds.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.on_event("startup")
def startup():
    init_db()
    print("[OK] Database initialized")


@app.get("/")
def root():
    return {"message": "AI Document Chat API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if get_user(req.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    create_user(req.username, hash_password(req.password))
    token = create_access_token(req.username)
    return TokenResponse(access_token=token)


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(req.username)
    return TokenResponse(access_token=token)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    content = await file.read()
    text = ""

    # Try PDF extraction
    if file.filename.lower().endswith(".pdf"):
        try:
            import io
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    doc_id = save_document(user["id"], file.filename, text)
    return {"id": doc_id, "filename": file.filename, "chars": len(text)}


@app.get("/documents")
def list_docs(user: Dict[str, Any] = Depends(get_current_user)):
    return list_documents(user["id"])


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: Dict[str, Any] = Depends(get_current_user)):
    docs = get_documents_content(user["id"])
    if not docs:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    rag = SimpleRAG()
    rag.index(docs)
    results = rag.query(req.question, top_k=req.top_k)

    answer = simple_answer(req.question, results)
    save_chat(user["id"], req.question, answer)

    sources = [
        {"filename": r["filename"], "relevance": round(r["score"], 3), "preview": r["chunk"][:120]}
        for r in results
    ]

    return ChatResponse(question=req.question, answer=answer, sources=sources)


@app.get("/history")
def history(user: Dict[str, Any] = Depends(get_current_user)):
    return get_chat_history(user["id"])
