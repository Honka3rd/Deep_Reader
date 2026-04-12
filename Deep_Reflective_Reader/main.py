from fastapi import FastAPI, HTTPException

from coordinator import Coordinator
from api_schemas import (
    PrepareDocumentRequest,
    AskDocumentRequest,
    AskDocumentResponse,
    StatusResponse,
)

# 建立 FastAPI app
app = FastAPI(
    title="Deep Reader API",
    version="0.1.0",
)

# ⭐ 這裡很關鍵：Coordinator 是全域 singleton
coordinator = Coordinator(
    chunk_size=300,
    chunk_overlap=50,
    embedding_model="text-embedding-3-small",
)

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health", response_model=StatusResponse)
def health():
    return StatusResponse(
        status="ok",
        message="Deep Reader API is running",
    )

# ---------------------------
# Ask Question
# ---------------------------
@app.post("/documents/ask", response_model=AskDocumentResponse)
def ask_document(request: AskDocumentRequest):
    try:
        answer = coordinator.ask(
            doc_name=request.doc_name,
            question=request.query,
            top_k=request.top_k,
        )

        return AskDocumentResponse(
            doc_name=request.doc_name,
            query=request.query,
            answer=answer,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))