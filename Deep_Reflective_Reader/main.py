from uuid import uuid4

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
coordinator = Coordinator()

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health", response_model=StatusResponse)
def health():
    """Return API health status.

Returns:
    Service health payload used by readiness checks."""
    return StatusResponse(
        status="ok",
        message="Deep Reader API is running",
    )

# ---------------------------
# Ask Question
# ---------------------------
@app.post("/documents/ask", response_model=AskDocumentResponse)
def ask_document(request: AskDocumentRequest):
    """Execute `/documents/ask` request and return answer payload.

Args:
    request: API request payload model.

Returns:
    QA response payload including answer text and effective session id."""
    try:
        session_id = request.session_id.strip() if request.session_id else ""
        if not session_id:
            session_id = str(uuid4())

        answer = coordinator.ask(
            doc_name=request.doc_name,
            question=request.query,
            top_k=request.top_k,
            session_id=session_id,
        )

        return AskDocumentResponse(
            doc_name=request.doc_name,
            query=request.query,
            answer=answer,
            session_id=session_id,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
