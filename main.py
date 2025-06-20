from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from rag_utils import process_uploaded_document, process_uploaded_transcript, get_rag_response, reset_context
import uuid
from fastapi import UploadFile
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload policy documents, service descriptions, evidence-based practices, state standards"""
    filename = f"{uuid.uuid4()}_{file.filename}"
    contents = await file.read()
    
    # Create documents directory if it doesn't exist
    os.makedirs("docs/documents", exist_ok=True)
    
    with open(f"docs/documents/{filename}", "wb") as f:
        f.write(contents)
    
    process_uploaded_document(f"docs/documents/{filename}")
    return {"status": "success", "filename": filename, "type": "document"}

@app.post("/upload/transcript")
async def upload_transcript(
    file: UploadFile = File(...), 
    client_id: str = Form(None),
    session_date: str = Form(None),
    case_manager: str = Form(None)
):
    """Upload client transcripts with optional metadata"""
    filename = f"{uuid.uuid4()}_{file.filename}"
    contents = await file.read()
    
    # Create transcripts directory if it doesn't exist
    os.makedirs("docs/transcripts", exist_ok=True)
    
    with open(f"docs/transcripts/{filename}", "wb") as f:
        f.write(contents)
    
    # Pass metadata to processing function
    metadata = {
        "client_id": client_id,
        "session_date": session_date,
        "case_manager": case_manager,
        "filename": filename
    }
    
    process_uploaded_transcript(f"docs/transcripts/{filename}", metadata)
    return {"status": "success", "filename": filename, "type": "transcript", "metadata": metadata}

class QuestionInput(BaseModel):
    question: str
    query_type: str = "general"  # general, policy_check, recommendation, cross_reference

@app.post("/ask")
async def ask_question(payload: QuestionInput):
    response = get_rag_response(payload.question, payload.query_type)
    return {"answer": response, "query_type": payload.query_type}

@app.delete("/reset")
async def reset_vectorstore():
    reset_context()
    return {"status": "success", "message": "All context cleared."}

@app.get("/stats")
async def get_stats():
    """Get statistics about uploaded documents and transcripts"""
    from rag_utils import get_collection_stats
    stats = get_collection_stats()
    return stats