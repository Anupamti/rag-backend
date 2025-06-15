from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
from rag_utils import process_uploaded_file, get_rag_response

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    contents = await file.read()
    
    with open(f"docs/{filename}", "wb") as f:
        f.write(contents)
    
    process_uploaded_file(f"docs/{filename}")
    return {"status": "success", "filename": filename}

class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: QuestionInput):
    response = get_rag_response(payload.question)
    return {"answer": response}
