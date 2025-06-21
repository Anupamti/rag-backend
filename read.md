🧠 RAG-based Insight Engine
This is a Retrieval-Augmented Generation (RAG) system built using FastAPI.
---
🛠️ Getting Started
1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-insight-engine.git
cd rag-insight-engine
```
2. Change the Branch
```bash
git checkout casebasev2
```

2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
# On Linux/macOS
source venv/bin/activate
```
⚠️ Make sure you have Python 3.8+ installed.
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Create a .env File
Create a file named `.env` in the root directory and add the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4

# CORS Configuration (comma-separated list of allowed origins)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,https://yourdomain.com

LOG_LEVEL=INFO
```
5. Run the Application
Start the FastAPI app using uvicorn:
```bash
python -m uvicorn main:app --reload
```
This command works on both macOS and Linux.
