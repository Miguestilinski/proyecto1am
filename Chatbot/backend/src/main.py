from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_setup import answer_question, get_context_for_query

app = FastAPI(title="Hospital El Pino - ChatBot de Gestión de Camas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    answer: str
    context_used: str
    sources: List[dict]

@app.get("/")
def root():
    return {"message": "API del ChatBot Hospital El Pino - Fase 2"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
        result = answer_question(request.message, history=history_dicts)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/{query}")
async def get_context(query: str):
    try:
        # Ahora desempaquetamos los 3 valores
        context, stats_context, sources = get_context_for_query(query, k=10)
        return {
            "stats": stats_context, 
            "context": context, 
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)