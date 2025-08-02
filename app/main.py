from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retriever import retrieve_relevant_chunks
from app.openai_chat import ask_openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_bot(q: Question):
    relevant_chunks = retrieve_relevant_chunks(q.question)
    answer = ask_openai(q.question, relevant_chunks)
    return {"answer": answer}