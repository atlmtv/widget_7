# Структура проекта:
# /app
#   |- main.py          # сервер FastAPI
#   |- retriever.py     # поиск по базе знаний
#   |- openai_chat.py   # отправка запросов в OpenAI
#   |- embedder.py      # нарезка базы знаний
#   |- knowledge.txt    # файл базы знаний
# /frontend
#   |- widget.html      # скрипт для сайта
# Dockerfile            # docker-сборка

# --------------------
# app/embedder.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []

    def split_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_index(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.text_chunks = self.split_text(text)
        embeddings = self.model.encode(self.text_chunks)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.text_chunks[i] for i in I[0]]


# --------------------
# app/retriever.py
from app.embedder import Embedder

embedder = Embedder()
embedder.build_index('app/knowledge.txt')

def retrieve_relevant_chunks(query):
    return embedder.search(query)


# --------------------
# app/openai_chat.py
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
Ты — ИИ-ассистент для сайта портала закупок СКК. Твоя основная задача — помогать пользователям разбираться в процессе, предоставлять пошаговые инструкции и актуальную информацию.

Тебя зовут Айсулу.
Если нет информации для ответа, скажи: \"Извините, я не нашел ответа на ваш вопрос.\"
"""

def ask_openai(user_question, knowledge_snippets):
    context = "\n".join(f"Факт {i+1}: {snippet}" for i, snippet in enumerate(knowledge_snippets))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nВопрос: {user_question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message['content']


# --------------------
# app/main.py
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


# --------------------
# frontend/widget.html
<div id="chatbot-widget"></div>

<script>
document.addEventListener("DOMContentLoaded", function() {
  const widget = document.getElementById("chatbot-widget");
  widget.innerHTML = '<div id="chatbox" style="width: 300px; height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px;"></div>' +
                     '<input id="user-input" style="width: 80%;" placeholder="Ваш вопрос..."/>' +
                     '<button id="send-btn">Отправить</button>';

  const chatbox = document.getElementById("chatbox");
  const history = [];

  function renderHistory() {
    chatbox.innerHTML = '';
    history.forEach(entry => {
      chatbox.innerHTML += `<div><b>${entry.sender}:</b> ${entry.message}</div>`;
    });
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  document.getElementById("send-btn").onclick = async () => {
    const input = document.getElementById("user-input").value;
    if (!input.trim()) return;

    history.push({ sender: "Вы", message: input });
    renderHistory();

    const loadingMessage = { sender: "Бот", message: "Печатает..." };
    history.push(loadingMessage);
    renderHistory();

    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: input })
    });

    const data = await res.json();
    history.pop(); // Убираем "Печатает..."
    history.push({ sender: "Бот", message: data.answer });
    renderHistory();
  };
});
</script>


# --------------------
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY app app/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# --------------------
# requirements.txt
fastapi
uvicorn
openai
faiss-cpu
sentence-transformers
python-dotenv


# --------------------
# Пример содержимого knowledge.txt
Как пройти регистрацию? 
"Для работы на веб-портале закупок необходимо установить приложение NCALayer. Чтобы зарегистрировать юридическое лицо, выполните следующие шаги:
В правом верхнем углу нажмите ""Зарегистрироваться"" — получите данные ключа.
Заполните заявку на регистрацию.
Подпишите заявку электронно-цифровой подписью первого руководителя компании.
Чтобы зарегистрироваться в качестве индивидуального предпринимателя, необходимо загрузить талон о приеме уведомления о начале деятельности с портала электронного лицензирования www.elicense.kz. Регистрацию может осуществлять только сам индивидуальный предприниматель. Если у вас нет уведомления о начале деятельности, вы можете зарегистрироваться с помощью электронной цифровой подписи. Рассмотрение заявления на регистрацию происходит в течение 5 рабочих дней.