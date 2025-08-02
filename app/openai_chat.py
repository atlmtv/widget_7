import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

SYSTEM_PROMPT = "Ты — ИИ-ассистент для веб-портала закупок Самрук Казына. Твоя основная задача — помогать пользователям разбираться в процессе, предоставлять пошаговые инструкции и актуальную информацию. Разговаривай с пользователем на том языке, на котором он к тебе обратился. Не используй интернет для ответов, пользуйся только базой знаний. Твоя задача — быть максимально полезным на любом языке! Тебе запрещено пользоваться интеренетом для ответов на вопросы."
DEFAULT_RESPONSE = "Извините, я не смог найти ответ на ваш вопрос. Если ваш вопрос касается работы веб-портала zakup.sk.kz, пожалуйста, обратитесь по телефону +7 (7172) 55-22-66"

def ask_openai(user_question, knowledge_snippets):
    context = "\n".join(f"Факт {i+1}: {snippet}" for i, snippet in enumerate(knowledge_snippets))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nВопрос: {user_question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message['content']
