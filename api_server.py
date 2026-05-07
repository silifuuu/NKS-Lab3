# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------ 1. Инициализация FastAPI ------------------
app = FastAPI(title="ruDialoGPT Chat API", version="1.0")

# Монтируем папку static для раздачи HTML, CSS, JS файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ 2. Глобальная загрузка модели ------------------
model_path = "./my_finetuned_model_lab3"  # Указан правильный путь к папке с моделью
print("[INFO] Загрузка токенизатора и модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Переводим в режим инференса
print("[INFO] Модель успешно загружена и готова к запросам.")

# ------------------ 3. Pydantic модели (валидация JSON) ------------------
class ChatMessage(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    message: str                     
    history: list[ChatMessage] = []  

class ChatResponse(BaseModel):
    response: str

# ------------------ 4. Обработчик главной страницы ------------------
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# ------------------ 5. Основной API-эндпоинт ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # --- Шаг 1: Формирование промпта с историей ---
        prompt = ""
        for msg in request.history:
            prompt += f"@@ПЕРВЫЙ@@{msg.user}@@ВТОРОЙ@@{msg.bot}"
        prompt += f"@@ПЕРВЫЙ@@{request.message}@@ВТОРОЙ@@"

        # --- Шаг 2: Токенизация ---
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=900   
        )

        # --- Шаг 3: Генерация (инференс) ---
        with torch.no_grad():  
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,          
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        # --- Шаг 4: Декодирование и очистка ответа ---
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        input_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))
        new_part = full_response[input_length:]

        response = new_part.split("@@ПЕРВЫЙ@@")[0] if "@@ПЕРВЫЙ@@" in new_part else new_part
        response = response.replace("@@ВТОРОЙ@@", "").strip()

        if not response or response in ["@@", "@"]:
            response = "[Модель не смогла сформулировать ответ]"

        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")

# ------------------ 6. Точка входа ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)