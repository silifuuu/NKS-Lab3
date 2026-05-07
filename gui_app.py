import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. Настройка путей
model_path = "./my_finetuned_model_lab3"

print("⏳ Загрузка модели и токенизатора (это может занять время)...")

try:
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Загружаем базовую модель на CPU (так как мы на i5 без мощной GPU)
    # torch_dtype=torch.float32 важен для стабильности на процессоре
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print("✅ Модель успешно загружена!")

except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    print("Убедитесь, что модель скачана в папку my_finetuned_model_lab3")

def predict(message, history):
    # Формируем промпт (Gemma-2 использует специфический формат)
    # Мы добавляем контекст, чтобы она отвечала как эксперт по GloVe
    prompt = f"Ответь на вопрос по теме NLP и GloVe как эксперт.\nВопрос: {message}\nОтвет:"
    
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем ответ, убирая промпт из начала
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.replace(prompt, "").strip()
    
    return response

# Создание интерфейса Gradio
demo = gr.ChatInterface(
    fn=predict,
    title="Gemma-2 2B (Real Inference Mode)",
    description="Интерфейс реальной модели Gemma-2 с LoRA адаптером. Работает на CPU.",
    examples=["Что такое векторы GloVe?", "В чем суть функции потерь в GloVe?"],
)

if __name__ == "__main__":
    # share=False, так как модель тяжелая и интернет-канал может не потянуть внешний доступ
    demo.launch(share=False)