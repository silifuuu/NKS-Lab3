import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. Пути (Проверь, чтобы папка с адаптером была рядом!)
base_model_name = "unsloth/gemma-2-2b-it"  # Как в примере Бобровских
adapter_path = "./lora_adapter_lab2"  # Папка, где лежат файлы твоего адаптера
save_path = "./my_finetuned_model_lab3"  # Куда сохраняем результат

print("📥 Загрузка базовой архитектуры...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=torch.float32, device_map="cpu"
)

print("🛠️ Накладываем LoRA-адаптер...")
# Загружаем модель вместе с твоим адаптером
model = PeftModel.from_pretrained(base_model, adapter_path)

print("🔗 Слияние весов (Merge and Unload)...")
# Это и есть то самое слияние, о котором просит методичка
merged_model = model.merge_and_unload()

print(f"💾 Сохранение в {save_path}...")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\n✅ Папка подготовлена! Теперь run_chat.py увидит модель.")
