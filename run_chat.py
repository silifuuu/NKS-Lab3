from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./my_finetuned_model_lab3"  # Текущая директория с файлами модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Настройка технического токена
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Перевод модели в режим инференса (отключает Dropout)

chat_history = ""
while True:
    user_input = input("Вы: ")
    if user_input.lower() == 'выход':
        break

    # 2. Формат промпта для диалоговой модели (используем управляющие токены)
    if not chat_history:
        prompt = f"@@ПЕРВЫЙ@@{user_input}@@ВТОРОЙ@@"
    else:
        prompt = f"{chat_history}@@ПЕРВЫЙ@@{user_input}@@ВТОРОЙ@@"

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=900)

    # 3. Параметры генерации (отключаем построение графа вычислений)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,      # Максимальная длина генерируемого ответа в токенах
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,          # Включаем случайность (False — жадный поиск)
            top_k=50,                # Рассматриваем только 50 самых вероятных токенов
            top_p=0.9,               # Ядерная фильтрация (nucleus sampling)
            temperature=0.8,         # Степень "размытия" вероятностей
            repetition_penalty=1.2,  # Штраф за повтор слов
            no_repeat_ngram_size=3   # Запрещаем повторять n-граммы длиной 3
        )

    # Декодирование ответа
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    input_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))
    new_part = full_response[input_length:]
    
    # Очистка от технических токенов
    response = new_part.split("@@ПЕРВЫЙ@@")[0] if "@@ПЕРВЫЙ@@" in new_part else new_part
    response = response.replace("@@ВТОРОЙ@@", "").strip()

    print(f"Бот: {response}")
    
    # Сохраняем историю
    chat_history += f"@@ПЕРВЫЙ@@{user_input}@@ВТОРОЙ@@{response}"