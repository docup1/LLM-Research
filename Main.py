import os
import shutil
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

# Настройка стиля графиков
sns.set_theme(style="whitegrid")

# Включаем логирование HuggingFace, чтобы видеть прогресс загрузки
hf_logging.set_verbosity_info()

# --- КОНФИГУРАЦИЯ ---
CACHE_DIR = "./llm_cache"
AVAILABLE_MODELS = {
    "1": {"name": "distilgpt2", "id": "distilgpt2", "desc": "Очень легкая (82M)"},
    "2": {"name": "gpt2", "id": "gpt2", "desc": "Стандартная (124M)"},
    "3": {"name": "DialoGPT-small", "id": "microsoft/DialoGPT-small", "desc": "Диалоговая (117M)"},
    "4": {"name": "TinyLlama-1.1B", "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "desc": "Мощная, но тяжелая (1.1B)"},
}

class ModelManager:
    """Класс для управления загрузкой и удалением моделей."""
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.model_name = None

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def is_downloaded(self, model_id):
        """Проверяет, существует ли папка модели в кэше."""
        model_path = os.path.join(CACHE_DIR, f"models--{model_id.replace('/', '--')}")
        return os.path.exists(model_path)

    def load_model(self, selection_key):
        """Загружает выбранную модель в память с логами."""
        if selection_key not in AVAILABLE_MODELS:
            print("❌ Ошибка: Неверный выбор модели.")
            return False

        model_info = AVAILABLE_MODELS[selection_key]
        print(f"\n" + "="*50)
        print(f"🚀 ИНИЦИАЛИЗАЦИЯ ЗАГРУЗКИ: {model_info['name']}")
        print(f"🆔 ID модели: {model_info['id']}")
        print("="*50)

        try:
            # 1. Загрузка токенизатора
            print(f"\n[1/3] 📖 Загрузка токенизатора...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_info['id'],
                cache_dir=CACHE_DIR
            )
            print("✅ Токенизатор успешно загружен.")

            # Фикс для моделей без pad_token
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
                print("ℹ️ Pad token отсутствовал, установлен EOS token.")

            # 2. Загрузка модели
            print(f"\n[2/3] 🧠 Скачивание и загрузка весов нейросети...")
            print("      (Если модель скачивается впервые, вы увидите логи загрузки файлов ниже)\n")

            # Определяем устройство
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"💻 Используемое устройство: {device.upper()}")

            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_info['id'],
                cache_dir=CACHE_DIR,
                device_map="auto" if torch.cuda.is_available() else None, # auto лучше работает с GPU
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Исправлено torch_dtype -> dtype
            )

            # Если device_map="auto" не сработал (на CPU иногда глючит), переносим вручную
            if device == "cpu":
                self.current_model.to("cpu")

            self.model_name = model_info['name']

            print(f"\n[3/3] ✨ Финализация...")
            print(f"✅ Модель {self.model_name} готова к работе!")
            print("="*50 + "\n")
            return True

        except Exception as e:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАГРУЗКЕ: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_model(self, selection_key):
        """Удаляет файлы модели с диска."""
        if selection_key not in AVAILABLE_MODELS:
            print("❌ Ошибка: Неверный выбор.")
            return

        model_id = AVAILABLE_MODELS[selection_key]['id']
        folder_name = f"models--{model_id.replace('/', '--')}"
        path = os.path.join(CACHE_DIR, folder_name)

        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"🗑️ Файлы модели {AVAILABLE_MODELS[selection_key]['name']} удалены.")
            except Exception as e:
                print(f"❌ Ошибка удаления: {e}")
        else:
            print(f"⚠️ Файлы не найдены.")

    def unload_model(self):
        """Выгружает модель из оперативной памяти."""
        if self.current_model:
            print("🧹 Очистка памяти...")
            del self.current_model
            del self.current_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.current_model = None
            self.current_tokenizer = None
            self.model_name = None
            print("✅ Память очищена.")
        else:
            print("⚠️ Нет загруженной модели.")

class ExperimentRunner:
    """Класс для проведения экспериментов с генерацией."""
    def __init__(self, manager):
        self.manager = manager

    def generate_text(self, prompt, **kwargs):
        if not self.manager.current_model:
            print("❌ Сначала загрузите модель!")
            return None

        # Подготовка входных данных
        inputs = self.manager.current_tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.manager.current_model.device)

        gen_kwargs = {
            "max_new_tokens": 50,
            "do_sample": True,
            "pad_token_id": self.manager.current_tokenizer.pad_token_id
        }
        gen_kwargs.update(kwargs)

        try:
            # Отключаем логирование на момент самой генерации, чтобы не мусорить
            hf_logging.set_verbosity_error()

            with torch.no_grad():
                outputs = self.manager.current_model.generate(**inputs, **gen_kwargs)

            # Возвращаем логирование обратно
            hf_logging.set_verbosity_info()

            text = self.manager.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return None

    def run_temperature_experiment(self):
        print("\n--- 🌡️ ЭКСПЕРИМЕНТ: TEMPERATURE ---")
        prompt = "The future of artificial intelligence is"
        temps = [0.1, 0.4, 0.7, 1.0]
        results = []

        print(f"Промпт: '{prompt}'\n")

        for t in temps:
            print(f"⚙️ Temperature = {t}...")
            text = self.generate_text(prompt, temperature=t, top_k=50)
            print(f"➤ Результат: {text[len(prompt):].strip()}...\n")
            results.append(len(text))

        plt.figure(figsize=(8, 4))
        plt.plot(temps, results, marker='o')
        plt.title(f"Длина генерации vs Temperature ({self.manager.model_name})")
        plt.xlabel("Temperature")
        plt.ylabel("Длина текста (символы)")
        plt.savefig("temp_experiment.png")
        print("📊 График сохранен как 'temp_experiment.png'")

    def run_sampling_experiment(self):
        print("\n--- 🎲 ЭКСПЕРИМЕНТ: TOP-K и TOP-P ---")
        prompt = "Once upon a time in a galaxy far away"

        configs = [
            {"top_k": 5, "top_p": None, "name": "Top-k=5 (Strict)"},
            {"top_k": 50, "top_p": None, "name": "Top-k=50 (Diverse)"},
            {"top_k": None, "top_p": 0.3, "name": "Top-p=0.3 (Nucleus Strict)"},
            {"top_k": None, "top_p": 0.9, "name": "Top-p=0.9 (Nucleus Creative)"},
        ]

        combinations = [
            {"top_k": 50, "top_p": 0.9},
            {"top_k": 10, "top_p": 0.5},
            {"top_k": 100, "top_p": 0.95}
        ]

        print(f"Промпт: '{prompt}'\n")

        for conf in configs:
            kwargs = {k: v for k, v in conf.items() if k != "name" and v is not None}
            print(f"⚙️ {conf['name']}...")
            text = self.generate_text(prompt, **kwargs)
            if text:
                print(f"➤ {text[len(prompt):].strip()}...\n")

        print("--- Комбинации ---")
        for comb in combinations:
            print(f"⚙️ k={comb['top_k']}, p={comb['top_p']}...")
            text = self.generate_text(prompt, top_k=comb['top_k'], top_p=comb['top_p'])
            if text:
                print(f"➤ {text[len(prompt):].strip()}...\n")

    def run_prompt_types(self):
        print("\n--- 📝 ЭКСПЕРИМЕНТ: ТИПЫ ПРОМПТОВ ---")
        prompts = {
            "Утверждение": "Python is the best language because",
            "Вопрос": "What is the capital of France?",
            "Творчество": "Write a short poem about a robot:",
            "Список": "List 5 fruits:\n1.",
        }

        for p_type, prompt in prompts.items():
            print(f"\n🔹 Тип: {p_type} | Промпт: {prompt}")
            text = self.generate_text(prompt, temperature=0.7, top_k=40)
            if text:
                print(f"➤ Ответ: {text}")

def main_menu():
    manager = ModelManager()
    runner = ExperimentRunner(manager)

    while True:
        print("\n" + "="*40)
        print(f"🤖 LLM LAB CONTROL PANEL | Модель: {manager.model_name if manager.model_name else 'Не выбрана'}")
        print("="*40)
        print("1. 📂 Управление моделями")
        print("2. 🌡️ Исследование Temperature")
        print("3. 🎲 Исследование Top-k / Top-p")
        print("4. 📝 Тест разных типов промптов")
        print("5. ✍️ Ручной ввод")
        print("0. 🚪 Выход")

        choice = input("\nВаш выбор: ")

        if choice == "1":
            while True:
                print("\n--- МЕНЮ МОДЕЛЕЙ ---")
                for k, v in AVAILABLE_MODELS.items():
                    status = "💾 (Скачана)" if manager.is_downloaded(v['id']) else "☁️ (Нужно качать)"
                    current = "⭐ [АКТИВНА]" if manager.model_name == v['name'] else ""
                    print(f"{k}. {v['name']} | {status} {current}")

                print("U. Unload (Очистить память)")
                print("D. Delete (Удалить файлы)")
                print("B. Back (Назад)")

                sub = input("Выбор: ").lower()
                if sub == 'b': break
                elif sub == 'u': manager.unload_model()
                elif sub == 'd':
                    key = input("Номер модели: ")
                    manager.delete_model(key)
                elif sub in AVAILABLE_MODELS:
                    manager.load_model(sub)
                else: print("Неверный ввод.")

        elif choice == "2":
            if manager.current_model: runner.run_temperature_experiment()
            else: print("⚠️ Сначала выберите модель (Пункт 1)")
        elif choice == "3":
            if manager.current_model: runner.run_sampling_experiment()
            else: print("⚠️ Сначала выберите модель (Пункт 1)")
        elif choice == "4":
            if manager.current_model: runner.run_prompt_types()
            else: print("⚠️ Сначала выберите модель (Пункт 1)")
        elif choice == "5":
            if manager.current_model:
                p = input("Промпт: ")
                res = runner.generate_text(p, temperature=0.7)
                if res: print(f"\n➤ {res}")
            else: print("⚠️ Сначала выберите модель (Пункт 1)")
        elif choice == "0": break
        else: print("Ошибка ввода")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nВыход.")