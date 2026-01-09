from pathlib import Path

# Корень проекта (папка, где лежит config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Путь к адаптерам (локально в проекте бота)
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"

# Базовая модель (скачивается из HF Hub)
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Токен бота
TELEGRAM_BOT_TOKEN = "HERE_YOUR_TG_TOKEN"

# Домены
DOMAINS = {
    "cryptography": "Криптография",
    "algorithms": "Алгоритмы и структуры данных",
    "networks": "Компьютерные сети",
}

# Папка для файлов с билетами
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
