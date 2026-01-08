import logging
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InputFile,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.request import HTTPXRequest
from telegram.error import TimedOut
from config import TELEGRAM_BOT_TOKEN, DOMAINS
from generator_client import ExamGenerator

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

generator = ExamGenerator()
executor = ThreadPoolExecutor(max_workers=1)

# Предзагрузка модели
logger.info("Предзагрузка модели (первый адаптер)...")
try:
    generator._load_adapter("crypto")
    logger.info("✓ Модель загружена и готова к работе")
except Exception as e:
    logger.error(f"❌ Ошибка при предзагрузке модели: {e}")


async def delete_file_after_delay(file_path: Path, delay_minutes: int = 30):
    """Удаляет файл через заданное количество минут."""
    await asyncio.sleep(delay_minutes * 60)
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Удалён файл: {file_path.name}")
    except Exception as e:
        logger.error(f"Ошибка при удалении {file_path.name}: {e}")


async def cleanup_old_files(interval_minutes: int = 30, max_age_minutes: int = 60):
    """Удаляет файлы старше max_age_minutes каждые interval_minutes."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            now = time.time()
            output_dir = Path("outputs")
            if not output_dir.exists():
                continue
                
            for file_path in output_dir.glob("tickets_*"):
                if file_path.is_file():
                    file_age_minutes = (now - file_path.stat().st_mtime) / 60
                    if file_age_minutes > max_age_minutes:
                        file_path.unlink()
                        logger.info(f"Удалён старый файл: {file_path.name} (возраст: {file_age_minutes:.1f} мин)")
        except Exception as e:
            logger.error(f"Ошибка при очистке файлов: {e}")


async def send_file_with_retry(message, path: Path, caption: str, max_retries: int = 3):
    """Отправляет файл с повторными попытками при таймауте."""
    for attempt in range(max_retries):
        try:
            with open(path, "rb") as f:
                await message.reply_document(
                    document=InputFile(f, filename=path.name),
                    caption=caption,
                )
            logger.info(f"Файл {path.name} успешно отправлен")
            return
        except TimedOut:
            if attempt < max_retries - 1:
                logger.warning(f"Таймаут при отправке {path.name}, попытка {attempt + 2}/{max_retries}")
                await asyncio.sleep(2)
            else:
                logger.error(f"Не удалось отправить {path.name} после {max_retries} попыток")
                await message.reply_text("❌ Не удалось отправить файл. Попробуйте позже или выберите другой формат.")
                raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Привет! Я бот для генерации экзаменационных билетов.\n\n"
        "Как пользоваться:\n"
        "1️⃣ Выберите предметную область.\n"
        "2️⃣ Укажите количество вопросов в билете и количество билетов.\n"
        "3️⃣ Выберите формат выдачи: сообщение, TXT или PDF.\n\n"
        "Начнем: выберите предметную область ⬇️"
    )
    keyboard = [
        [InlineKeyboardButton("🔐 Криптография", callback_data="domain:crypto")],
        [InlineKeyboardButton("📊 Алгоритмы и структуры данных", callback_data="domain:algorithms")],
        [InlineKeyboardButton("🌐 Компьютерные сети", callback_data="domain:networks")],
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


async def domain_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, domain_key = query.data.split(":")
    context.user_data["domain_key"] = domain_key

    keyboard = [
        [
            InlineKeyboardButton("2 вопроса", callback_data="questions:2"),
            InlineKeyboardButton("3 вопроса", callback_data="questions:3"),
        ],
        [
            InlineKeyboardButton("4 вопроса", callback_data="questions:4"),
            InlineKeyboardButton("5 вопросов", callback_data="questions:5"),
        ],
    ]
    await query.edit_message_text(
        text=f"Вы выбрали: {DOMAINS[domain_key]}\nТеперь выберите количество вопросов в билете:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def questions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, n = query.data.split(":")
    num_questions = int(n)
    context.user_data["num_questions"] = num_questions

    keyboard = [
        [
            InlineKeyboardButton("1 билет", callback_data="tickets:1"),
            InlineKeyboardButton("3 билета", callback_data="tickets:3"),
        ],
        [
            InlineKeyboardButton("5 билетов", callback_data="tickets:5"),
            InlineKeyboardButton("10 билетов", callback_data="tickets:10"),
        ],
    ]
    await query.edit_message_text(
        text=f"Вопросов в билете: {num_questions}\nТеперь выберите количество билетов:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def tickets_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, n = query.data.split(":")
    num_tickets = int(n)
    context.user_data["num_tickets"] = num_tickets

    keyboard = [
        [InlineKeyboardButton("📩 В сообщении", callback_data="format:text")],
        [InlineKeyboardButton("📄 TXT файл", callback_data="format:txt")],
        [InlineKeyboardButton("📕 PDF файл", callback_data="format:pdf")],
    ]
    await query.edit_message_text(
        text=f"Билетов: {num_tickets}\nВыберите формат:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def format_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, fmt = query.data.split(":")

    domain_key = context.user_data.get("domain_key")
    num_questions = context.user_data.get("num_questions")
    num_tickets = context.user_data.get("num_tickets")

    if not (domain_key and num_questions and num_tickets):
        await query.edit_message_text("Начните заново с /start.")
        return

    await query.edit_message_text("⏳ Генерирую билеты...")
    
    status_msg = await query.message.reply_text("🔄 Инициализация... (0 сек)")

    async def update_status():
        elapsed = 0
        while True:
            await asyncio.sleep(20)
            elapsed += 20
            try:
                await status_msg.edit_text(f"🔄 Генерация... ({elapsed} сек)")
            except:
                pass

    status_task = asyncio.create_task(update_status())

    try:
        loop = asyncio.get_event_loop()
        logger.info(f"START: {domain_key}, {num_questions}q, {num_tickets}t")
        
        tickets = await loop.run_in_executor(
            executor,
            generator.generate_tickets,
            domain_key,
            num_questions,
            num_tickets
        )
        
        logger.info(f"DONE: {len(tickets)} билетов")
        
    except Exception as e:
        logger.exception("ERROR")
        await status_msg.edit_text(f"❌ Ошибка: {e}")
        return
    finally:
        status_task.cancel()
        try:
            await status_msg.delete()
        except:
            pass

    if fmt == "text":
        header = f"📄 {DOMAINS[domain_key]}\nВопросов: {num_questions}\nБилетов: {num_tickets}\n\n"
        await query.message.reply_text(header)
        for idx, t in enumerate(tickets, start=1):
            await query.message.reply_markdown(f"*Билет №{idx}*\n{t}")

    elif fmt == "txt":
        path = generator.save_as_txt(tickets, domain_key, num_questions)
        await send_file_with_retry(query.message, path, "Билеты в TXT")
        asyncio.create_task(delete_file_after_delay(path, delay_minutes=30))

    elif fmt == "pdf":
        path = generator.save_as_pdf(tickets, domain_key, num_questions)
        await send_file_with_retry(query.message, path, "Билеты в PDF")
        asyncio.create_task(delete_file_after_delay(path, delay_minutes=30))

    keyboard = [[InlineKeyboardButton("🔁 Ещё", callback_data="restart")]]
    await query.message.reply_text("Готово!", reply_markup=InlineKeyboardMarkup(keyboard))


async def restart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    
    text = (
        "👋 Начнём сначала!\n\n"
        "Выберите предметную область ⬇️"
    )
    keyboard = [
        [InlineKeyboardButton("🔐 Криптография", callback_data="domain:crypto")],
        [InlineKeyboardButton("📊 Алгоритмы и структуры данных", callback_data="domain:algorithms")],
        [InlineKeyboardButton("🌐 Компьютерные сети", callback_data="domain:networks")],
    ]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))


def main():
    # Создаём request с увеличенными таймаутами
    request = HTTPXRequest(
        connection_pool_size=8,
        read_timeout=60.0,
        write_timeout=60.0,
        connect_timeout=30.0,
        pool_timeout=30.0
    )
    
    app = Application.builder()\
        .token(TELEGRAM_BOT_TOKEN)\
        .request(request)\
        .build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(domain_handler, pattern=r"^domain:"))
    app.add_handler(CallbackQueryHandler(questions_handler, pattern=r"^questions:"))
    app.add_handler(CallbackQueryHandler(tickets_handler, pattern=r"^tickets:"))
    app.add_handler(CallbackQueryHandler(format_handler, pattern=r"^format:"))
    app.add_handler(CallbackQueryHandler(restart_handler, pattern=r"^restart$"))

    logger.info("Bot ready")
    app.run_polling()
    
    # Запускаем фоновую очистку ПОСЛЕ создания event loop
    asyncio.create_task(cleanup_old_files(interval_minutes=30, max_age_minutes=60))


if __name__ == "__main__":
    main()
