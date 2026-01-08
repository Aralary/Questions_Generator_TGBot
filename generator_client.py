from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import ADAPTERS_DIR, BASE_MODEL_NAME, DOMAINS, OUTPUT_DIR


class ExamGenerator:
    """Генерация экзаменационных билетов с помощью Mistral + LoRA-адаптеров."""

    def __init__(self):
        self.base_model_name = BASE_MODEL_NAME
        self.adapters_dir = Path(ADAPTERS_DIR)
        self.current_domain_key = None
        self.model = None
        self.tokenizer = None

        self.temperature = 0.7
        self.top_p = 0.9
        self.max_new_tokens = 512
        self.repetition_penalty = 1.15

    def _load_adapter(self, domain_key: str):
        """Загрузка базовой модели и нужного адаптера."""
        if self.current_domain_key == domain_key and self.model is not None:
            return

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        adapter_path = self.adapters_dir / domain_key
        if not adapter_path.exists():
            raise RuntimeError(
                f"Адаптер для домена '{domain_key}' не найден. "
                f"Ожидается директория: {adapter_path}"
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.current_domain_key = domain_key

    def _build_prompt(self, domain_key: str, num_questions: int, ticket_index: int) -> str:
        domain_name = DOMAINS[domain_key]
        instruction = (
            f"Сгенерируй экзаменационный билет №{ticket_index} "
            f"по предмету '{domain_name}' с {num_questions} вопросами открытого типа. "
            f"Нумеруй вопросы, не добавляй ответы."
        )
        return f"<s>[INST] {instruction} [/INST]"

    def generate_ticket(self, domain_key: str, num_questions: int, ticket_index: int) -> str:
        """Генерация одного билета."""
        self._load_adapter(domain_key)

        prompt = self._build_prompt(domain_key, num_questions, ticket_index)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[1].strip()
        return text.strip()

    def generate_tickets(self, domain_key: str, num_questions: int, num_tickets: int) -> List[str]:
        return [
            self.generate_ticket(domain_key, num_questions, i)
            for i in range(1, num_tickets + 1)
        ]

    def _format_tickets_block(self, tickets: List[str], domain_key: str, num_questions: int) -> str:
        header = (
            f"Экзаменационные билеты по дисциплине: {DOMAINS[domain_key]}\n"
            f"Количество вопросов в каждом билете: {num_questions}\n"
            + "=" * 80 + "\n\n"
        )
        parts = [header]
        for idx, t in enumerate(tickets, start=1):
            parts.append(f"Билет №{idx}\n")
            parts.append(t.strip() + "\n\n")
            parts.append("-" * 40 + "\n\n")
        return "".join(parts)

    def save_as_txt(self, tickets: List[str], domain_key: str, num_questions: int) -> Path:
        text_block = self._format_tickets_block(tickets, domain_key, num_questions)
        filename = OUTPUT_DIR / f"tickets_{domain_key}_{num_questions}q.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text_block)
        return filename

    def save_as_pdf(self, tickets: List[str], domain_key: str, num_questions: int) -> Path:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from datetime import datetime

        filename = OUTPUT_DIR / f"tickets_{domain_key}_{num_questions}q.pdf"

        # Регистрируем шрифты
        font_path = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"
        bold_font_path = Path(__file__).resolve().parent / "fonts" / "DejaVuSans-Bold.ttf"
        
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_path)))
        if bold_font_path.exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_font_path)))
        else:
            pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(font_path)))

        c = canvas.Canvas(str(filename), pagesize=A4)
        width, height = A4

        # Параметры страницы
        margin_top = 50
        margin_left = 40
        margin_right = 40
        margin_bottom = 40
        content_width = width - margin_left - margin_right

        # Цвета
        header_color = colors.HexColor("#1f4788")
        accent_color = colors.HexColor("#2196F3")
        text_color = colors.HexColor("#333333")

        y_position = height - margin_top

        # Заголовок документа
        c.setFont("DejaVuSans-Bold", 16)
        c.setFillColor(header_color)
        title = f"Экзаменационные билеты"
        c.drawString(margin_left, y_position, title)
        y_position -= 25

        # Информация о документе
        c.setFont("DejaVuSans", 10)
        c.setFillColor(text_color)
        domain_name = DOMAINS.get(domain_key, domain_key)
        info_text = f"Дисциплина: {domain_name}  |  Вопросов в билете: {num_questions}  |  Дата: {datetime.now().strftime('%d.%m.%Y')}"
        c.drawString(margin_left, y_position, info_text)
        y_position -= 20

        # Горизонтальная линия
        c.setStrokeColor(accent_color)
        c.setLineWidth(2)
        c.line(margin_left, y_position, width - margin_right, y_position)
        y_position -= 20

        # Флаг для отслеживания первой страницы
        is_first_page = True

        # Генерируем билеты
        for ticket_idx, ticket_text in enumerate(tickets, start=1):
            # Убираем "Экзаменационный билет №XXX по..." из текста
            clean_ticket_text = self._clean_ticket_text(ticket_text)
            
            # Проверяем, нужна ли новая страница
            text_height = self._calculate_text_height(
                clean_ticket_text, 
                content_width - 20, 
                "DejaVuSans", 
                10
            )
            
            if y_position - text_height - 30 < margin_bottom and not is_first_page:
                c.showPage()
                y_position = height - margin_top
                # НЕ добавляем больше "Продолжение" или другие надписи

            is_first_page = False

            # Отступ перед заголовком билета
            y_position -= 10

            # Заголовок билета
            c.setFont("DejaVuSans-Bold", 12)
            c.setFillColor(header_color)
            ticket_title = f"Билет № {ticket_idx}"
            c.drawString(margin_left, y_position, ticket_title)
            y_position -= 18

            # Содержимое билета
            c.setFont("DejaVuSans", 10)
            c.setFillColor(text_color)

            # Парсим вопросы
            questions = clean_ticket_text.split("\n\n")
            for question in questions:
                if not question.strip():
                    continue

                # Переносим длинный текст
                wrapped_text = self._wrap_text(question.strip(), content_width - 20, "DejaVuSans", 10)
                for line in wrapped_text:
                    c.drawString(margin_left + 10, y_position, line)
                    y_position -= 14

                y_position -= 5  # Зазор между вопросами

            y_position -= 15  # Зазор между билетами

        # Нижний колонтитул с упоминанием автора
        c.setFont("DejaVuSans", 8)
        c.setFillColor(colors.grey)
        c.drawString(margin_left, margin_bottom - 10, "Сгенерировано с помощью Questions Generator Bot | Made by Aralary")
        
        c.save()
        return filename


    def _clean_ticket_text(self, text: str) -> str:
        """Удаляет строку 'Экзаменационный билет №XXX по...' если она есть."""
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Пропускаем строку, которая начинается с "Экзаменационный билет"
            if not line.strip().startswith("Экзаменационный билет"):
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()


    def _calculate_text_height(self, text: str, max_width: float, font_name: str, font_size: int) -> float:
        """Более точно вычисляет высоту текста с учётом переносов."""
        total_height = 0
        
        questions = text.split("\n\n")
        for question in questions:
            if not question.strip():
                continue
            
            wrapped_lines = self._wrap_text(question.strip(), max_width, font_name, font_size)
            total_height += len(wrapped_lines) * 14  # 14 пикселей на строку
            total_height += 5  # Зазор между вопросами
        
        return total_height


    def _wrap_text(self, text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
        """Переносит текст по ширине."""
        from reportlab.pdfbase import pdfmetrics
        
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_width = pdfmetrics.stringWidth(test_line, font_name, font_size)

            if test_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]
