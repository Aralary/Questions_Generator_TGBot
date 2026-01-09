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

    def _normalize_math_symbols(self, text: str) -> str:
        """Заменяет математические Unicode на обычные символы."""
        replacements = {
            # ═══ Blackboard Bold (множества) ═══
            '𝔽': 'F', 'ℤ': 'Z', 'ℕ': 'N', 'ℚ': 'Q', 'ℝ': 'R', 'ℂ': 'C', '𝔾': 'G',
            
            # ═══ Mathematical Italic - ГРЕЧЕСКИЕ ═══
            '𝜉': 'ξ', '𝜂': 'η', '𝜁': 'ζ', '𝜃': 'θ', '𝛼': 'α', '𝛽': 'β', 
            '𝛾': 'γ', '𝛿': 'δ', '𝜀': 'ε', '𝜆': 'λ', '𝜇': 'μ', '𝜈': 'ν',
            '𝜋': 'π', '𝜌': 'ρ', '𝜎': 'σ', '𝜏': 'τ', '𝜑': 'φ', '𝜒': 'χ',
            '𝜓': 'ψ', '𝜔': 'ω',
            
            # ═══ Mathematical Italic - ЛАТИНСКИЕ (малые) ═══
            '𝑎': 'a', '𝑏': 'b', '𝑐': 'c', '𝑑': 'd', '𝑒': 'e', '𝑓': 'f',
            '𝑔': 'g', '𝒉': 'h', '𝑖': 'i', '𝑗': 'j', '𝑘': 'k', '𝑙': 'l',
            '𝑚': 'm', '𝑛': 'n', '𝑜': 'o', '𝑝': 'p', '𝑞': 'q', '𝑟': 'r',
            '𝑠': 's', '𝑡': 't', '𝑢': 'u', '𝑣': 'v', '𝑤': 'w', '𝑥': 'x',
            '𝑦': 'y', '𝑧': 'z',
            
            # ═══ Mathematical Italic - ЛАТИНСКИЕ (большие) ═══
            '𝐴': 'A', '𝐵': 'B', '𝐶': 'C', '𝐷': 'D', '𝐸': 'E', '𝐹': 'F',
            '𝐺': 'G', '𝐻': 'H', '𝐼': 'I', '𝐽': 'J', '𝐾': 'K', '𝐿': 'L',
            '𝑀': 'M', '𝑁': 'N', '𝑂': 'O', '𝑃': 'P', '𝑄': 'Q', '𝑅': 'R',
            '𝑆': 'S', '𝑇': 'T', '𝑈': 'U', '𝑉': 'V', '𝑊': 'W', '𝑋': 'X',
            '𝑌': 'Y', '𝑍': 'Z',
            
            # ═══ Индексы ═══
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            
            # ═══ Операторы ═══
            '⊕': '⊕', '⊗': '⊗', '×': '×', '∈': '∈', '⊆': '⊆', '∉': '∉', '←': '←', '∑': 'Σ', '∏': 'Π'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def save_as_pdf(self, tickets: List[str], domain_key: str, num_questions: int) -> Path:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from datetime import datetime
        
        filename = OUTPUT_DIR / f"tickets_{domain_key}_{num_questions}q.pdf"
        
        # Регистрируем шрифт
        font_path = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"
        if font_path.exists():
            pdfmetrics.registerFont(TTFont('DejaVuSans', str(font_path)))
            font_name = 'DejaVuSans'
        else:
            font_name = 'Helvetica'
            print(f"⚠️ Шрифт DejaVuSans.ttf не найден в {font_path}")
        
        # Создаем документ
        doc = SimpleDocTemplate(
            str(filename),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Стили
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        ticket_number_style = ParagraphStyle(
            'TicketNumber',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=14,
            spaceAfter=12
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            alignment=TA_LEFT,
            spaceAfter=16,
            leading=16
        )
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=8,
            alignment=TA_CENTER
        )
        
        story = []
        
        # Заголовок
        story.append(Paragraph("Экзаменационные билеты", title_style))
        
        domain_name = DOMAINS.get(domain_key, domain_key)
        current_date = datetime.now().strftime("%d.%m.%Y")
        subtitle = f"Дисциплина: {domain_name}  |  Вопросов в билете: {num_questions}  |  Дата: {current_date}"
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 0.5*cm))
        
        # Билеты с ОБЯЗАТЕЛЬНОЙ нормализацией
        for ticket_idx, ticket_text in enumerate(tickets, 1):
            # КРИТИЧЕСКИ ВАЖНО: вызываем нормализацию!
            normalized_ticket = self._normalize_math_symbols(ticket_text)
            clean_text = self._clean_ticket_text(normalized_ticket)
            
            # Номер билета
            story.append(Paragraph(f"Билет № {ticket_idx}", ticket_number_style))
            
            # Вопросы
            questions = clean_text.split('\n\n')
            for question in questions:
                question = question.strip()
                if not question:
                    continue
                
                # Экранируем HTML
                question = question.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(question, question_style))
            
            if ticket_idx < len(tickets):
                story.append(PageBreak())
        
        # Футер
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("Сгенерировано с помощью Questions Generator Bot | Made by Aralary", footer_style))
        
        doc.build(story)
        print(f"✓ PDF сохранен: {filename}")
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
