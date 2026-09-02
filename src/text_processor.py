import re
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Обработка и нормализация текстов"""
    
    def __init__(self):
        # Стоп-слова для русского языка
        self.stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'из', 'от', 'до', 'при',
            'без', 'через', 'около', 'возле', 'после', 'перед', 'за',
            'над', 'под', 'об', 'про', 'у', 'о', 'к', 'ко', 'со'
        }
        
        # Паттерны для извлечения информации
        self.patterns = {
            'code': re.compile(r'(?:код|тн вэд|товарная позиция)\s*(\d{4,10})', re.IGNORECASE),
            'weight': re.compile(r'(\d+[.,]?\d*)\s*(?:кг|г|т|тонн)', re.IGNORECASE),
            'size': re.compile(r'(\d+)\s*[xX×]\s*(\d+)', re.IGNORECASE),
            'year': re.compile(r'(19|20)\d{2}', re.IGNORECASE),
            'material': re.compile(r'(?:из|сталь|алюминий|медь|пластик|стекло|дерево|бумага|хлопок|полиэфир)', re.IGNORECASE),
        }
        
    def normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        if pd.isna(text) or not text:
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем специальные символы, но сохраняем буквы и цифры
        text = re.sub(r'[^а-яёa-z0-9\s\-\.]', ' ', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def prepare_declaration_text(self, row: pd.Series) -> str:
        """Подготовка текста декларации"""
        parts = []
        
        if pd.notna(row.get('G31_1')):
            title = row['G31_1']
            parts.append(title)
        
        if pd.notna(row.get('desc_extention')):
            desc = row['desc_extention']
            important = self.extract_important_features(desc)
            if important:
                parts.append(important)
        
        if pd.notna(row.get('G34')):
            parts.append(f"страна {row['G34']}")
        
        text = ' '.join(parts)
        text = self.normalize_text(text)
        
        return f"[НАЗВАНИЕ] {text}"
    
    def prepare_regulation_text(self, row: pd.Series) -> str:
        """Подготовка текста регуляции"""
        parts = []
        if pd.notna(row.get('description')):
            desc = row['description']
            parts.append(f"[ОПИСАНИЕ] {desc}")
        
        if pd.notna(row.get('explanation')):
            expl = row['explanation']
            parts.append(f"[ПОЯСНЕНИЕ] {expl}")
        
        if pd.notna(row.get('notes')):
            notes = row['notes']
            if len(notes) > 500:
                notes = notes[:500] + '...'
            parts.append(f"[ПРИМЕЧАНИЯ] {notes}")
        
        if pd.notna(row.get('code')):
            parts.append(f"[КОД] {row['code']}")
        
        text = ' '.join(parts)
        text = self.normalize_text(text)
        
        return text
    
    def extract_important_features(self, text: str) -> str:
        """Извлечение важных характеристик из текста"""
        features = []
        for key, pattern in self.patterns.items():
            if key == 'code':
                continue
            matches = pattern.findall(text)
            if matches:
                features.append(f"{key}: {matches[0]}")
        keywords = ['новый', 'бывший', 'употребление', 'оригинальный', 
                   'коллекционный', 'прямой', 'отжим', 'нерафинированный',
                   'полуобработанный', 'обработанный', 'цельный']
        
        for kw in keywords:
            if kw.lower() in text.lower():
                features.append(f"[{kw}]")
        
        return ' '.join(features) if features else text
    
    def extract_hs_code_hints(self, text: str) -> List[str]:
        """Извлечение упоминаний кодов ТН ВЭД из текста"""
        hints = []
        text = self.normalize_text(text)
        matches = self.patterns['code'].findall(text)
        hints.extend(matches)
        group_pattern = re.compile(r'групп[аы]?\s*(\d{2})', re.IGNORECASE)
        matches = group_pattern.findall(text)
        hints.extend(matches)
        
        return list(set(hints))
    
    def create_keyword_features(self, text: str) -> set:
        """Создание множества ключевых слов для текста"""
        text = self.normalize_text(text)
        words = set(text.split())
        words = words - self.stop_words
        words = {w for w in words if len(w) > 3}
        
        return words