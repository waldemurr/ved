#!/usr/bin/env python3
"""
Скрипт для парсинга дампа ТН ВЭД с битыми переносами (из PDF).
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TNVEDParserV2:
    """Парсер ТН ВЭД с предварительной склейкой строк."""

    # Паттерны для определения, что строка начинается с нового узла
    # Узел может начинаться с:
    # - Римской цифры (раздел): "I |"
    # - Двузначного кода с пробелом: "01 |" или "01 "
    # - Кода с отступом: "  0101 |" или "    0101210000 |"
    NODE_START_PATTERNS = [
        re.compile(r'^[IVXLCDM]+\s*\|'),  # Римская цифра + |
        re.compile(r'^\s*\d{2}\s*\|'),    # 2 цифры + |
        re.compile(r'^\s*\d{4}\s*\|'),    # 4 цифры + |
        re.compile(r'^\s*\d{6}\s*\|'),    # 6 цифр + |
        re.compile(r'^\s*\d{8}\s*\|'),    # 8 цифр + |
        re.compile(r'^\s*\d{10}\s*\|'),   # 10 цифр + |
        re.compile(r'^\s*\d{2}\s+'),      # 2 цифры + пробел (без |)
        re.compile(r'^\s*\d{4}\s+'),      # 4 цифры + пробел
        re.compile(r'^\s*\d{6}\s+'),      # 6 цифр + пробел
        re.compile(r'^\s*\d{8}\s+'),      # 8 цифр + пробел
        re.compile(r'^\s*\d{10}\s+'),     # 10 цифр + пробел
    ]

    # Паттерны для определения уровня по отступу
    INDENT_PATTERNS = {
        0: 'section',      # "I |"
        2: 'group',        # "  01 |"
        4: 'position',     # "    0101 |"
        6: 'subposition',  # "      010121 |"
        8: 'subsubposition', # "        0101210000 |"
        10: 'code',        # "          0101210000 |"
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.nodes = []
        self.code_map = {}

    def _fix_line_breaks(self, lines: List[str]) -> List[str]:
        """
        Склеивает строки, которые были разорваны в PDF.
        """
        fixed = []
        buffer = ""

        for line in lines:
            stripped = line.strip()

            # Пропускаем пустые строки
            if not stripped:
                continue

            # Проверяем, начинается ли строка с нового узла
            is_new_node = False
            for pattern in self.NODE_START_PATTERNS:
                if pattern.match(stripped):
                    is_new_node = True
                    break

            # Проверяем, не является ли строка служебной (заголовки, примечания)
            is_metadata = any([
                stripped.startswith('СПРАВОЧНЫЙ'),
                stripped.startswith('Источник:'),
                stripped.startswith('Дата получения:'),
                stripped.startswith('Версия источника:'),
                stripped.startswith('Состав:'),
                stripped.startswith('РАЗДЕЛ'),
                stripped.startswith('ПРИМЕЧАНИЯ'),
                stripped.startswith('ГРУППА'),
                stripped.startswith('ИЕРАРХИЯ'),
                stripped.startswith('ПОЯСНЕНИЯ'),
                stripped.startswith('ОБЩИЕ ПОЛОЖЕНИЯ'),
                stripped.startswith('Примечания:'),
                stripped.startswith('В данную группу'),
                stripped.startswith('(') and ')' in stripped and len(stripped) < 50,
            ])

            if is_metadata:
                continue

            if is_new_node:
                # Сохраняем накопленный буфер
                if buffer:
                    fixed.append(buffer.strip())
                buffer = stripped
            else:
                # Склеиваем с предыдущей строкой
                if buffer:
                    buffer += " " + stripped
                else:
                    buffer = stripped

        # Добавляем последний буфер
        if buffer:
            fixed.append(buffer.strip())

        return fixed

    def _parse_line(self, line: str):
        """Парсит одну строку иерархии."""
        stripped = line.strip()

        # Определяем уровень по отступу
        indent = len(line) - len(line.lstrip(' '))

        # Нормализуем отступ: если отступ нестандартный, определяем по первому символу
        if indent not in self.INDENT_PATTERNS:
            # Пробуем определить по наличию кода
            if re.match(r'^[IVXLCDM]+\s*\|', stripped):
                indent = 0
            elif re.match(r'^\s*\d{2}\s*[|\s]', stripped):
                indent = 2
            elif re.match(r'^\s*\d{4}\s*[|\s]', stripped):
                indent = 4
            elif re.match(r'^\s*\d{6}\s*[|\s]', stripped):
                indent = 6
            elif re.match(r'^\s*\d{8}\s*[|\s]', stripped):
                indent = 8
            elif re.match(r'^\s*\d{10}\s*[|\s]', stripped):
                indent = 10

        level = self.INDENT_PATTERNS.get(indent, 'unknown')

        # Разбираем строку на код и описание
        # Формат: "01 | ЖИВЫЕ ЖИВОТНЫЕ" или "01 ЖИВЫЕ ЖИВОТНЫЕ" или "0101210000 | Лошади..."
        parts = re.split(r'\s*\|\s*', stripped, maxsplit=1)

        code = None
        description = ""

        if len(parts) >= 2:
            code_part = parts[0].strip()
            description = parts[1].strip()
            # Извлекаем код
            code_match = re.search(r'([IVXLCDM]+|\d{2,10})', code_part)
            if code_match:
                code = code_match.group(1)
        else:
            # Нет разделителя, пытаемся извлечь код из начала строки
            code_match = re.search(r'^([IVXLCDM]+|\d{2,10})', stripped)
            if code_match:
                code = code_match.group(1)
                description = stripped.replace(code, '', 1).strip()
                # Убираем лишние символы
                description = re.sub(r'^[–\-\s\|]+', '', description).strip()

        if not code:
            logger.warning(f"Не удалось извлечь код из: {line[:100]}...")
            return

        # Определяем родителя
        parent_code = None
        if level == 'group':
            parent_code = 'I'  # Все группы в разделе I
            # Пытаемся найти реальный раздел
            for node in reversed(self.nodes):
                if node['level'] == 'section':
                    parent_code = node['code']
                    break
        elif level == 'position':
            parent_code = code[:2]
        elif level == 'subposition':
            parent_code = code[:4]
        elif level == 'subsubposition':
            parent_code = code[:6]
        elif level == 'code':
            parent_code = code[:8]

        node = {
            'code': code,
            'description': description,
            'level': level,
            'parent_code': parent_code,
            'raw_text': line,
            'indent': indent,
        }
        self.nodes.append(node)
        self.code_map[code] = node

        logger.debug(f"{level}: {code} - {description[:50]}...")

    def parse(self) -> pd.DataFrame:
        """Основной метод парсинга."""
        logger.info(f"Чтение файла: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        logger.info(f"Прочитано {len(raw_lines)} строк")

        # Склеиваем битые строки
        logger.info("Склейка битых строк...")
        fixed_lines = self._fix_line_breaks(raw_lines)
        logger.info(f"Получено {len(fixed_lines)} строк после склейки")

        # Парсим каждую строку
        logger.info("Парсинг строк...")
        for line in fixed_lines:
            self._parse_line(line)

        logger.info(f"Парсинг завершён. Всего узлов: {len(self.nodes)}")
        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Преобразует узлы в DataFrame."""
        if not self.nodes:
            return pd.DataFrame()

        df = pd.DataFrame(self.nodes)

        # Добавляем иерархические уровни
        for length in [2, 4, 6, 8]:
            col = f'level_{length}'
            df[col] = df['code'].apply(lambda x: x[:length] if isinstance(x, str) and x.isdigit() else x)

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Парсер ТН ВЭД с битыми переносами')
    parser.add_argument('--input', '-i', required=True, help='Путь к файлу дампа')
    parser.add_argument('--output', '-o', default='tnved_parsed.csv', help='Путь для сохранения CSV')
    args = parser.parse_args()

    tnved_parser = TNVEDParserV2(args.input)
    df = tnved_parser.parse()

    df.to_csv(args.output, index=False, encoding='utf-8')
    logger.info(f"Сохранено в {args.output}")

    logger.info(f"\n{'='*60}")
    logger.info(f"СТАТИСТИКА:")
    logger.info(f"  Всего узлов: {len(df)}")
    if not df.empty:
        logger.info(f"  По уровням:\n{df['level'].value_counts()}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()