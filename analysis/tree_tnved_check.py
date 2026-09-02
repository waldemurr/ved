#!/usr/bin/env python3
"""
Скрипт для парсинга дампа ТН ВЭД ЕАЭС.
Поддерживает иерархическую структуру: разделы → группы → позиции → субпозиции → подсубпозиции.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TNVEDNode:
    """Узел дерева ТН ВЭД."""
    code: str                    # Полный код (4, 6, 8, 10 знаков)
    description: str             # Описание
    level: str                   # section, group, position, subposition, subsubposition, code
    parent_code: Optional[str] = None
    children: List['TNVEDNode'] = field(default_factory=list)
    raw_text: str = ""


class TNVEDParser:
    """Парсер дампа ТН ВЭД ЕАЭС."""

    # Паттерны для определения уровня
    PATTERNS = {
        'section': re.compile(r'^([IVXLCDM]+)\s*\|\s*(.+?)(?:\s*\((\d+)-(\d+)\))?$'),
        'group': re.compile(r'^\s*(\d{2})\s*\|\s*(.+?)(?:\s*\((\d+)-(\d+)\))?$'),
        'position_4': re.compile(r'^\s*(\d{4})(?:\s+\|\s+(.+?))?$'),
        'subposition_6': re.compile(r'^\s*(\d{6})(?:\s+\|\s+(.+?))?$'),
        'subsubposition_8': re.compile(r'^\s*(\d{8})(?:\s+\|\s+(.+?))?$'),
        'code_10': re.compile(r'^\s*(\d{10})(?:\s+\|\s+(.+?))?$'),
    }

    # Дополнительный паттерн для строк с дефисами (подпункты)
    LINE_PATTERN = re.compile(
        r'^\s*(?P<prefix>[-–]+\s*)?'
        r'(?P<code>\d{4,10})?'
        r'(?:\s*\|\s*)?'
        r'(?P<desc>.+?)?$'
    )

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.nodes: List[TNVEDNode] = []
        self.code_map: Dict[str, TNVEDNode] = {}
        self.current_section: Optional[TNVEDNode] = None
        self.current_group: Optional[TNVEDNode] = None
        self.current_position: Optional[TNVEDNode] = None
        self.current_subposition: Optional[TNVEDNode] = None
        self.current_subsubposition: Optional[TNVEDNode] = None

    def parse(self) -> pd.DataFrame:
        """Основной метод парсинга."""
        logger.info(f"Начинаем парсинг файла: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Пропускаем заголовки
        start_idx = 0
        for i, line in enumerate(lines):
            if 'ИЕРАРХИЯ ТН ВЭД' in line.upper():
                start_idx = i + 1
                break

        for line in lines[start_idx:]:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            # Пропускаем служебные строки
            if line.startswith('СПРАВОЧНЫЙ') or line.startswith('Источник:') or line.startswith('Дата получения:'):
                continue
            if line.startswith('РАЗДЕЛ') or line.startswith('ПРИМЕЧАНИЯ') or line.startswith('ГРУППА'):
                continue
            if 'ИЕРАРХИЯ ТН ВЭД' in line:
                continue

            self._parse_line(line.strip())

        logger.info(f"Парсинг завершён. Всего узлов: {len(self.nodes)}")
        return self._to_dataframe()

    def _parse_line(self, line: str):
        """Парсит одну строку иерархии."""
        # Определяем уровень по отступу (пробелы в начале)
        indent = len(line) - len(line.lstrip())
        stripped = line.lstrip()

        # Определяем уровень по отступу
        if indent == 0:
            self._parse_section(stripped)
        elif indent == 2:
            self._parse_group(stripped)
        elif indent == 4:
            self._parse_position(stripped)
        elif indent == 6:
            self._parse_subposition(stripped)
        elif indent == 8:
            self._parse_subsubposition(stripped)
        elif indent >= 10:
            self._parse_detailed_code(stripped)
        else:
            # Fallback: пытаемся определить по паттерну
            self._parse_unknown(stripped, indent)

    def _parse_section(self, line: str):
        """Парсит раздел (уровень 0)."""
        match = self.PATTERNS['section'].match(line)
        if match:
            roman = match.group(1)
            desc = match.group(2).strip()
            node = TNVEDNode(
                code=roman,
                description=desc,
                level='section',
                raw_text=line
            )
            self.nodes.append(node)
            self.code_map[roman] = node
            self.current_section = node
            self.current_group = None
            self.current_position = None
            self.current_subposition = None
            self.current_subsubposition = None
            logger.debug(f"Раздел: {roman} - {desc}")
        else:
            logger.warning(f"Не удалось распарсить раздел: {line}")

    def _parse_group(self, line: str):
        """Парсит группу (уровень 2 пробела)."""
        # Группы обычно имеют формат: "01 | ЖИВЫЕ ЖИВОТНЫЕ"
        parts = line.split('|', 1)
        if len(parts) == 2:
            code_part = parts[0].strip()
            desc_part = parts[1].strip()
            # Извлекаем код группы (2 цифры)
            code_match = re.match(r'^(\d{2})', code_part)
            if code_match:
                code = code_match.group(1)
                node = TNVEDNode(
                    code=code,
                    description=desc_part,
                    level='group',
                    parent_code=self.current_section.code if self.current_section else None,
                    raw_text=line
                )
                self.nodes.append(node)
                self.code_map[code] = node
                if self.current_section:
                    self.current_section.children.append(node)
                self.current_group = node
                self.current_position = None
                self.current_subposition = None
                self.current_subsubposition = None
                logger.debug(f"Группа: {code} - {desc_part}")
                return

        logger.warning(f"Не удалось распарсить группу: {line}")

    def _parse_position(self, line: str):
        """Парсит позицию (4 знака, уровень 4 пробела)."""
        parts = line.split('|', 1)
        if len(parts) >= 1:
            code_part = parts[0].strip()
            # Ищем 4-значный код
            code_match = re.search(r'\b(\d{4})\b', code_part)
            if code_match:
                code = code_match.group(1)
                desc = parts[1].strip() if len(parts) > 1 else ''
                # Если описание не указано, пытаемся извлечь из первой части
                if not desc:
                    desc = code_part.replace(code, '').strip()
                    if desc.startswith('-'):
                        desc = desc[1:].strip()

                node = TNVEDNode(
                    code=code,
                    description=desc,
                    level='position',
                    parent_code=self.current_group.code if self.current_group else None,
                    raw_text=line
                )
                self.nodes.append(node)
                self.code_map[code] = node
                if self.current_group:
                    self.current_group.children.append(node)
                self.current_position = node
                self.current_subposition = None
                self.current_subsubposition = None
                logger.debug(f"Позиция: {code} - {desc}")
                return

        logger.warning(f"Не удалось распарсить позицию: {line}")

    def _parse_subposition(self, line: str):
        """Парсит субпозицию (6 знаков, уровень 6 пробелов)."""
        parts = line.split('|', 1)
        if len(parts) >= 1:
            code_part = parts[0].strip()
            code_match = re.search(r'\b(\d{6})\b', code_part)
            if code_match:
                code = code_match.group(1)
                desc = parts[1].strip() if len(parts) > 1 else ''
                if not desc:
                    desc = code_part.replace(code, '').strip()
                    if desc.startswith('-'):
                        desc = desc[1:].strip()

                node = TNVEDNode(
                    code=code,
                    description=desc,
                    level='subposition',
                    parent_code=self.current_position.code if self.current_position else None,
                    raw_text=line
                )
                self.nodes.append(node)
                self.code_map[code] = node
                if self.current_position:
                    self.current_position.children.append(node)
                self.current_subposition = node
                self.current_subsubposition = None
                logger.debug(f"Субпозиция: {code} - {desc}")
                return

        logger.warning(f"Не удалось распарсить субпозицию: {line}")

    def _parse_subsubposition(self, line: str):
        """Парсит подсубпозицию (8 знаков, уровень 8 пробелов)."""
        parts = line.split('|', 1)
        if len(parts) >= 1:
            code_part = parts[0].strip()
            code_match = re.search(r'\b(\d{8})\b', code_part)
            if code_match:
                code = code_match.group(1)
                desc = parts[1].strip() if len(parts) > 1 else ''
                if not desc:
                    desc = code_part.replace(code, '').strip()
                    if desc.startswith('-'):
                        desc = desc[1:].strip()

                node = TNVEDNode(
                    code=code,
                    description=desc,
                    level='subsubposition',
                    parent_code=self.current_subposition.code if self.current_subposition else None,
                    raw_text=line
                )
                self.nodes.append(node)
                self.code_map[code] = node
                if self.current_subposition:
                    self.current_subposition.children.append(node)
                self.current_subsubposition = node
                logger.debug(f"Подсубпозиция: {code} - {desc}")
                return

        logger.warning(f"Не удалось распарсить подсубпозицию: {line}")

    def _parse_detailed_code(self, line: str):
        """Парсит детальный код (10 знаков, уровень >= 10 пробелов)."""
        parts = line.split('|', 1)
        if len(parts) >= 1:
            code_part = parts[0].strip()
            code_match = re.search(r'\b(\d{10})\b', code_part)
            if code_match:
                code = code_match.group(1)
                desc = parts[1].strip() if len(parts) > 1 else ''
                if not desc:
                    desc = code_part.replace(code, '').strip()
                    if desc.startswith('-'):
                        desc = desc[1:].strip()

                node = TNVEDNode(
                    code=code,
                    description=desc,
                    level='code',
                    parent_code=self.current_subsubposition.code if self.current_subsubposition else self.current_subposition.code if self.current_subposition else None,
                    raw_text=line
                )
                self.nodes.append(node)
                self.code_map[code] = node
                if self.current_subsubposition:
                    self.current_subsubposition.children.append(node)
                elif self.current_subposition:
                    self.current_subposition.children.append(node)
                logger.debug(f"Код: {code} - {desc}")
                return

        # Если не удалось найти 10-значный код, пробуем 8-значный
        code_match = re.search(r'\b(\d{8})\b', line)
        if code_match:
            self._parse_subsubposition(line)
            return

        logger.warning(f"Не удалось распарсить детальный код: {line}")

    def _parse_unknown(self, line: str, indent: int):
        """Обработка строк с неизвестным форматом."""
        # Пробуем извлечь любой код
        code_match = re.search(r'\b(\d{4,10})\b', line)
        if code_match:
            code = code_match.group(1)
            desc = line.replace(code, '').strip()
            if desc.startswith('|'):
                desc = desc[1:].strip()
            if desc.startswith('-'):
                desc = desc[1:].strip()

            # Определяем уровень по длине кода
            level_map = {
                4: 'position',
                6: 'subposition',
                8: 'subsubposition',
                10: 'code'
            }
            level = level_map.get(len(code), 'unknown')

            parent = None
            if level == 'position':
                parent = self.current_group.code if self.current_group else None
            elif level == 'subposition':
                parent = self.current_position.code if self.current_position else None
            elif level == 'subsubposition':
                parent = self.current_subposition.code if self.current_subposition else None
            elif level == 'code':
                parent = self.current_subsubposition.code if self.current_subsubposition else None

            node = TNVEDNode(
                code=code,
                description=desc,
                level=level,
                parent_code=parent,
                raw_text=line
            )
            self.nodes.append(node)
            self.code_map[code] = node

    def _to_dataframe(self) -> pd.DataFrame:
        """Преобразует узлы в DataFrame."""
        data = []
        for node in self.nodes:
            data.append({
                'code': node.code,
                'description': node.description,
                'level': node.level,
                'parent_code': node.parent_code,
                'raw_text': node.raw_text,
                'full_path': self._get_full_path(node),
            })
        df = pd.DataFrame(data)

        df['level_1'] = df['code'].str[:1]
        df['level_2'] = df['code'].str[:2]
        df['level_4'] = df['code'].str[:4]
        df['level_6'] = df['code'].str[:6]
        df['level_8'] = df['code'].str[:8]
        for level in [2, 4, 6, 8]:
            col = f'parent_{level}'
            df[col] = df['code'].str[:level] if level < 10 else df['code']

        return df

    def _get_full_path(self, node: TNVEDNode) -> str:
        """Возвращает полный путь к узлу."""
        parts = []
        current = node
        while current:
            parts.insert(0, f"{current.code}: {current.description[:30]}")
            current = self.code_map.get(current.parent_code) if current.parent_code else None
        return ' → '.join(parts)


class TNVEDValidator:
    """Валидатор дерева ТН ВЭД."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Dict:
        """Проверяет целостность дерева."""
        results = {
            'total_nodes': len(df),
            'by_level': df['level'].value_counts().to_dict(),
            'orphans': [],
            'duplicates': df[df.duplicated('code', keep=False)],
            'invalid_codes': [],
            'missing_parents': [],
        }
        if len(results['duplicates']) > 0:
            results['duplicates'] = results['duplicates']['code'].tolist()

        for _, row in df.iterrows():
            if pd.notna(row['parent_code']):
                if row['parent_code'] not in df['code'].values:
                    results['missing_parents'].append({
                        'code': row['code'],
                        'parent': row['parent_code']
                    })
        for code in df['code']:
            if not code.isdigit() and not re.match(r'^[IVXLCDM]+$', code):
                results['invalid_codes'].append(code)

        return results


def main():
    """Основная функция."""
    import argparse

    parser = argparse.ArgumentParser(description='Парсер ТН ВЭД ЕАЭС')
    parser.add_argument('--input', '-i', required=True, help='Путь к файлу дампа')
    parser.add_argument('--output', '-o', default='tnved_parsed.csv', help='Путь для сохранения CSV')
    parser.add_argument('--validate', action='store_true', help='Проверить целостность дерева')
    args = parser.parse_args()

    # Парсим
    tnved_parser = TNVEDParser(args.input)
    df = tnved_parser.parse()

    # Сохраняем
    df.to_csv(args.output, index=False, encoding='utf-8')
    logger.info(f"Сохранено в {args.output}")

    # Статистика
    logger.info(f"\n{'='*60}")
    logger.info(f"СТАТИСТИКА:")
    logger.info(f"  Всего узлов: {len(df)}")
    logger.info(f"  По уровням:\n{df['level'].value_counts()}")
    logger.info(f"{'='*60}")

    # Валидация
    if args.validate:
        validator = TNVEDValidator()
        results = validator.validate(df)
        logger.info(f"\n{'='*60}")
        logger.info(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
        logger.info(f"  Дубликатов: {len(results['duplicates'])}")
        if results['duplicates']:
            logger.warning(f"    {results['duplicates'][:10]}...")
        logger.info(f"  Кодов без родителей: {len(results['missing_parents'])}")
        if results['missing_parents']:
            logger.warning(f"    {results['missing_parents'][:10]}")
        logger.info(f"  Невалидных кодов: {len(results['invalid_codes'])}")
        logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()