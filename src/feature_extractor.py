import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class TNVEDTree:
    """Иерархия ТН ВЭД для кодовых признаков."""

    def __init__(self, tree_path: Optional[str] = None):
        self.tree = {}
        self.code_to_path = {}
        self.level_codes = {1: set(), 2: set(), 3: set(), 4: set()}

        self.group_categories = {
            '01-05': ['животные', 'продукты животного происхождения', 'лошади', 'скот', 'свин', 'мясо', 'молоко'],
            '06-15': ['растения', 'продукты растительного происхождения', 'зерно', 'фрукты', 'овощи', 'кофе', 'чай', 'сахар'],
            '16-24': ['продукты питания', 'напитки', 'сок', 'сахар', 'крупа', 'мясо', 'рыба', 'молочные'],
            '25-27': ['минеральные продукты', 'нефть', 'газ', 'уголь', 'руды', 'соль'],
            '28-38': ['химические продукты', 'лекарственные', 'витамины', 'косметика', 'удобрения', 'краски'],
            '39-40': ['пластмассы', 'резина', 'полимеры', 'пластик'],
            '41-43': ['кожа', 'мех', 'шкуры', 'сумки'],
            '44-46': ['древесина', 'изделия из дерева', 'поддон', 'паллет', 'бумага', 'картон'],
            '47-49': ['целлюлоза', 'бумага', 'книги', 'картон'],
            '50-63': ['текстиль', 'одежда', 'ткань', 'волокно', 'хлопок', 'костюм', 'трикотаж'],
            '64-67': ['обувь', 'головные уборы', 'сапоги', 'ботинки'],
            '68-70': ['изделия из камня', 'стекло', 'керамика', 'цемент'],
            '71': ['драгоценные металлы', 'серебро', 'золото', 'платина', 'украшения'],
            '72-83': ['металлы', 'изделия из металлов', 'сталь', 'алюминий', 'медь', 'железо'],
            '84-85': ['машины', 'электрооборудование', 'двигатель', 'электрический', 'компьютер'],
            '86-89': ['транспортные средства', 'автомобиль', 'экскаватор', 'гусеничный', 'самолет', 'корабль'],
            '90-92': ['оптика', 'часы', 'измерительный', 'приборы', 'медицинские инструменты'],
            '93': ['оружие', 'патроны', 'боеприпасы'],
            '94-96': ['мебель', 'игрушки', 'разное', 'спортивные', 'светильники'],
            '97': ['произведения искусства', 'марка', 'офорт', 'гравюра', 'антиквариат'],
        }

        self.code_to_category = {}
        for group in self.group_categories:
            start = int(group.split('-')[0]) if '-' in group else int(group)
            end = int(group.split('-')[1]) if '-' in group else start
            for code_num in range(start, end + 1):
                self.code_to_category[str(code_num).zfill(2)] = group

        if tree_path:
            self.load_tree(tree_path)

    def load_tree(self, tree_path: str):
        """Загрузка дерева из файла (опционально)."""
        logger.info(f"Загрузка дерева ТН ВЭД из {tree_path}")
        try:
            with open(tree_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.warning(f"Файл {tree_path} не найден, используем встроенную иерархию")
            return

        current_path = []
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('СПРАВОЧНЫЙ') or line.startswith('Источник') or line.startswith('Дата'):
                continue

            indent = len(line) - len(line.lstrip())
            level = indent // 2
            line = re.sub(r'^\s*[\d\s\-|–—]+', '', line).strip()
            if not line:
                continue

            code_match = re.search(r'\b(\d{4,10})\b', line)
            if code_match:
                code = code_match.group(1)
                name = re.sub(r'\b\d{4,10}\b', '', line).strip()
                name = re.sub(r'[–—|-]\s*$', '', name).strip()

                self.code_to_path[code] = current_path + [code]
                if level not in self.level_codes:
                    self.level_codes[level] = set()
                self.level_codes[level].add(code)

                if len(current_path) > level:
                    current_path = current_path[:level] + [code]
                else:
                    current_path.append(code)

                self.tree[code] = {
                    'code': code,
                    'name': name,
                    'level': level,
                    'path': current_path.copy(),
                    'parent': current_path[-2] if len(current_path) > 1 else None,
                }
            else:
                if 'РАЗДЕЛ' in line or 'ГРУППА' in line:
                    current_path = []

        logger.info(f"Загружено {len(self.tree)} узлов дерева ТН ВЭД")

    def get_category(self, code: str) -> Optional[str]:
        if not code:
            return None
        prefix = code[:2]
        return self.code_to_category.get(prefix)

    def get_category_keywords(self, category: str) -> List[str]:
        return self.group_categories.get(category, [])

    def get_level(self, code: str) -> int:
        if not code:
            return 0
        for level in sorted(self.level_codes.keys(), reverse=True):
            if code in self.level_codes[level]:
                return level
        return 0


class HybridFeatureExtractor:
    """Извлечение признаков: TF-IDF + опционально FastText/SBERT + иерархия ТН ВЭД."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        fasttext_path: Optional[str] = None,
        use_gpu: bool = False,
        tnved_tree_path: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        self.sbert_model = None
        self.ft_model = None
        self.ft_dim = 0
        self.sbert_dim = 0

        # Попытка загрузить SBERT (если доступен)
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(model_name)
                if use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        self.sbert_model = self.sbert_model.to('cuda')
                self.sbert_dim = self.sbert_model.get_sentence_embedding_dimension()
                logger.info(f"Загружена SBERT модель {model_name}, dim={self.sbert_dim}")
            except Exception as e:
                logger.warning(f"SBERT не доступен ({e}), используем только TF-IDF/FastText")

        # Попытка загрузить FastText (если доступен)
        if fasttext_path and Path(fasttext_path).exists():
            try:
                import fasttext
                self.ft_model = fasttext.load_model(str(fasttext_path))
                self.ft_dim = self.ft_model.get_dimension()
                logger.info(f"Загружена FastText модель {fasttext_path}, dim={self.ft_dim}")
            except Exception as e:
                logger.warning(f"FastText не доступен ({e})")

        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            lowercase=True,
            analyzer='word',
            token_pattern=r'(?u)\b[а-яёa-z0-9]{2,}\b',
        )

        self.char_tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(2, 5),
            lowercase=True,
            analyzer='char_wb',
        )

        self.reg_df = None
        self.reg_ids = None
        self.reg_tfidf = None
        self.reg_char_tfidf = None
        self.reg_sbert = None
        self.reg_fasttext = None
        self.reg_codes = None
        self.reg_hierarchy = None

        self.tnved_tree = TNVEDTree(tnved_tree_path)

        self.code_pattern = re.compile(r'(?:код\s*тн\s*вэд|тн\s*вэд|товарная\s*позиция|код)\s*:?\s*(\d{4,10})', re.IGNORECASE)
        self.weight_pattern = re.compile(r'(\d+[.,]?\d*)\s*(?:кг|г|т|тонн)', re.IGNORECASE)
        self.year_pattern = re.compile(r'(19|20)\d{2}')

    def _normalize(self, text) -> str:
        if pd.isna(text) or not text:
            return ''
        text = str(text).lower()
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _prepare_regulation_text(self, row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get('description')):
            parts.append(str(row['description']))
        if pd.notna(row.get('explanation')):
            parts.append(str(row['explanation']))
        if pd.notna(row.get('notes')):
            notes = str(row['notes'])
            parts.append(notes[:500])
        if pd.notna(row.get('code')):
            parts.append(str(row['code']))
        return self._normalize(' '.join(parts))

    def _prepare_declaration_text(self, row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get('G31_1')):
            parts.append(str(row['G31_1']))
        if pd.notna(row.get('desc_extention')):
            parts.append(str(row['desc_extention']))
        if pd.notna(row.get('G34')):
            parts.append(f"страна {row['G34']}")
        if pd.notna(row.get('G32')):
            parts.append(str(row['G32']))
        return self._normalize(' '.join(parts))

    def _fasttext_embedding(self, text: str) -> np.ndarray:
        if self.ft_model is None:
            return np.zeros(self.ft_dim)
        words = text.split()
        if not words:
            return np.zeros(self.ft_dim)
        vectors = []
        for word in words:
            try:
                vectors.append(self.ft_model.get_word_vector(word))
            except Exception:
                continue
        if not vectors:
            return np.zeros(self.ft_dim)
        vec = np.mean(vectors, axis=0)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _sbert_embedding(self, text: str) -> np.ndarray:
        if self.sbert_model is None:
            return np.zeros(self.sbert_dim)
        vec = self.sbert_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec

    def fit_regulations(self, reg_df: pd.DataFrame):
        logger.info("Кэширование признаков регуляций...")
        self.reg_df = reg_df.copy().reset_index(drop=True)
        self.reg_ids = self.reg_df['regulation_id'].values
        self.reg_codes = self.reg_df['code'].values if 'code' in self.reg_df.columns else np.array([None] * len(self.reg_df))

        reg_texts = [self._prepare_regulation_text(row) for _, row in self.reg_df.iterrows()]

        logger.info("  - TF-IDF")
        self.reg_tfidf = self.tfidf.fit_transform(reg_texts)
        self.reg_char_tfidf = self.char_tfidf.fit_transform(reg_texts)

        if self.sbert_model is not None:
            logger.info("  - SBERT эмбеддинги")
            self.reg_sbert = self.sbert_model.encode(
                reg_texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
            )

        if self.ft_model is not None:
            logger.info("  - FastText эмбеддинги")
            self.reg_fasttext = np.array([self._fasttext_embedding(t) for t in reg_texts])

        self.reg_hierarchy = []
        for code in self.reg_codes:
            self.reg_hierarchy.append({
                'category': self.tnved_tree.get_category(str(code)) if code else None,
                'level': self.tnved_tree.get_level(str(code)) if code else 0,
            })

        logger.info(f"Закэшировано {len(self.reg_df)} регуляций")

    def _extract_codes(self, text: str) -> List[str]:
        return self.code_pattern.findall(text)

    def _code_prefix_score(self, decl_text: str, reg_code: str) -> int:
        if not reg_code:
            return 0
        matches = self._extract_codes(decl_text)
        best = 0
        for match in matches:
            if reg_code.startswith(match) or match.startswith(reg_code):
                common = 0
                for a, b in zip(reg_code, match):
                    if a == b:
                        common += 1
                    else:
                        break
                best = max(best, common)
        return best

    def _common_ngrams(self, text1: str, text2: str, n: int = 2) -> int:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if len(words1) < n or len(words2) < n:
            return 0
        ngrams1 = set(zip(*[list(words1)[i:] for i in range(n)]))
        ngrams2 = set(zip(*[list(words2)[i:] for i in range(n)]))
        return len(ngrams1 & ngrams2)

    def _common_char_bigrams(self, text1: str, text2: str) -> int:
        def bigrams(text):
            return set(text[i:i+2] for i in range(len(text) - 1))
        return len(bigrams(text1) & bigrams(text2))

    def _extract_weight(self, text: str) -> float:
        match = self.weight_pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                return 0.0
        return 0.0

    def _category_match(self, decl_text: str, reg_code: str) -> float:
        if not reg_code:
            return 0.0
        category = self.tnved_tree.get_category(reg_code)
        if not category:
            return 0.0
        keywords = self.tnved_tree.get_category_keywords(category)
        decl_words = decl_text.split()
        matched = 0
        for kw in keywords:
            for word in decl_words:
                if kw in word or word in kw:
                    matched += 1
                    break
        return min(1.0, matched / max(1, len(keywords) / 3))

    def extract_features_for_pair(self, decl_row: pd.Series, reg_row_or_idx) -> Dict[str, float]:
        """Извлечение признаков для пары декларация-регуляция.
        reg_row_or_idx: либо pd.Series с данными регуляции, либо индекс в reg_df.
        """
        features = {}

        if isinstance(reg_row_or_idx, int):
            reg_idx = reg_row_or_idx
            reg_row = self.reg_df.iloc[reg_idx]
            reg_text = self._prepare_regulation_text(reg_row)
        else:
            reg_row = reg_row_or_idx
            reg_text = self._prepare_regulation_text(reg_row)
            reg_idx = None
            if self.reg_df is not None and 'regulation_id' in reg_row:
                matches = np.where(self.reg_ids == reg_row['regulation_id'])[0]
                if len(matches) > 0:
                    reg_idx = matches[0]

        decl_text = self._prepare_declaration_text(decl_row)
        reg_code = str(reg_row.get('code', '')) if pd.notna(reg_row.get('code', '')) else ''

        # 1. TF-IDF сходство
        if self.reg_tfidf is not None and reg_idx is not None:
            decl_tfidf = self.tfidf.transform([decl_text])
            features['tfidf_cosine'] = float((decl_tfidf * self.reg_tfidf[reg_idx].T).toarray()[0, 0])
            decl_char = self.char_tfidf.transform([decl_text])
            features['char_tfidf_cosine'] = float((decl_char * self.reg_char_tfidf[reg_idx].T).toarray()[0, 0])
        else:
            features['tfidf_cosine'] = 0.0
            features['char_tfidf_cosine'] = 0.0

        # 2. FastText сходство
        if self.ft_model is not None and reg_idx is not None:
            decl_ft = self._fasttext_embedding(decl_text)
            reg_ft = self.reg_fasttext[reg_idx]
            denom = np.linalg.norm(decl_ft) * np.linalg.norm(reg_ft) + 1e-8
            features['fasttext_cosine'] = float(np.dot(decl_ft, reg_ft) / denom)
        else:
            features['fasttext_cosine'] = 0.0

        # 3. SBERT сходство
        if self.sbert_model is not None and reg_idx is not None:
            decl_sbert = self._sbert_embedding(decl_text)
            reg_sbert = self.reg_sbert[reg_idx]
            denom = np.linalg.norm(decl_sbert) * np.linalg.norm(reg_sbert) + 1e-8
            features['sbert_cosine'] = float(np.dot(decl_sbert, reg_sbert) / denom)
        else:
            features['sbert_cosine'] = 0.0

        # 4. Кодовые признаки
        prefix_len = self._code_prefix_score(decl_text, reg_code)
        features['code_prefix_len'] = prefix_len
        for length in [2, 4, 6, 8, 10]:
            features[f'code_match_{length}'] = 1.0 if prefix_len >= length else 0.0

        # 5. Иерархия ТН ВЭД
        hierarchy = self.reg_hierarchy[reg_idx] if reg_idx is not None else {'category': None, 'level': 0}
        features['category_match'] = self._category_match(decl_text, reg_code)
        features['code_level'] = hierarchy['level']

        # 6. Текстовые метрики
        decl_words = set(decl_text.split())
        reg_words = set(reg_text.split())
        common_words = decl_words & reg_words
        features['common_words'] = len(common_words)
        features['common_word_ratio'] = len(common_words) / max(1, len(decl_words | reg_words))
        features['common_bigrams'] = self._common_ngrams(decl_text, reg_text, 2)
        features['common_char_bigrams'] = self._common_char_bigrams(decl_text, reg_text)
        features['len_ratio'] = len(decl_text) / (len(reg_text) + 1)

        # Jaccard и направленные перекрытия
        union = decl_words | reg_words
        features['jaccard'] = len(common_words) / max(1, len(union))
        features['decl_words_in_reg'] = len(common_words) / max(1, len(decl_words))
        features['reg_words_in_decl'] = len(common_words) / max(1, len(reg_words))

        # 7. Страна
        decl_country = str(decl_row.get('G34', '')) if pd.notna(decl_row.get('G34', '')) else ''
        reg_country = str(reg_row.get('country', '')) if pd.notna(reg_row.get('country', '')) else ''
        features['country_match'] = 1.0 if decl_country and reg_country and decl_country == reg_country else 0.0

        # 8. Вес/количество
        decl_weight = self._extract_weight(decl_text)
        reg_weight = self._extract_weight(reg_text)
        features['decl_weight'] = decl_weight
        features['reg_weight'] = reg_weight
        if decl_weight > 0 and reg_weight > 0:
            features['weight_ratio'] = min(decl_weight, reg_weight) / max(decl_weight, reg_weight)
        else:
            features['weight_ratio'] = 0.0
        features['has_weight'] = 1.0 if decl_weight > 0 else 0.0
        features['has_year'] = 1.0 if self.year_pattern.search(decl_text) else 0.0
        features['has_measurements'] = 1.0 if re.search(r'\d+\s*[xX×]\s*\d+', str(decl_row.get('desc_extention', ''))) else 0.0

        return features
