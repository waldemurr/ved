import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class TNVEDTree:
    """Иерархия ТН ВЭД и группы товаров."""

    GROUP_CATEGORIES = {
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
        '71': ['драгоценные металлы', 'серебро', 'золото', 'платина', 'украшения', 'рубины', 'сапфиры', 'алмазы'],
        '72-83': ['металлы', 'изделия из металлов', 'сталь', 'алюминий', 'медь', 'железо'],
        '84-85': ['машины', 'электрооборудование', 'двигатель', 'электрический', 'компьютер'],
        '86-89': ['транспортные средства', 'автомобиль', 'экскаватор', 'гусеничный', 'самолет', 'корабль'],
        '90-92': ['оптика', 'часы', 'измерительный', 'приборы', 'медицинские инструменты'],
        '93': ['оружие', 'патроны', 'боеприпасы', 'ружья', 'пистолеты', 'гладкоствольные'],
        '94-96': ['мебель', 'игрушки', 'разное', 'спортивные', 'светильники'],
        '97': ['произведения искусства', 'марка', 'офорт', 'гравюра', 'антиквариат'],
    }

    def __init__(self, tree_path: Optional[str] = None):
        self.code_to_category = {}
        self.group_categories = self.GROUP_CATEGORIES
        for group in self.group_categories:
            start = int(group.split('-')[0]) if '-' in group else int(group)
            end = int(group.split('-')[1]) if '-' in group else start
            for code_num in range(start, end + 1):
                self.code_to_category[str(code_num).zfill(2)] = group

    def get_category(self, code: str) -> Optional[str]:
        return self.code_to_category.get(code[:2]) if code else None

    def get_category_keywords(self, category: str) -> List[str]:
        return self.group_categories.get(category, [])


class FeatureExtractor:
    """SOTA feature extractor: SBERT + BM25 + TF-IDF + иерархия ТН ВЭД + FastText."""

    def __init__(
        self,
        sbert_model: Optional[str] = None,
        fasttext_path: Optional[str] = None,
        use_gpu: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        self.sbert = None
        self.cross_encoder = None
        self.bm25 = None
        self.ft_model = None
        self.cache_dir = cache_dir

        self.reg_df: Optional[pd.DataFrame] = None
        self.reg_ids: Optional[np.ndarray] = None
        self.reg_codes: List[Optional[str]] = []
        self.reg_texts: List[str] = []
        self.reg_sbert: Optional[np.ndarray] = None
        self.reg_fasttext: Optional[np.ndarray] = None

        self.tfidf_word = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            analyzer='word',
            token_pattern=r'(?u)\b[а-яёa-z0-9]{2,}\b',
            sublinear_tf=True,
        )
        self.tfidf_char = TfidfVectorizer(
            max_features=4000,
            ngram_range=(3, 5),
            analyzer='char_wb',
            sublinear_tf=True,
        )
        self.reg_tfidf_word = None
        self.reg_tfidf_char = None

        self.tnved_tree = TNVEDTree()

        self.code_pattern = re.compile(
            r'(?:код\s*тн\s*вэд|тн\s*вэд|товарная\s*позиция|код)\s*:?\s*(\d{4,10})',
            re.IGNORECASE,
        )
        self.weight_pattern = re.compile(r'(\d+[.,]?\d*)\s*(?:кг|г|т|тонн)', re.IGNORECASE)
        self.year_pattern = re.compile(r'(19|20)\d{2}')

        if sbert_model:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Загрузка SBERT модели {sbert_model}")
                self.sbert = SentenceTransformer(sbert_model, cache_folder=cache_dir)
                if use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        self.sbert = self.sbert.to('cuda')
                logger.info(f"SBERT загружена: dim={self.sbert.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.warning(f"SBERT недоступна: {e}")

        if fasttext_path and Path(fasttext_path).exists():
            try:
                import fasttext
                logger.info(f"Загрузка FastText модели {fasttext_path}")
                self.ft_model = fasttext.load_model(str(fasttext_path))
                logger.info(f"FastText загружена: dim={self.ft_model.get_dimension()}")
            except Exception as e:
                logger.warning(f"FastText недоступна: {e}")

    def set_cross_encoder(self, model_name: Optional[str], cache_dir: Optional[str] = None):
        if not model_name:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Загрузка cross-encoder модели {model_name}")
            self.cross_encoder = CrossEncoder(model_name, max_length=256, cache_dir=cache_dir)
            logger.info("Cross-encoder загружена")
        except Exception as e:
            logger.warning(f"Cross-encoder недоступна: {e}")

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

    def fit_regulations(self, reg_df: pd.DataFrame):
        logger.info("Кэширование признаков регуляций...")
        self.reg_df = reg_df.copy().reset_index(drop=True)
        self.reg_ids = self.reg_df['regulation_id'].values
        self.reg_codes = [str(c) if pd.notna(c) else None for c in self.reg_df.get('code', [None] * len(self.reg_df))]
        self.reg_texts = [self._prepare_regulation_text(row) for _, row in self.reg_df.iterrows()]

        logger.info("  - TF-IDF word/char")
        self.reg_tfidf_word = self.tfidf_word.fit_transform(self.reg_texts)
        self.reg_tfidf_char = self.tfidf_char.fit_transform(self.reg_texts)

        if self.sbert is not None:
            logger.info("  - SBERT эмбеддинги")
            self.reg_sbert = self.sbert.encode(
                self.reg_texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=32,
            )

        if self.ft_model is not None:
            logger.info("  - FastText эмбеддинги")
            self.reg_fasttext = np.array([self._fasttext_embedding(t) for t in self.reg_texts])

        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.split() for t in self.reg_texts]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("  - BM25 индекс построен")
        except Exception as e:
            logger.warning(f"BM25 недоступен: {e}")

        logger.info(f"Закэшировано {len(self.reg_df)} регуляций")

    def _fasttext_embedding(self, text: str) -> np.ndarray:
        if self.ft_model is None:
            return np.zeros(0)
        words = text.split()
        if not words:
            return np.zeros(self.ft_model.get_dimension())
        vectors = [self.ft_model.get_word_vector(w) for w in words]
        vec = np.mean(vectors, axis=0)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _sbert_embedding(self, text: str) -> np.ndarray:
        if self.sbert is None:
            return np.zeros(0)
        return self.sbert.encode(text, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    def _extract_codes(self, text: str) -> List[str]:
        return self.code_pattern.findall(text)

    def _code_prefix_len(self, decl_text: str, reg_code: Optional[str]) -> int:
        if not reg_code:
            return 0
        matches = self._extract_codes(decl_text)
        best = 0
        for match in matches:
            common = 0
            for a, b in zip(reg_code, match):
                if a == b:
                    common += 1
                else:
                    break
            best = max(best, common)
        return best

    def _category_match(self, decl_text: str, reg_code: Optional[str]) -> float:
        if not reg_code:
            return 0.0
        category = self.tnved_tree.get_category(reg_code)
        if not category:
            return 0.0
        decl_words = decl_text.split()
        keywords = self.tnved_tree.get_category_keywords(category)
        matched = 0
        for kw in keywords:
            for word in decl_words:
                if kw in word or word in kw:
                    matched += 1
                    break
        return min(1.0, matched / max(1, len(keywords) / 3))

    def _common_char_ngrams(self, text1: str, text2: str, n: int = 3) -> float:
        def ngrams(text):
            return set(text[i:i + n] for i in range(len(text) - n + 1))
        g1, g2 = ngrams(text1), ngrams(text2)
        return len(g1 & g2) / max(1, len(g1 | g2))

    def _extract_weight(self, text: str) -> float:
        match = self.weight_pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                return 0.0
        return 0.0

    def compute_sparse_scores(self, decl_text: str) -> Optional[np.ndarray]:
        if self.bm25 is not None:
            return np.array(self.bm25.get_scores(decl_text.split()))
        if self.reg_tfidf_word is not None:
            decl_vec = self.tfidf_word.transform([decl_text])
            return np.asarray(decl_vec.dot(self.reg_tfidf_word.T).toarray())[0]
        return None

    def compute_sbert_similarities(self, decl_text: str) -> Optional[np.ndarray]:
        if self.reg_sbert is None:
            return None
        decl_emb = self._sbert_embedding(decl_text)
        return np.dot(self.reg_sbert, decl_emb)

    def compute_fasttext_similarities(self, decl_text: str) -> Optional[np.ndarray]:
        if self.reg_fasttext is None:
            return None
        decl_emb = self._fasttext_embedding(decl_text)
        return np.dot(self.reg_fasttext, decl_emb)

    def extract_features_for_pair(
        self,
        decl_row: pd.Series,
        reg_idx: int,
        precomputed: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        features = {}
        decl_text = self._prepare_declaration_text(decl_row)
        reg_text = self.reg_texts[reg_idx]
        reg_code = self.reg_codes[reg_idx]

        if precomputed and 'sbert_sim' in precomputed:
            features['sbert_sim'] = float(precomputed['sbert_sim'][reg_idx])
        else:
            sims = self.compute_sbert_similarities(decl_text)
            features['sbert_sim'] = float(sims[reg_idx]) if sims is not None else 0.0

        if precomputed and 'fasttext_sim' in precomputed:
            features['fasttext_sim'] = float(precomputed['fasttext_sim'][reg_idx])
        else:
            sims = self.compute_fasttext_similarities(decl_text)
            features['fasttext_sim'] = float(sims[reg_idx]) if sims is not None else 0.0

        if precomputed and 'sparse_score' in precomputed:
            sparse = precomputed['sparse_score'][reg_idx]
        else:
            sparse_scores = self.compute_sparse_scores(decl_text)
            sparse = sparse_scores[reg_idx] if sparse_scores is not None else 0.0
        features['sparse_score'] = float(sparse)

        decl_word = self.tfidf_word.transform([decl_text])
        features['tfidf_word_cosine'] = float((decl_word * self.reg_tfidf_word[reg_idx].T).toarray()[0, 0])
        decl_char = self.tfidf_char.transform([decl_text])
        features['tfidf_char_cosine'] = float((decl_char * self.reg_tfidf_char[reg_idx].T).toarray()[0, 0])

        prefix_len = self._code_prefix_len(decl_text, reg_code)
        features['code_prefix_len'] = float(prefix_len)
        for length in [2, 4, 6, 8, 10]:
            features[f'code_match_{length}'] = 1.0 if prefix_len >= length else 0.0

        features['category_match'] = self._category_match(decl_text, reg_code)

        decl_words = set(decl_text.split())
        reg_words = set(reg_text.split())
        common = decl_words & reg_words
        union = decl_words | reg_words
        features['jaccard'] = len(common) / max(1, len(union))
        features['common_word_ratio'] = len(common) / max(1, len(decl_words))
        features['reg_word_ratio'] = len(common) / max(1, len(reg_words))
        features['char_ngram_jaccard'] = self._common_char_ngrams(decl_text, reg_text, n=3)
        features['len_ratio'] = len(decl_text) / (len(reg_text) + 1)

        decl_country = str(decl_row.get('G34', '')) if pd.notna(decl_row.get('G34', '')) else ''
        reg_country = str(self.reg_df.iloc[reg_idx].get('country', '')) if pd.notna(self.reg_df.iloc[reg_idx].get('country', '')) else ''
        features['country_match'] = 1.0 if decl_country and reg_country and decl_country == reg_country else 0.0

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

    def get_top_candidates(self, decl_row: pd.Series, k: int = 50) -> np.ndarray:
        """Быстрый отбор кандидатов через SBERT + BM25 + код."""
        decl_text = self._prepare_declaration_text(decl_row)
        scores = np.zeros(len(self.reg_df))

        sparse = self.compute_sparse_scores(decl_text)
        if sparse is not None:
            sparse_norm = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-8)
            scores += 0.4 * sparse_norm

        sbert = self.compute_sbert_similarities(decl_text)
        if sbert is not None:
            scores += 0.4 * sbert

        code_matches = self._extract_codes(decl_text)
        if code_matches:
            for idx, reg_code in enumerate(self.reg_codes):
                if not reg_code:
                    continue
                for match in code_matches:
                    common = sum(1 for a, b in zip(reg_code, match) if a == b)
                    if common >= 6:
                        scores[idx] += 0.5
                    elif common >= 4:
                        scores[idx] += 0.3
                    elif common >= 2:
                        scores[idx] += 0.1

        for idx, reg_code in enumerate(self.reg_codes):
            cat_match = self._category_match(decl_text, reg_code)
            if cat_match > 0.5:
                scores[idx] += 0.2

        top_k = min(k, len(scores))
        return np.argsort(scores)[-top_k:][::-1]
