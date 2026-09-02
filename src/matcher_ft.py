#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import fasttext.util
import re
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastTextFeatureExtractor:
    """Извлечение признаков на основе FastText + TF-IDF"""
    
    def __init__(self, model_path=None):
        if model_path and Path(model_path).exists():
            logger.info(f"Загрузка FastText модели из {model_path}")
            self.model = fasttext.load_model(model_path)
        else:
            # Скачиваем легкую модель
            logger.info("Скачивание FastText модели...")
            fasttext.util.download_model('ru', if_exists='ignore')
            self.model = fasttext.load_model('cc.ru.300.bin')
        
        # TF-IDF для дополнительных признаков
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            lowercase=True,
            analyzer='char_wb'
        )
        
        self.reg_ft_embeddings = None
        self.reg_tfidf = None
        self.reg_df = None
        self.reg_ids = None
        
        # Регексы
        self.code_pattern = re.compile(r'(?:код|тн вэд|товарная позиция)\s*(\d{4,10})', re.IGNORECASE)
        self.weight_pattern = re.compile(r'(\d+[.,]?\d*)\s*(?:кг|г|т|тонн|карат)', re.IGNORECASE)
        self.year_pattern = re.compile(r'(19|20)\d{2}')
        
    def _get_ft_embedding(self, text):
        """Получение эмбеддинга через FastText"""
        # FastText ожидает текст, возвращает средний вектор слов
        words = text.lower().split()
        if not words:
            return np.zeros(self.model.get_dimension())
        
        vectors = []
        for word in words:
            try:
                vectors.append(self.model.get_word_vector(word))
            except:
                continue
        
        if not vectors:
            return np.zeros(self.model.get_dimension())
        
        return np.mean(vectors, axis=0)
    
    def _prepare_regulation_text(self, row):
        parts = []
        if pd.notna(row.get('description')):
            parts.append(str(row['description']))
        if pd.notna(row.get('explanation')):
            parts.append(str(row['explanation']))
        if pd.notna(row.get('notes')):
            notes = str(row['notes'])[:300]
            parts.append(notes)
        return ' '.join(parts)
    
    def _prepare_declaration_text(self, row):
        parts = []
        if pd.notna(row.get('G31_1')):
            parts.append(str(row['G31_1']))
        if pd.notna(row.get('desc_extention')):
            parts.append(str(row['desc_extention']))
        return ' '.join(parts)
    
    def fit_regulations(self, reg_df):
        """Кэшируем эмбеддинги регуляций"""
        logger.info("Кэширование эмбеддингов регуляций...")
        self.reg_df = reg_df.copy()
        self.reg_ids = reg_df['regulation_id'].values
        
        # 1. FastText эмбеддинги
        logger.info("  - Вычисление FastText эмбеддингов...")
        reg_texts = []
        for _, row in reg_df.iterrows():
            text = self._prepare_regulation_text(row)
            reg_texts.append(text)
        
        self.reg_ft_embeddings = []
        for text in tqdm(reg_texts, desc="FastText"):
            emb = self._get_ft_embedding(text)
            self.reg_ft_embeddings.append(emb)
        self.reg_ft_embeddings = np.array(self.reg_ft_embeddings)
        
        # 2. TF-IDF признаки
        logger.info("  - Вычисление TF-IDF...")
        self.reg_tfidf = self.tfidf.fit_transform(reg_texts)
        
        logger.info(f"Закэшировано {len(self.reg_ids)} регуляций")
    
    def extract_features_for_pair(self, decl_row, reg_idx):
        """Извлечение признаков для пары"""
        features = {}
        
        decl_text = self._prepare_declaration_text(decl_row)
        
        # 1. FastText сходство
        decl_ft_emb = self._get_ft_embedding(decl_text)
        reg_ft_emb = self.reg_ft_embeddings[reg_idx]
        features['ft_similarity'] = np.dot(decl_ft_emb, reg_ft_emb) / (
            np.linalg.norm(decl_ft_emb) * np.linalg.norm(reg_ft_emb) + 1e-8
        )
        
        # 2. TF-IDF сходство
        decl_tfidf = self.tfidf.transform([decl_text])
        reg_tfidf = self.reg_tfidf[reg_idx]
        features['tfidf_similarity'] = cosine_similarity(decl_tfidf, reg_tfidf)[0][0]
        
        # 3. Совпадение кодов
        reg_code = self.reg_df.iloc[reg_idx].get('code', '')
        features['code_match'] = self._code_match_score(decl_text, reg_code)
        
        # 4. Текстовые метрики
        reg_text = self._prepare_regulation_text(self.reg_df.iloc[reg_idx])
        features['common_words'] = len(set(decl_text.split()) & set(reg_text.split()))
        features['len_ratio'] = len(decl_text) / (len(reg_text) + 1)
        
        # 5. Числовые признаки
        features['has_weight'] = 1 if self.weight_pattern.search(decl_text) else 0
        features['has_year'] = 1 if self.year_pattern.search(decl_text) else 0
        
        return features
    
    def _code_match_score(self, decl_text, reg_code):
        if not reg_code:
            return 0
        
        matches = self.code_pattern.findall(decl_text)
        max_match = 0
        for match in matches:
            if reg_code.startswith(match):
                max_match = max(max_match, len(match))
        
        if max_match >= 8:
            return 1.0
        elif max_match >= 6:
            return 0.8
        elif max_match >= 4:
            return 0.5
        elif max_match >= 2:
            return 0.3
        return 0


class MLMatcher:
    """Матчер с линейной регрессией"""
    
    def __init__(self, feature_extractor, use_ridge=True):
        self.feature_extractor = feature_extractor
        self.model = Ridge(alpha=1.0) if use_ridge else LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def generate_training_data(self, decl_df, reg_df, n_samples=None):
        """Генерация обучающих данных"""
        logger.info("Генерация обучающих данных...")
        
        X = []
        y = []
        
        for _, decl in tqdm(decl_df.iterrows(), total=len(decl_df)):
            decl_text = self.feature_extractor._prepare_declaration_text(decl)
            code_matches = self.feature_extractor.code_pattern.findall(decl_text)
            
            for reg_idx, _ in enumerate(reg_df):
                label = self._calculate_label(decl_text, reg_idx, code_matches)
                
                if label > 0 or np.random.random() < 0.1:
                    features = self.feature_extractor.extract_features_for_pair(decl, reg_idx)
                    X.append(features)
                    y.append(label)
            
            if n_samples and len(X) > n_samples:
                break
        
        X_df = pd.DataFrame(X)
        y = np.array(y)
        
        logger.info(f"Сгенерировано {len(X_df)} обучающих примеров")
        return X_df, y
    
    def _calculate_label(self, decl_text, reg_idx, code_matches):
        label = 0
        reg_code = self.feature_extractor.reg_df.iloc[reg_idx].get('code', '')
        
        if reg_code and code_matches:
            for match in code_matches:
                if reg_code.startswith(match):
                    if len(match) >= 8:
                        label += 2
                    elif len(match) >= 6:
                        label += 1.5
                    elif len(match) >= 4:
                        label += 1
                    elif len(match) >= 2:
                        label += 0.5
                    break
        
        return min(label, 2)
    
    def train(self, X_df, y):
        if len(X_df) == 0:
            logger.warning("Нет данных для обучения!")
            return None
        
        logger.info("Обучение модели...")
        
        self.feature_names = X_df.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_df)
        
        self.model.fit(X_scaled, y)
        
        # Выводим важность признаков
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coef': coefs
            }).sort_values('coef', ascending=False)
            
            logger.info("Топ-5 важнейших признаков:")
            for _, row in feature_importance.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['coef']:.4f}")
        
        return self.model
    
    def predict(self, decl_row, top_k=10):
        """Предсказание для декларации"""
        if self.model is None or self.feature_names is None:
            return self.predict_fallback(decl_row, top_k)
        
        X = []
        reg_ids = []
        
        for reg_idx, reg_id in enumerate(self.feature_extractor.reg_ids):
            features = self.feature_extractor.extract_features_for_pair(decl_row, reg_idx)
            X.append([features.get(f, 0) for f in self.feature_names])
            reg_ids.append(reg_id)
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.predict(X_scaled)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'regulation_id': reg_ids[idx],
                'score': scores[idx]
            })
        
        # Нормализуем
        if results:
            scores_arr = np.array([r['score'] for r in results])
            min_score, max_score = scores_arr.min(), scores_arr.max()
            
            if max_score > min_score:
                for r in results:
                    r['score'] = 50 + 50 * ((r['score'] - min_score) / (max_score - min_score))
            else:
                for r in results:
                    r['score'] = 50.0
            
            for r in results:
                r['score'] = round(r['score'], 4)
        
        return results
    
    def predict_fallback(self, decl_row, reg_df, top_k=10):
        """Fallback: на основе FastText + TF-IDF"""
        logger.info("Используется fallback")
        
        decl_text = self.feature_extractor._prepare_declaration_text(decl_row)
        decl_ft = self.feature_extractor._get_ft_embedding(decl_text)
        decl_tfidf = self.feature_extractor.tfidf.transform([decl_text])
        
        scores = []
        reg_ids = []
        
        for reg_idx, reg_id in enumerate(self.feature_extractor.reg_ids):
            # FastText сходство
            ft_sim = np.dot(decl_ft, self.feature_extractor.reg_ft_embeddings[reg_idx])
            
            # TF-IDF сходство
            tfidf_sim = cosine_similarity(
                decl_tfidf, 
                self.feature_extractor.reg_tfidf[reg_idx]
            )[0][0]
            
            # Комбинируем
            score = 0.6 * ft_sim + 0.4 * tfidf_sim
            scores.append(score)
            reg_ids.append(reg_id)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'regulation_id': reg_ids[idx],
                'score': scores[idx]
            })
        
        # Нормализуем
        if results:
            scores_arr = np.array([r['score'] for r in results])
            min_score, max_score = scores_arr.min(), scores_arr.max()
            
            if max_score > min_score:
                for r in results:
                    r['score'] = 50 + 50 * ((r['score'] - min_score) / (max_score - min_score))
            else:
                for r in results:
                    r['score'] = 50.0
            
            for r in results:
                r['score'] = round(r['score'], 4)
        
        return results