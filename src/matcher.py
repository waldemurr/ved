import json
import logging
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class MLMatcher:
    """ML-матчер на основе градиентного бустинга или Ridge регрессии."""

    def __init__(self, feature_extractor, ranker: str = 'auto'):
        """
        ranker: 'catboost', 'lightgbm', 'xgboost', 'ridge' или 'auto'
        auto: catboost -> lightgbm -> ridge
        """
        self.feature_extractor = feature_extractor
        self.ranker_name = ranker
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.available_rankers = self._detect_available_rankers()

    def _detect_available_rankers(self) -> List[str]:
        rankers = []
        try:
            import catboost
            rankers.append('catboost')
        except Exception:
            pass
        try:
            import lightgbm
            rankers.append('lightgbm')
        except Exception:
            pass
        try:
            import xgboost
            rankers.append('xgboost')
        except Exception:
            pass
        rankers.append('ridge')
        return rankers

    def _choose_ranker(self) -> str:
        if self.ranker_name != 'auto':
            if self.ranker_name in self.available_rankers:
                return self.ranker_name
            logger.warning(f"Ранжировщик {self.ranker_name} не доступен, выбираем fallback")
        for r in ['catboost', 'lightgbm', 'xgboost', 'ridge']:
            if r in self.available_rankers:
                return r
        return 'ridge'

    def _build_ranker(self):
        ranker = self._choose_ranker()
        logger.info(f"Используется ранжировщик: {ranker}")
        if ranker == 'catboost':
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                loss_function='RMSE',
                verbose=False,
                random_state=42,
                thread_count=-1,
            )
        elif ranker == 'lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                objective='regression',
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        elif ranker == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
            )
        else:
            return Ridge(alpha=1.0)

    def _calculate_label(self, decl_row: pd.Series, reg_row: pd.Series, decl_text: str) -> float:
        """Слабая разметка на основе кода ТН ВЭД и категории."""
        reg_code = str(reg_row.get('code', '')) if pd.notna(reg_row.get('code', '')) else ''
        label = 0.0

        code_matches = self.feature_extractor._extract_codes(decl_text)
        if reg_code and code_matches:
            for match in code_matches:
                common = 0
                for a, b in zip(reg_code, match):
                    if a == b:
                        common += 1
                    else:
                        break
                if common >= 10:
                    label = max(label, 1.0)
                elif common >= 8:
                    label = max(label, 0.8)
                elif common >= 6:
                    label = max(label, 0.6)
                elif common >= 4:
                    label = max(label, 0.4)
                elif common >= 2:
                    label = max(label, 0.2)

        label += self.feature_extractor._category_match(decl_text, reg_code) * 0.5

        # Текстовое сходство как часть слабой разметки
        reg_text = self.feature_extractor._prepare_regulation_text(reg_row)
        decl_words = set(decl_text.split())
        reg_words = set(reg_text.split())
        if decl_words and reg_words:
            common = decl_words & reg_words
            jaccard = len(common) / len(decl_words | reg_words)
            label = max(label, jaccard * 0.5)

        # Бонус за совпадение страны, если код тоже близок
        decl_country = str(decl_row.get('G34', '')) if pd.notna(decl_row.get('G34', '')) else ''
        reg_country = str(reg_row.get('country', '')) if pd.notna(reg_row.get('country', '')) else ''
        if decl_country and reg_country and decl_country == reg_country and label > 0:
            label += 0.05

        return min(label, 1.0)

    def generate_training_data(
        self,
        decl_df: pd.DataFrame,
        reg_df: pd.DataFrame,
        n_samples: Optional[int] = None,
        max_pairs_per_decl: int = 100,
        max_time_seconds: int = 300,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Генерация обучающих данных с ограничением по времени."""
        logger.info("Генерация обучающих данных...")

        X = []
        y = []
        start_time = time.time()

        # Ограничиваем число деклараций для скорости
        decl_sample = decl_df if len(decl_df) <= 60 else decl_df.sample(n=60, random_state=42)

        for _, decl in decl_sample.iterrows():
            if time.time() - start_time > max_time_seconds:
                logger.info(f"Достигнут лимит времени {max_time_seconds} секунд")
                break

            decl_text = self.feature_extractor._prepare_declaration_text(decl)
            code_matches = self.feature_extractor._extract_codes(decl_text)

            # Если есть кодовые подсказки, берём релевантные и случайные негативы
            if code_matches:
                positive_indices = set()
                negative_indices = set(range(len(reg_df)))
                for idx, reg in reg_df.iterrows():
                    reg_code = str(reg.get('code', '')) if pd.notna(reg.get('code', '')) else ''
                    for match in code_matches:
                        if reg_code and (reg_code.startswith(match) or match.startswith(reg_code)):
                            positive_indices.add(idx)
                            negative_indices.discard(idx)
                            break

                sampled = list(positive_indices)[:30]
                sampled += list(negative_indices)[:max(30, max_pairs_per_decl - len(sampled))]
            else:
                sampled = list(range(len(reg_df)))[:max_pairs_per_decl]

            for reg_idx in sampled:
                reg = reg_df.iloc[reg_idx]
                label = self._calculate_label(decl, reg, decl_text)
                features = self.feature_extractor.extract_features_for_pair(decl, reg_idx)
                X.append(features)
                y.append(label)

            if n_samples and len(X) >= n_samples:
                break

        if not X:
            logger.warning("Не сгенерировано ни одного примера!")
            return pd.DataFrame(), np.array([])

        X_df = pd.DataFrame(X)
        y = np.array(y)
        logger.info(f"Сгенерировано {len(X_df)} обучающих примеров")
        logger.info(f"Распределение лейблов: mean={y.mean():.3f}, max={y.max():.3f}")
        return X_df, y

    def train(self, X_df: pd.DataFrame, y: np.ndarray):
        """Обучение модели."""
        if len(X_df) == 0:
            logger.warning("Нет данных для обучения!")
            return None

        self.feature_names = X_df.columns.tolist()
        logger.info(f"Признаки: {self.feature_names}")

        # Заполняем пропуски
        X_filled = X_df.fillna(0.0)

        self.model = self._build_ranker()

        if isinstance(self.model, Ridge):
            X_scaled = self.scaler.fit_transform(X_filled)
            self.model.fit(X_scaled, y)
        else:
            self.scaler.fit(X_filled)
            self.model.fit(X_filled, y)

        if isinstance(self.model, Ridge):
            X_scaled = self.scaler.transform(X_filled)
            y_pred = self.model.predict(X_scaled)
        else:
            y_pred = self.model.predict(X_filled)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        logger.info(f"R² на обучении: {r2:.4f}")
        logger.info(f"MSE на обучении: {mse:.4f}")

        # Важность признаков
        try:
            if isinstance(self.model, Ridge):
                importances = self.model.coef_
            else:
                importances = self.model.feature_importances_

            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances,
            }).sort_values('importance', ascending=False)

            logger.info("Топ-10 важнейших признаков:")
            for _, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            logger.warning(f"Не удалось получить важность признаков: {e}")

        return self.model

    def predict(self, decl_row: pd.Series, reg_df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        """Предсказание top-k регуляций для декларации."""
        if self.model is None or self.feature_names is None:
            return self.predict_fallback(decl_row, reg_df, top_k)

        X = []
        reg_ids = []

        for reg_idx in range(len(reg_df)):
            reg = reg_df.iloc[reg_idx]
            features = self.feature_extractor.extract_features_for_pair(decl_row, reg_idx)
            X.append([features.get(f, 0.0) for f in self.feature_names])
            reg_ids.append(reg['regulation_id'])

        X = pd.DataFrame(X, columns=self.feature_names).fillna(0.0)

        if isinstance(self.model, Ridge):
            X_scaled = self.scaler.transform(X)
            scores = self.model.predict(X_scaled)
        else:
            scores = self.model.predict(X)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'regulation_id': reg_ids[idx],
                'score': float(scores[idx]),
            })

        # Нормализация в 0-100
        if results:
            scores_arr = np.array([r['score'] for r in results])
            min_score = scores_arr.min()
            max_score = scores_arr.max()
            if max_score > min_score:
                for r in results:
                    r['score'] = 10.0 + 89.9 * (r['score'] - min_score) / (max_score - min_score)
            else:
                for r in results:
                    r['score'] = 50.0
            for i, r in enumerate(results):
                # Небольшой tie-breaker по рангу, чтобы избежать дублирующихся скоров
                r['score'] = round(r['score'] + (len(results) - i) * 0.01, 4)
                r['score'] = max(0.0, min(100.0, r['score']))

        return results

    def predict_fallback(self, decl_row: pd.Series, reg_df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        """Fallback на основе TF-IDF и кодовых признаков."""
        logger.info("Используется fallback (TF-IDF + кодовые признаки)")

        scores = []
        reg_ids = []
        decl_text = self.feature_extractor._prepare_declaration_text(decl_row)

        for reg_idx in range(len(reg_df)):
            reg = reg_df.iloc[reg_idx]
            features = self.feature_extractor.extract_features_for_pair(decl_row, reg_idx)
            score = (
                features.get('tfidf_cosine', 0.0) * 0.5
                + features.get('char_tfidf_cosine', 0.0) * 0.3
                + features.get('code_prefix_len', 0.0) / 10.0 * 0.2
            )
            scores.append(score)
            reg_ids.append(reg['regulation_id'])

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({'regulation_id': reg_ids[idx], 'score': scores[idx]})

        if results:
            scores_arr = np.array([r['score'] for r in results])
            min_score = scores_arr.min()
            max_score = scores_arr.max()
            if max_score > min_score:
                for r in results:
                    r['score'] = 10.0 + 89.9 * (r['score'] - min_score) / (max_score - min_score)
            else:
                for r in results:
                    r['score'] = 50.0
            for i, r in enumerate(results):
                r['score'] = round(r['score'] + (len(results) - i) * 0.01, 4)
                r['score'] = max(0.0, min(100.0, r['score']))

        return results
