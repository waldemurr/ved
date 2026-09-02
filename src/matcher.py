import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MLMatcher:
    """SOTA матчер: GBDT reranker + опциональный cross-encoder reranking."""

    def __init__(self, feature_extractor, ranker: str = 'auto', use_cross_encoder: bool = False):
        self.feature_extractor = feature_extractor
        self.ranker_name = ranker
        self.use_cross_encoder = use_cross_encoder
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _detect_rankers(self) -> List[str]:
        rankers = []
        for name in ['catboost', 'lightgbm', 'xgboost']:
            try:
                __import__(name)
                rankers.append(name)
            except Exception:
                pass
        rankers.append('ridge')
        return rankers

    def _choose_ranker(self) -> str:
        available = self._detect_rankers()
        if self.ranker_name != 'auto' and self.ranker_name in available:
            return self.ranker_name
        for r in ['catboost', 'lightgbm', 'xgboost', 'ridge']:
            if r in available:
                return r
        return 'ridge'

    def _build_ranker(self):
        ranker = self._choose_ranker()
        logger.info(f"Используется ранжировщик: {ranker}")
        if ranker == 'catboost':
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=800,
                depth=8,
                learning_rate=0.05,
                loss_function='RMSE',
                verbose=False,
                random_seed=42,
                thread_count=-1,
            )
        elif ranker == 'lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                objective='regression',
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        elif ranker == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
            )
        else:
            return Ridge(alpha=1.0)

    def _calculate_label(self, decl_row: pd.Series, reg_idx: int) -> float:
        """Слабая разметка: код, категория, SBERT/BM25 сходство."""
        decl_text = self.feature_extractor._prepare_declaration_text(decl_row)
        reg_code = self.feature_extractor.reg_codes[reg_idx]
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

        label += self.feature_extractor._category_match(decl_text, reg_code) * 0.4

        # Добавляем слабый сигнал от семантического/лексического сходства
        sbert_sims = self.feature_extractor.compute_sbert_similarities(decl_text)
        if sbert_sims is not None:
            label = max(label, float(sbert_sims[reg_idx]) * 0.5)
        else:
            sparse = self.feature_extractor.compute_sparse_scores(decl_text)
            if sparse is not None:
                sparse_norm = (sparse[reg_idx] - sparse.min()) / (sparse.max() - sparse.min() + 1e-8)
                label = max(label, sparse_norm * 0.3)

        return min(label, 1.0)

    def generate_training_data(
        self,
        decl_df: pd.DataFrame,
        reg_df: pd.DataFrame,
        max_pairs_per_decl: int = 360,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        logger.info("Генерация обучающих данных...")
        X = []
        y = []

        for _, decl in decl_df.iterrows():
            decl_text = self.feature_extractor._prepare_declaration_text(decl)

            # Предвычисляем общие сигналы для декларации
            precomputed = {}
            sbert_sims = self.feature_extractor.compute_sbert_similarities(decl_text)
            if sbert_sims is not None:
                precomputed['sbert_sim'] = sbert_sims
            fasttext_sims = self.feature_extractor.compute_fasttext_similarities(decl_text)
            if fasttext_sims is not None:
                precomputed['fasttext_sim'] = fasttext_sims
            sparse_scores = self.feature_extractor.compute_sparse_scores(decl_text)
            if sparse_scores is not None:
                precomputed['sparse_score'] = sparse_scores

            # Отбираем кандидатов: топ по SBERT/BM25 + случайные
            candidates = self.feature_extractor.get_top_candidates(decl, k=80)
            if len(candidates) < max_pairs_per_decl:
                extra = np.setdiff1d(np.arange(len(reg_df)), candidates)
                np.random.seed(42)
                np.random.shuffle(extra)
                candidates = np.concatenate([candidates, extra[:max_pairs_per_decl - len(candidates)]])

            for reg_idx in candidates[:max_pairs_per_decl]:
                features = self.feature_extractor.extract_features_for_pair(decl, reg_idx, precomputed)
                label = self._calculate_label(decl, reg_idx)
                X.append(features)
                y.append(label)

        X_df = pd.DataFrame(X)
        y = np.array(y)
        logger.info(f"Сгенерировано {len(X_df)} примеров, mean_label={y.mean():.3f}, max={y.max():.3f}")
        return X_df, y

    def train(self, X_df: pd.DataFrame, y: np.ndarray):
        if len(X_df) == 0:
            logger.warning("Нет данных для обучения!")
            return None

        self.feature_names = X_df.columns.tolist()
        X_filled = X_df.fillna(0.0)

        self.model = self._build_ranker()
        if isinstance(self.model, Ridge):
            X_scaled = self.scaler.fit_transform(X_filled)
            self.model.fit(X_scaled, y)
            y_pred = self.model.predict(self.scaler.transform(X_filled))
        else:
            self.scaler.fit(X_filled)
            self.model.fit(X_filled, y)
            y_pred = self.model.predict(X_filled)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        logger.info(f"R² на обучении: {r2:.4f}, MSE: {mse:.4f}")

        try:
            importances = self.model.coef_ if isinstance(self.model, Ridge) else self.model.feature_importances_
            importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
            importance_df = importance_df.sort_values('importance', ascending=False)
            logger.info("Топ-10 признаков:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            logger.warning(f"Не удалось получить важность признаков: {e}")

        return self.model

    def predict(self, decl_row: pd.Series, reg_df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        if self.model is None or self.feature_names is None:
            return self.predict_fallback(decl_row, reg_df, top_k)

        decl_text = self.feature_extractor._prepare_declaration_text(decl_row)

        # Предвычисляем общие сигналы
        precomputed = {}
        sbert_sims = self.feature_extractor.compute_sbert_similarities(decl_text)
        if sbert_sims is not None:
            precomputed['sbert_sim'] = sbert_sims
        fasttext_sims = self.feature_extractor.compute_fasttext_similarities(decl_text)
        if fasttext_sims is not None:
            precomputed['fasttext_sim'] = fasttext_sims
        sparse_scores = self.feature_extractor.compute_sparse_scores(decl_text)
        if sparse_scores is not None:
            precomputed['sparse_score'] = sparse_scores

        # Отбираем кандидатов
        candidates = self.feature_extractor.get_top_candidates(decl_row, k=80)

        # GBDT скоры для кандидатов
        X = []
        candidate_ids = []
        for reg_idx in candidates:
            features = self.feature_extractor.extract_features_for_pair(decl_row, reg_idx, precomputed)
            X.append([features.get(f, 0.0) for f in self.feature_names])
            candidate_ids.append(reg_idx)

        X_df = pd.DataFrame(X, columns=self.feature_names).fillna(0.0)
        if isinstance(self.model, Ridge):
            scores = self.model.predict(self.scaler.transform(X_df))
        else:
            scores = self.model.predict(X_df)

        # Опционально переранжируем топ кросс-энкодером
        if self.use_cross_encoder and self.feature_extractor.cross_encoder is not None:
            top_gbdt_idx = np.argsort(scores)[-min(30, len(scores)):][::-1]
            pairs = []
            pair_reg_ids = []
            for local_idx in top_gbdt_idx:
                reg_idx = candidate_ids[local_idx]
                reg_text = self.feature_extractor.reg_texts[reg_idx]
                pairs.append((decl_text, reg_text))
                pair_reg_ids.append(reg_idx)
            ce_scores = self.feature_extractor.cross_encoder.predict(pairs)
            # Комбинируем GBDT и cross-encoder
            final_scores = []
            for local_idx in top_gbdt_idx:
                final_scores.append(0.6 * ce_scores[np.where(top_gbdt_idx == local_idx)[0][0]] + 0.4 * scores[local_idx])
            candidate_ids = [candidate_ids[i] for i in top_gbdt_idx]
            scores = np.array(final_scores)

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'regulation_id': self.feature_extractor.reg_ids[candidate_ids[idx]],
                'score': float(scores[idx]),
            })

        return self._normalize_scores(results)

    def predict_fallback(self, decl_row: pd.Series, reg_df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        logger.info("Используется fallback (SBERT/BM25/TF-IDF)")
        decl_text = self.feature_extractor._prepare_declaration_text(decl_row)

        precomputed = {}
        sbert_sims = self.feature_extractor.compute_sbert_similarities(decl_text)
        if sbert_sims is not None:
            precomputed['sbert_sim'] = sbert_sims
        fasttext_sims = self.feature_extractor.compute_fasttext_similarities(decl_text)
        if fasttext_sims is not None:
            precomputed['fasttext_sim'] = fasttext_sims
        sparse_scores = self.feature_extractor.compute_sparse_scores(decl_text)
        if sparse_scores is not None:
            precomputed['sparse_score'] = sparse_scores

        candidates = self.feature_extractor.get_top_candidates(decl_row, k=50)
        scores = []
        reg_ids = []
        for reg_idx in candidates:
            f = self.feature_extractor.extract_features_for_pair(decl_row, reg_idx, precomputed)
            score = (
                f.get('sbert_sim', 0.0) * 0.5
                + f.get('sparse_score', 0.0) * 0.3
                + f.get('category_match', 0.0) * 0.2
            )
            scores.append(score)
            reg_ids.append(self.feature_extractor.reg_ids[reg_idx])

        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [{'regulation_id': reg_ids[i], 'score': scores[i]} for i in top_indices]
        return self._normalize_scores(results)

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores_arr = np.array([r['score'] for r in results])
        min_score = scores_arr.min()
        max_score = scores_arr.max()
        if max_score > min_score:
            for i, r in enumerate(results):
                r['score'] = round(10.0 + 89.9 * (r['score'] - min_score) / (max_score - min_score) + (len(results) - i) * 0.01, 4)
                r['score'] = max(0.0, min(100.0, r['score']))
        else:
            for i, r in enumerate(results):
                r['score'] = round(50.0 + (len(results) - i) * 0.01, 4)
        return results
