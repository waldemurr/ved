#!/usr/bin/env python3
import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_loader import DataLoader
from src.feature_extractor import HybridFeatureExtractor
from src.matcher import MLMatcher
from src.validator import Validator

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Матчинг деклараций с регуляциями ТН ВЭД')
    parser.add_argument('--data', default='./data', help='Путь к директории с данными')
    parser.add_argument('--out', default='./out', help='Путь к выходной директории')
    parser.add_argument('--model_name', default=None, help='Название SBERT модели (опционально)')
    parser.add_argument('--fasttext_path', default=None, help='Путь к FastText модели (опционально)')
    parser.add_argument('--ranker', default='auto', help='Ранжировщик: auto, catboost, lightgbm, xgboost, ridge')
    parser.add_argument('--gpu', action='store_true', help='Использовать GPU если доступен')
    parser.add_argument('--ml_model_path', default=None, help='Путь к сохраненной ML модели (опционально)')
    parser.add_argument('--no_train', action='store_true', help='Не обучать модель, использовать fallback')
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Запуск ML матчинга деклараций с регуляциями ТН ВЭД")
    logger.info("=" * 60)

    # 1. Загрузка данных
    logger.info("Шаг 1: Загрузка данных")
    loader = DataLoader(data_path)
    decl_df = loader.load_declarations()
    reg_df = loader.load_regulations()
    tnved_path = data_path / 'tnved_knowledge.txt'

    # 2. Инициализация Feature Extractor
    logger.info("Шаг 2: Инициализация Feature Extractor")
    feature_extractor = HybridFeatureExtractor(
        model_name=args.model_name,
        fasttext_path=args.fasttext_path,
        use_gpu=args.gpu,
        tnved_tree_path=str(tnved_path) if tnved_path.exists() else None,
    )
    feature_extractor.fit_regulations(reg_df)

    # 3. Инициализация ML матчера
    logger.info("Шаг 3: Инициализация ML матчера")
    matcher = MLMatcher(
        feature_extractor=feature_extractor,
        ranker=args.ranker,
    )

    # 4. Обучение или загрузка модели
    if args.ml_model_path and Path(args.ml_model_path).exists() and not args.no_train:
        logger.info(f"Шаг 4a: Загрузка модели из {args.ml_model_path}")
        with open(args.ml_model_path, 'rb') as f:
            saved = pickle.load(f)
            matcher.model = saved['model']
            matcher.scaler = saved['scaler']
            matcher.feature_names = saved['feature_names']
            matcher.ranker_name = saved.get('ranker', args.ranker)
        logger.info("Модель загружена")
    elif not args.no_train:
        logger.info("Шаг 4b: Обучение ML модели")
        X_train, y_train = matcher.generate_training_data(decl_df, reg_df, max_pairs_per_decl=360)

        if len(X_train) > 0:
            matcher.train(X_train, y_train)

            # Сохраняем модель
            model_path = out_path / 'ml_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': matcher.model,
                    'scaler': matcher.scaler,
                    'feature_names': matcher.feature_names,
                    'ranker': matcher.ranker_name,
                }, f)
            logger.info(f"Модель сохранена в {model_path}")
        else:
            logger.warning("Не удалось обучить модель (нет данных), используем fallback")
    else:
        logger.info("Шаг 4c: Пропускаем обучение (используем fallback)")

    # 5. Предсказание для деклараций
    logger.info("Шаг 5: Предсказание для деклараций")
    all_predictions = []

    for _, decl in tqdm(decl_df.iterrows(), total=len(decl_df), desc="Обработка деклараций"):
        decl_id = decl['declaration_id']

        if matcher.model is not None and matcher.feature_names is not None:
            predictions = matcher.predict(decl, reg_df, top_k=10)
        else:
            predictions = matcher.predict_fallback(decl, reg_df, top_k=10)

        for rank, pred in enumerate(predictions, 1):
            all_predictions.append({
                'declaration_id': decl_id,
                'rank': rank,
                'regulation_id': pred['regulation_id'],
                'score': pred['score'],
            })

    predictions_df = pd.DataFrame(all_predictions)

    # 6. Валидация и сохранение
    logger.info("Шаг 6: Валидация и сохранение")

    try:
        Validator.validate_predictions(predictions_df)
    except Exception as e:
        logger.error(f"Ошибка валидации: {e}")
        sys.exit(1)

    output_path = out_path / 'predictions.csv'
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Результаты сохранены в {output_path}")

    # Анализ ошибок
    errors = Validator.analyze_errors(predictions_df, decl_df, reg_df)
    if errors['total_errors'] > 0:
        logger.warning(f"Найдено {errors['total_errors']} потенциальных проблем")
        for error in errors['errors']:
            logger.warning(f"  {error['declaration_id']}: {error['error']} - {error['details']}")

    # Финальная статистика
    logger.info("=" * 60)
    logger.info("Финальная статистика:")
    logger.info(f"  - Деклараций обработано: {len(decl_df)}")
    logger.info(f"  - Регуляций в кэше: {len(reg_df)}")
    logger.info(f"  - Всего предсказаний: {len(predictions_df)}")
    logger.info(f"  - Уникальных регуляций использовано: {len(predictions_df['regulation_id'].unique())}")

    if matcher.model is not None:
        logger.info(f"  - Использована ML модель: {type(matcher.model).__name__}")
    else:
        logger.info("  - Использован fallback")

    logger.info("=" * 60)
    logger.info("Готово!")


if __name__ == '__main__':
    main()
