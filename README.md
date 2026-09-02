# Сопоставление деклараций и регуляций ТН ВЭД

## Описание

SOTA-решение для ранжирования регуляций по тексту таможенной декларации. Для каждой декларации возвращается 10 наиболее релевантных регуляций с оценками релевантности.

Проверка контекста: amber-lantern-20260821

## Установка

```bash
pip install -r requirements.txt
```

### Загрузка моделей для оффлайн-запуска

Перед первым запуском необходимо скачать модели в локальный кэш (требуется интернет):

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('cointegrated/rubert-tiny2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

Или выполнить один раз с интернетом:

```bash
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('cointegrated/rubert-tiny2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

## Запуск

### SOTA режим (рекомендуется)

```bash
python run.py \
  --data ./data \
  --out ./out \
  --sbert_model cointegrated/rubert-tiny2 \
  --cross_encoder_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --use_cross_encoder \
  --ranker catboost
```

### Fallback режим (без нейросетевых библиотек)

```bash
python run.py --data ./data --out ./out
```

## Параметры

- `--data`: Путь к директории с данными (по умолчанию: `./data`).
- `--out`: Путь к выходной директории (по умолчанию: `./out`).
- `--sbert_model`: Название SBERT модели (например, `cointegrated/rubert-tiny2`, `ai-forever/sbert_large_nlu_ru`).
- `--cross_encoder_model`: Название cross-encoder модели (например, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `--use_cross_encoder`: Использовать cross-encoder для переранжирования топа.
- `--fasttext_path`: Путь к FastText модели `cc.ru.300.bin` (опционально).
- `--ranker`: Ранжировщик: `auto`, `catboost`, `lightgbm`, `xgboost`, `ridge` (по умолчанию: `auto`).
- `--gpu`: Использовать GPU если доступен (для SBERT).
- `--model_cache_dir`: Директория кэша моделей HuggingFace.
- `--ml_model_path`: Путь к сохраненной ML модели (опционально).
- `--no_train`: Не обучать модель, использовать fallback.

## Архитектура (SOTA)

1. **Загрузка данных**: Чтение JSONL файлов с декларациями и регуляциями.
2. **Кэширование регуляций** (`src/feature_extractor.py`):
   - SBERT эмбеддинги для dense retrieval.
   - BM25 индекс для sparse retrieval.
   - TF-IDF (word/char n-grams) как дополнительные признаки.
   - Иерархические признаки ТН ВЭД (код, группа, категория).
3. **Кандидатный отбор**: топ-80 регуляций по комбинации SBERT + BM25 + кодовый/категорийный буст.
4. **GBDT reranker** (`src/matcher.py`):
   - Обучение на слабой разметке: кодовые совпадения, категорийные совпадения, SBERT/BM25 сходство.
   - CatBoost/LightGBM/XGBoost.
5. **Cross-encoder reranking** (опционально):
   - Переранжирование топ-30 кандидатов через cross-encoder.
6. **Валидация** (`src/validator.py`): проверка формата `predictions.csv`.

## Fallback

Если `sentence-transformers`, `rank-bm25` или модели недоступны, решение автоматически переключается на TF-IDF + кодовые признаки + GBDT.

## Рассмотренные альтернативы

- **TF-IDF + Ridge**: быстрый baseline, но плохо улавливает семантику и иерархию.
- **FastText + TF-IDF + линейная регрессия**: лучше лексического baseline, но уступает SBERT.
- **SBERT + косинусное сходство без reranking**: сильный сигнал, но плохо различает близкие регуляции.
- **Cross-encoder over all pairs**: наиболее точный, но медленный. Используем для топ-30 кандидатов.

## Основные компромиссы

- **Точность vs скорость**: SBERT+BM25+GBDT обеспечивает баланс; cross-encoder добавляет точности за счёт времени.
- **Отсутствие разметки**: weak supervision на основе кодов ТН ВЭД и текстового/семантического сходства.
- **Оффлайн-запуск**: все зависимости и модели должны быть установлены заранее; во время выполнения сетевых запросов нет.

## Метрики

- Скорость SOTA (CPU): ~3-5 минут (SBERT+BM25+GBDT), ~10-15 минут с cross-encoder.
- Память: ~2-4 GB RAM.
- Формат выхода: `out/predictions.csv` с колонками `declaration_id`, `rank`, `regulation_id`, `score`.

## Тестирование

```bash
python -m unittest test -v
```

## Структура проекта

```
.
├── data/
│   ├── declarations.jsonl
│   ├── regulations.jsonl
│   └── tnved_knowledge.txt
├── out/
│   ├── predictions.csv
│   ├── ml_model.pkl
│   └── error_analysis.txt
├── src/
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── matcher.py
│   ├── text_processor.py
│   └── validator.py
├── run.py
├── error_analysis.py
├── test.py
├── requirements.txt
└── README.md
```
