#!/usr/bin/env python3
"""Анализ 5 сомнительных или ошибочных примеров ранжирования."""
import json
import pandas as pd
from pathlib import Path


def main():
    data_path = Path('./data')
    out_path = Path('./out')

    decls = []
    with open(data_path / 'declarations.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                decls.append(json.loads(line))
    decl_df = pd.DataFrame(decls)

    regs = []
    with open(data_path / 'regulations.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                regs.append(json.loads(line))
    reg_df = pd.DataFrame(regs)

    predictions = pd.read_csv(out_path / 'predictions.csv')

    # Находим примеры для анализа
    examples = []

    # 1. Низкий скор топ-1
    top1 = predictions[predictions['rank'] == 1].copy()
    top1 = top1.sort_values('score')
    examples.append(('Низкий скор топ-1', top1.iloc[0]['declaration_id']))

    # 2. Маленький отрыв между топ-1 и топ-2
    merged = predictions.merge(predictions, on='declaration_id', suffixes=('_1', '_2'))
    merged = merged[(merged['rank_1'] == 1) & (merged['rank_2'] == 2)]
    merged['gap'] = merged['score_1'] - merged['score_2']
    small_gap = merged.sort_values('gap').iloc[0]
    examples.append(('Маленький отрыв топ-1 от топ-2', small_gap['declaration_id']))

    # 3. Высокая неопределенность: много регуляций из разных групп
    pred_with_code = predictions.merge(reg_df[['regulation_id', 'code']], on='regulation_id')
    pred_with_code['group'] = pred_with_code['code'].astype(str).str[:2]
    group_diversity = pred_with_code.groupby('declaration_id')['group'].nunique().reset_index()
    most_diverse = group_diversity.sort_values('group', ascending=False).iloc[0]
    examples.append(('Высокая групповая неопределенность', most_diverse['declaration_id']))

    # 4. Топ-10 содержит коды из разных разделов
    section_diversity = pred_with_code.copy()
    section_diversity['section'] = section_diversity['code'].astype(str).str[:2].astype(int)
    section_diversity = section_diversity.groupby('declaration_id')['section'].apply(lambda x: x.max() - x.min()).reset_index()
    most_sections = section_diversity.sort_values('section', ascending=False).iloc[0]
    examples.append(('Разнообразие разделов в топ-10', most_sections['declaration_id']))

    # 5. Случайный пример
    examples.append(('Случайный пример для проверки', predictions['declaration_id'].iloc[50]))

    # Записываем анализ
    output = []
    output.append("=" * 100)
    output.append("АНАЛИЗ 5 СОМНИТЕЛЬНЫХ ПРИМЕРОВ")
    output.append("=" * 100)

    for reason, decl_id in examples:
        decl = decl_df[decl_df['declaration_id'] == decl_id].iloc[0]
        preds = predictions[predictions['declaration_id'] == decl_id].sort_values('rank')
        preds = preds.merge(reg_df[['regulation_id', 'code', 'description']], on='regulation_id')

        output.append(f"\n{reason}: {decl_id}")
        output.append(f"Товар: {decl.get('G31_1', '')}")
        output.append(f"Детали: {decl.get('desc_extention', '')}")
        output.append(f"Страна: {decl.get('G34', '')}")
        output.append("Топ-10 регуляций:")
        for _, row in preds.iterrows():
            desc = str(row['description'])[:120]
            output.append(f"  Ранг {row['rank']}: {row['regulation_id']} | Код: {row['code']} | Счёт: {row['score']:.2f} | {desc}...")
        output.append("")

    out_text = '\n'.join(output)
    print(out_text)

    with open(out_path / 'error_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(out_text)


if __name__ == '__main__':
    main()
