import json
import pandas as pd

# Загружаем декларации
declarations = []
with open('data/declarations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            declarations.append(json.loads(line))
decl_df = pd.DataFrame(declarations)

# Загружаем регуляции
regulations = []
with open('data/regulations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            regulations.append(json.loads(line))
reg_df = pd.DataFrame(regulations)

predictions = pd.read_csv('out/predictions.csv')

merged = predictions.merge(decl_df[['declaration_id', 'G31_1', 'desc_extention']], on='declaration_id')
merged = merged.merge(reg_df[['regulation_id', 'code', 'description']], on='regulation_id')

merged = merged.sort_values(['declaration_id', 'rank'])

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

print("="*100)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРЕДСКАЗАНИЙ")
print("="*100)

for decl_id in merged['declaration_id'].unique():
    print(f"\n{'='*100}")
    print(f"ДЕКЛАРАЦИЯ: {decl_id}")
    
    decl = decl_df[decl_df['declaration_id'] == decl_id].iloc[0]
    print(f"\nТовар: {decl['G31_1']}")
    print(f"Детали: {decl['desc_extention']}")
    
    if pd.notna(decl.get('G34')):
        print(f"Страна: {decl['G34']}")
    if pd.notna(decl.get('G32')):
        print(f"Вес/количество: {decl['G32']}")
    
    print(f"\nТОП-10 РЕГУЛЯЦИЙ:")
    print("-"*100)
    
    for _, row in merged[merged['declaration_id'] == decl_id].iterrows():
        print(f"Ранг {row['rank']}: {row['regulation_id']} | Код: {row['code']} | Счёт: {row['score']:.2f}")
        print(f"   Описание: {row['description'][:150]}...")
        print()