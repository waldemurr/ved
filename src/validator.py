import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Validator:
    """Валидация предсказаний"""
    
    @staticmethod
    def validate_predictions(predictions: pd.DataFrame) -> bool:
        """Валидация формата предсказаний"""
        logger.info("Валидация предсказаний...")
        
        required_cols = ['declaration_id', 'rank', 'regulation_id', 'score']
        for col in required_cols:
            if col not in predictions.columns:
                raise ValueError(f"Отсутствует колонка {col}")
        
        for decl_id in predictions['declaration_id'].unique():
            decl_rows = predictions[predictions['declaration_id'] == decl_id]
            
            if len(decl_rows) != 10:
                raise ValueError(f"Декларация {decl_id} имеет {len(decl_rows)} строк, ожидается 10")
            
            if set(decl_rows['rank']) != set(range(1, 11)):
                raise ValueError(f"Декларация {decl_id} имеет некорректные ранги")
            
            if len(decl_rows['regulation_id'].unique()) != 10:
                raise ValueError(f"Декларация {decl_id} имеет дублирующиеся регуляции")
            
            if (decl_rows['score'] < 0).any() or (decl_rows['score'] > 100).any():
                raise ValueError(f"Декларация {decl_id} имеет скоры вне диапазона 0-100")
        
        logger.info("Валидация пройдена успешно")
        return True
    
    @staticmethod
    def analyze_errors(predictions: pd.DataFrame, 
                       decl_df: pd.DataFrame,
                       reg_df: pd.DataFrame) -> dict:
        """Анализ ошибок и сложных случаев"""
        errors = []
        
        for decl_id in predictions['declaration_id'].unique():
            decl_rows = predictions[predictions['declaration_id'] == decl_id]
            decl = decl_df[decl_df['declaration_id'] == decl_id].iloc[0]
            
            if len(decl_rows) < 10:
                errors.append({
                    'declaration_id': decl_id,
                    'error': 'Неполное количество регуляций',
                    'details': f"Найдено {len(decl_rows)}, ожидается 10"
                })
            
            if len(decl_rows['score'].unique()) < 10:
                errors.append({
                    'declaration_id': decl_id,
                    'error': 'Дублирующиеся скоры',
                    'details': f"Уникальных скоров: {len(decl_rows['score'].unique())}"
                })
            if (decl_rows['score'] < 10).any():
                errors.append({
                    'declaration_id': decl_id,
                    'error': 'Очень низкие скоры',
                    'details': f"Минимальный скор: {decl_rows['score'].min()}"
                })
        
        return {'total_errors': len(errors), 'errors': errors}