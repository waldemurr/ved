#!/usr/bin/env python3
import unittest
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.text_processor import TextProcessor
from src.feature_extractor import HybridFeatureExtractor
from src.matcher import MLMatcher


class TestMatcher(unittest.TestCase):

    def setUp(self):
        """Подготовка тестовых данных"""
        self.loader = DataLoader('./data')
        self.processor = TextProcessor()

    def test_data_loading(self):
        """Тест загрузки данных"""
        decl_df = self.loader.load_declarations()
        reg_df = self.loader.load_regulations()

        self.assertIsNotNone(decl_df)
        self.assertIsNotNone(reg_df)
        self.assertTrue(len(decl_df) > 0)
        self.assertTrue(len(reg_df) > 0)

        required_decl = ['declaration_id', 'G31_1', 'desc_extention']
        required_reg = ['regulation_id', 'code', 'description']

        for col in required_decl:
            self.assertIn(col, decl_df.columns)
        for col in required_reg:
            self.assertIn(col, reg_df.columns)

    def test_text_processing(self):
        """Тест обработки текста"""
        test_decl = pd.Series({
            'G31_1': 'ТЕЛЕФОННЫЙ АВТООТВЕТЧИК CALLBACK A1',
            'desc_extention': 'ФУНКЦИЯ ЗАПИСИ ОТСУТСТВУЕТ, 150 ШТ.'
        })

        text = self.processor.prepare_declaration_text(test_decl)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)
        self.assertTrue('телефонный' in text.lower())

        test_reg = pd.Series({
            'description': 'Телефонные аппараты',
            'explanation': '8517 11',
            'notes': 'Примечания к группе 85'
        })

        text = self.processor.prepare_regulation_text(test_reg)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_feature_extractor(self):
        """Тест извлечения признаков"""
        reg_df = pd.DataFrame([
            {'regulation_id': 'R001', 'code': '3004500006', 'description': 'Лекарства', 'explanation': '', 'notes': ''},
            {'regulation_id': 'R002', 'code': '8407343009', 'description': 'Двигатели', 'explanation': '', 'notes': ''},
        ])
        decl = pd.Series({
            'declaration_id': 'D001',
            'G31_1': 'таблетки витамин с',
            'desc_extention': 'лекарственный препарат',
            'G34': 'CH',
            'G32': '100',
        })

        extractor = HybridFeatureExtractor()
        extractor.fit_regulations(reg_df)
        features = extractor.extract_features_for_pair(decl, 0)

        self.assertIn('tfidf_cosine', features)
        self.assertIn('code_prefix_len', features)
        self.assertGreaterEqual(features['code_prefix_len'], 0)

    def test_matcher(self):
        """Тест матчера"""
        reg_df = pd.DataFrame([
            {'regulation_id': 'R001', 'code': '3004500006', 'description': 'Лекарства витамины', 'explanation': '', 'notes': ''},
            {'regulation_id': 'R002', 'code': '8407343009', 'description': 'Двигатели автомобильные', 'explanation': '', 'notes': ''},
            {'regulation_id': 'R003', 'code': '9110111000', 'description': 'Часы механизмы', 'explanation': '', 'notes': ''},
        ])
        decl = pd.Series({
            'declaration_id': 'D001',
            'G31_1': 'лекарственное средство с витамином с',
            'desc_extention': '',
            'G34': '',
            'G32': '',
        })

        extractor = HybridFeatureExtractor()
        extractor.fit_regulations(reg_df)
        matcher = MLMatcher(extractor, ranker='ridge')

        predictions = matcher.predict_fallback(decl, reg_df, top_k=2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(predictions[0]['regulation_id'] in ['R001', 'R002', 'R003'])


if __name__ == '__main__':
    unittest.main()
