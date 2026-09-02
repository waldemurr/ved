import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Загрузка и предобработка данных"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.declarations = None
        self.regulations = None
        
    def load_declarations(self) -> pd.DataFrame:
        """Загрузка деклараций из jsonl"""
        logger.info("Загрузка деклараций...")
        decls = []
        
        with open(self.data_path / 'declarations.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    decls.append(json.loads(line))
        
        self.declarations = pd.DataFrame(decls)
        logger.info(f"Загружено {len(self.declarations)} деклараций")
        return self.declarations
    
    def load_regulations(self) -> pd.DataFrame:
        """Загрузка регуляций из jsonl"""
        logger.info("Загрузка регуляций...")
        regs = []
        
        with open(self.data_path / 'regulations.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    regs.append(json.loads(line))
        
        self.regulations = pd.DataFrame(regs)
        logger.info(f"Загружено {len(self.regulations)} регуляций")
        return self.regulations
    
    def load_tnved_knowledge(self) -> str:
        """Загрузка дополнительного контекста ТН ВЭД"""
        try:
            with open(self.data_path / 'tnved_knowledge.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Файл tnved_knowledge.txt не найден")
            return ""