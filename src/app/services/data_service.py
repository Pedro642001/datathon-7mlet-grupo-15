"""Data service wrapper for ETL tasks.
Delegates to the existing DataPreparation pipeline in src.data_preparation.
This keeps data-related operations grouped under the services layer for runtime usage.
"""
from src.data_preparation import DataPreparation, prepare_data

class DataService:
    @staticmethod
    def prepare_data(data_path: str = 'data/raw/bank_marketing.csv', test_size: float = 0.3):
        return prepare_data(data_path=data_path, test_size=test_size)

    @staticmethod
    def get_preparer(data_path: str = 'data/raw/bank_marketing.csv'):
        return DataPreparation(data_path=data_path)
