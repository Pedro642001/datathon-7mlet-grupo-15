"""Utilities (compatibility wrappers).

Most runtime model/scaler/feature logic is encapsulated in src.app.services.model_service_wrapper.ModelService.
These functions delegate to the singleton ModelService to keep backwards compatibility with scripts/notebooks.
"""
from typing import Dict, Tuple
from src.app.services.model_service_wrapper import ModelService


def load_model_and_scaler(model_name: str = 'thompson', scaler_path: str = 'data/processed/scaler.pkl', train_csv: str = 'data/processed/train_clean.csv') -> Tuple[object, object, list]:
    """Delegate to ModelService.load_resources and return (model, scaler, feature_names)."""
    svc = ModelService.get_instance()
    # if not loaded or different model requested, (re)load resources
    if not svc.ready or svc.model_name != model_name:
        svc.load_resources(model_name=model_name, scaler_path=scaler_path, train_csv=train_csv)
        svc.ready = True
    return svc.model, svc.scaler, svc.feature_names


def prepare_features(input_features: Dict[str, float]):
    """Delegate to ModelService.prepare_features for compatibility."""
    svc = ModelService.get_instance()
    if not svc.ready:
        raise RuntimeError('ModelService not initialized. Call load_model_and_scaler first.')
    return svc.prepare_features(input_features)

