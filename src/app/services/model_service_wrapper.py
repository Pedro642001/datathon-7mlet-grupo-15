import logging
from typing import Dict, Tuple
from src.recommender import OfferRecommender
import os
import json
import pandas as pd

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None

    def __init__(self, model_name: str = 'thompson'):
        self.model_name = model_name
        self.model: OfferRecommender | None = None
        self.scaler = None
        self.feature_names = None
        self.encoders = {}
        self.ready = False
        # load resources
        self._load()

    def _load(self):
        try:
            self.load_resources(self.model_name)
            self.ready = True
            logger.info('ModelService loaded model=%s features=%d', self.model_name, len(self.feature_names))
        except Exception as e:
            logger.exception('Failed to load model service: %s', e)
            self.ready = False

    def load_resources(self, model_name: str = 'thompson', scaler_path: str = 'data/processed/scaler.pkl', train_csv: str = 'data/processed/train_clean.csv'):
        """Load model JSON/pickle, scaler, feature names and encoders. Sets instance attributes."""
        # model
        path = f'models/{model_name}.json'
        if os.path.exists(path):
            self.model = OfferRecommender.load_json(path)
            logger.info('Loaded model from %s', path)
        else:
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/train.py' to train and save the models first."
            )

        # scaler
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info('Loaded scaler from %s', scaler_path)
        else:
            raise FileNotFoundError(scaler_path)

        # feature names: prefer o artefato pequeno e versionado
        # (data/processed/feature_names.json), gerado por data_preparation.py, para não
        # depender do dataset processado completo (não versionado) só para
        # obter a lista de colunas.
        feature_names_path = 'data/processed/feature_names.json'
        if os.path.exists(feature_names_path):
            with open(feature_names_path) as f:
                self.feature_names = json.load(f)
            logger.info('Feature names loaded from %s (%d)', feature_names_path, len(self.feature_names))
        elif os.path.exists(train_csv):
            df = pd.read_csv(train_csv)
            self.feature_names = df.drop('y', axis=1).columns.tolist()
            logger.info('Feature names loaded from %s (%d)', train_csv, len(self.feature_names))
        else:
            raise FileNotFoundError(f'{feature_names_path} or {train_csv}')

        # encoders
        enc_path = 'data/processed/encoders.json'
        if os.path.exists(enc_path):
            with open(enc_path, 'r') as f:
                self.encoders = json.load(f)
            logger.info('Loaded encoders from %s (%d columns)', enc_path, len(self.encoders))
        else:
            self.encoders = {}
            logger.info('No encoders file found at %s; encoders empty', enc_path)

    def prepare_features(self, input_features: Dict[str, float]):
        """Prepare features (map categorical strings to indices, fill missing, apply scaler). Returns DataFrame."""
        if self.feature_names is None:
            raise RuntimeError('Feature names not loaded')

        enc = self.encoders or {}
        row = {}
        for name in self.feature_names:
            if name in enc:
                val = input_features.get(name, None)
                if val is None:
                    row[name] = 0.0
                else:
                    if isinstance(val, str):
                        classes = enc[name]
                        try:
                            idx = classes.index(val)
                            row[name] = float(idx)
                        except ValueError:
                            logger.warning('Unknown categorical for %s: %s -> mapping to 0', name, val)
                            row[name] = 0.0
                    else:
                        try:
                            row[name] = float(val)
                        except Exception:
                            row[name] = 0.0
            else:
                try:
                    row[name] = float(input_features.get(name, 0.0))
                except Exception:
                    row[name] = 0.0

        df = pd.DataFrame([row], columns=self.feature_names)
        if self.scaler is not None:
            arr = self.scaler.transform(df)
            df_scaled = pd.DataFrame(arr, columns=self.feature_names)
            return df_scaled
        return df

    def recommend(self, features: Dict[str, float]) -> Tuple[int, float]:
        if not self.ready:
            raise RuntimeError('Service not ready')
        X = self.prepare_features(features)
        rec = self.model.recommend(X)
        probs = self.model.predict_proba(X)
        recommended = int(rec[0])
        confidence = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
        return recommended, confidence

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    # Persistence helpers (migrated from src/services/model_service)
    @staticmethod
    def save_model_json(recommender: OfferRecommender, name: str):
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f'{name}.json')
        recommender.save_json(path)
        logger.info('Saved model JSON to %s', path)
        return path

    @staticmethod
    def load_model_json(name: str) -> OfferRecommender:
        path = os.path.join('models', f'{name}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return OfferRecommender.load_json(path)