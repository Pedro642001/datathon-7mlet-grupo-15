"""
Baseline and Thompson Sampling Module
Implementa estratégias de recomendação de ofertas.

O Thompson Sampling é contextual por segmentação: o cliente é
classificado em um "segmento de comportamento" a partir de (age, job) e
cada segmento mantém sua própria posterior Beta(alpha, beta) de taxa de
aceitação. Isso faz o contexto do cliente entrar diretamente na decisão
(clientes de segmentos diferentes recebem confiança e recomendação
diferentes), em vez de um único par alpha/beta global.

Modelos são serializados em JSON (não pickle) para evitar execução de
código arbitrário ao carregar um modelo salvo.
"""

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd


class BaselineRecommender:
    """
    Baseline: regra fixa que sempre oferece o produto (política de controle).
    A métrica de comparação é a taxa de aceitação histórica observada no treino.
    """

    def __init__(self, acceptance_rate: float = None):
        self.acceptance_rate = acceptance_rate

    @classmethod
    def from_train(cls, y: pd.Series):
        acceptance_rate = (y == 1).sum() / len(y)
        return cls(acceptance_rate=acceptance_rate)

    def fit(self):
        return self

    def recommend(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1 - self.acceptance_rate
        probs[:, 1] = self.acceptance_rate
        return probs

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'baseline', 'acceptance_rate': float(self.acceptance_rate)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(acceptance_rate=float(data['acceptance_rate']))


class ThompsonSampler:
    """
    Thompson Sampling contextual e segmentado.

    Contexto: cada cliente é mapeado para um segmento a partir de duas
    features já presentes no dataset (age, job). 'age' é discretizada em
    faixas (quartis calculados no treino) e 'job' já é uma categoria
    codificada, a combinação (faixa_de_idade, profissão) define o
    segmento. Cada segmento tem seu próprio par (alpha, beta), então a
    posterior, e portanto a confiança e a recomendação, variam de
    cliente para cliente conforme o contexto.

    A decisão de ofertar (arm=1) ou não (arm=0) para um cliente é feita
    amostrando theta ~ Beta(alpha_segmento, beta_segmento) e comparando
    com um limiar de referência (por padrão, a taxa de aceitação global
    do treino): recomenda-se a oferta quando o valor amostrado sugere uma
    propensão acima da média histórica. A amostragem é o mecanismo de
    exploração do Thompson Sampling, chamadas repetidas para o mesmo
    cliente podem gerar decisões diferentes por natureza do algoritmo.
    """

    SEGMENT_COLUMNS = ('age', 'job')

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        age_bin_edges=None,
        segments: Dict[str, Dict[str, float]] = None,
        threshold: float = None,
        random_state: int = 42,
    ):
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.age_bin_edges = list(age_bin_edges) if age_bin_edges is not None else None
        self.segments: Dict[str, Dict[str, float]] = segments if segments is not None else {}
        self.threshold = threshold
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)

    def _segment_key(self, age_val: float, job_val: float) -> str:
        if self.age_bin_edges:
            age_bin = int(np.digitize([age_val], self.age_bin_edges)[0])
        else:
            age_bin = 0
        job_key = round(float(job_val), 4)
        return f"{age_bin}|{job_key}"

    def _segment_keys_for(self, X) -> np.ndarray:
        ages = X['age'].to_numpy()
        jobs = X['job'].to_numpy()
        return np.array([self._segment_key(a, j) for a, j in zip(ages, jobs)])

    def _get_segment(self, key: str) -> Dict[str, float]:
        return self.segments.get(key, {'alpha': self.alpha0, 'beta': self.beta0})

    def fit(self, X, y):
        if self.age_bin_edges is None:
            self.age_bin_edges = [float(v) for v in np.quantile(X['age'].to_numpy(), [0.25, 0.5, 0.75])]

        keys = self._segment_keys_for(X)
        y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)

        for key, reward in zip(keys, y_arr):
            seg = self.segments.setdefault(key, {'alpha': self.alpha0, 'beta': self.beta0})
            if int(reward) == 1:
                seg['alpha'] += 1
            else:
                seg['beta'] += 1

        if self.threshold is None:
            self.threshold = float(y_arr.mean())
        return self

    def recommend(self, X):
        keys = self._segment_keys_for(X)
        threshold = self.threshold if self.threshold is not None else 0.5
        out = np.empty(len(keys), dtype=int)
        for i, key in enumerate(keys):
            seg = self._get_segment(key)
            theta = self._rng.beta(seg['alpha'], seg['beta'])
            out[i] = 1 if theta >= threshold else 0
        return out

    def predict_proba(self, X):
        keys = self._segment_keys_for(X)
        probs = np.zeros((len(keys), 2))
        for i, key in enumerate(keys):
            seg = self._get_segment(key)
            p1 = seg['alpha'] / (seg['alpha'] + seg['beta'])
            probs[i, 1] = p1
            probs[i, 0] = 1 - p1
        return probs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'thompson',
            'alpha0': self.alpha0,
            'beta0': self.beta0,
            'age_bin_edges': self.age_bin_edges,
            'segments': self.segments,
            'threshold': self.threshold,
            'random_state': self.random_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            alpha0=data.get('alpha0', 1.0),
            beta0=data.get('beta0', 1.0),
            age_bin_edges=data.get('age_bin_edges'),
            segments=data.get('segments', {}),
            threshold=data.get('threshold'),
            random_state=data.get('random_state', 42),
        )


class OfferRecommender:
    """
    Wrapper que expõe uma interface única (fit/recommend/predict_proba) por
    cima de um BaselineRecommender ou ThompsonSampler, com serialização
    segura em JSON.
    """

    def __init__(self, strategy: str = 'thompson', model_obj=None):
        self.strategy = strategy
        if model_obj is None:
            self.model = BaselineRecommender() if strategy == 'baseline' else ThompsonSampler()
        else:
            self.model = model_obj

    def fit(self, X, y):
        if isinstance(self.model, BaselineRecommender):
            self.model = BaselineRecommender.from_train(y)
        else:
            self.model.fit(X, y)
        return self

    def recommend(self, X):
        return self.model.recommend(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def to_dict(self) -> Dict[str, Any]:
        data = self.model.to_dict()
        data['strategy'] = self.strategy
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        strat = data.get('strategy', data.get('type', 'thompson'))
        if data['type'] == 'baseline':
            model = BaselineRecommender.from_dict(data)
        else:
            model = ThompsonSampler.from_dict(data)
        return cls(strategy=strat, model_obj=model)

    def save_json(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load_json(filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return OfferRecommender.from_dict(data)


def evaluate_recommender(recommender, X_test, y_test, name=""):
    print(f"\nAvaliação: {name}")
    print("=" * 60)
    recommendations = recommender.recommend(X_test)
    probs = recommender.predict_proba(X_test)
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    expected_acceptance = probs[:, 1].mean() if probs.shape[1] > 1 else probs[:, 0].mean()
    correct = (recommendations == y_test).sum()
    accuracy = correct / len(y_test)
    recommended = (recommendations == 1).sum()
    precision = (recommendations[y_test == 1] == 1).sum() / recommended if recommended > 0 else 0
    recall = (recommendations[y_test == 1] == 1).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
    actual_acceptance_when_recommended = y_test[recommendations == 1].mean() if recommended > 0 else 0
    print(f"  Accuracy:                    {accuracy*100:.1f}%")
    print(f"  Precision:                   {precision*100:.1f}%")
    print(f"  Recall:                      {recall*100:.1f}%")
    print(f"  Taxa de aceitação esperada:  {expected_acceptance*100:.1f}%")
    print(f"  Taxa de aceitação real:      {actual_acceptance_when_recommended*100:.1f}%")
    print(f"  Clientes recomendados:       {recommended:,} ({recommended/len(y_test)*100:.1f}%)")
    print("=" * 60)
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'expected_acceptance': float(expected_acceptance),
        'actual_acceptance': float(actual_acceptance_when_recommended),
        'n_recommended': int(recommended),
    }


if __name__ == '__main__':
    print("Módulo de Thompson Sampling contextual carregado")
