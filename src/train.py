"""Pipeline de treino: prepara os dados (se necessário), treina o Baseline e
três configurações de prior do Thompson Sampling contextual, registra cada
execução no MLflow (experimento 'datathon-offers') e salva o melhor modelo
Thompson e o Baseline em models/*.json.

Uso:
    python src/train.py

É a mesma lógica usada pelo notebook notebooks/02_baseline_thompson.ipynb,
para que notebook e código de produção nunca fiquem dessincronizados.
"""

import os
import sys
from pathlib import Path

# Executado como 'python src/train.py', o Python coloca src/ no sys.path — e não
# a raiz do repositório —, então os imports 'from src.x import y' abaixo não
# resolvem. Inserimos a raiz antes deles, mesma abordagem já usada pelos
# notebooks. Assim funcionam tanto 'python src/train.py' quanto 'python -m src.train'.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# No Windows, redirecionar a saída (python src/train.py > treino.log) faz o
# Python usar cp1252, e os emojis dos prints de progresso derrubam o treino com
# UnicodeEncodeError antes de qualquer modelo ser treinado.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8')

import mlflow
import pandas as pd

from src.data_preparation import prepare_data
from src.recommender import BaselineRecommender, OfferRecommender, ThompsonSampler, evaluate_recommender

EXPERIMENT_NAME = 'datathon-offers'

# Fixa o tracking do MLflow em um arquivo SQLite dentro do repositório
# (mlflow/mlflow.db), independente de qualquer configuração global de MLflow
# na máquina de quem executa (o MLflow pode ter um backend padrão diferente
# definido em ~/.config/mlflow). A partir do MLflow 3.x o backend baseado
# em arquivos ('./mlruns') está em modo de manutenção para tracking de
# metadados, então usamos SQLite para isso — mas os artefatos (os JSON dos
# modelos logados) continuam indo para uma pasta de arquivos, que também
# fixamos dentro de mlflow/mlruns em vez do padrão './mlruns' na raiz.
MLFLOW_DIR = Path(__file__).resolve().parent.parent / 'mlflow'
MLFLOW_DB_PATH = MLFLOW_DIR / 'mlflow.db'
MLFLOW_ARTIFACT_DIR = MLFLOW_DIR / 'mlruns'

# Três priors diferentes para o Thompson Sampling contextual, do mais
# uninformativo ao mais informado pela taxa de aceitação histórica (~11%).
PRIOR_CONFIGS = [
    {'name': 'uninformative', 'alpha0': 1.0, 'beta0': 1.0},
    {'name': 'weakly-informative', 'alpha0': 2.0, 'beta0': 16.0},
    {'name': 'strong-informative', 'alpha0': 5.0, 'beta0': 40.0},
]


def load_or_prepare_data():
    if not (os.path.exists('data/processed/train_clean.csv') and os.path.exists('data/processed/test_clean.csv')):
        prepare_data()
    train_df = pd.read_csv('data/processed/train_clean.csv')
    test_df = pd.read_csv('data/processed/test_clean.csv')
    X_train, y_train = train_df.drop('y', axis=1), train_df['y']
    X_test, y_test = test_df.drop('y', axis=1), test_df['y']
    return X_train, y_train, X_test, y_test


def train_baseline(X_train, y_train, X_test, y_test):
    baseline = OfferRecommender(strategy='baseline')
    baseline.fit(X_train, y_train)
    metrics = evaluate_recommender(baseline, X_test, y_test, name='BASELINE')
    return baseline, metrics


def run_thompson_experiment(alpha0, beta0, X_train, y_train, X_test, y_test, name=''):
    sampler = ThompsonSampler(alpha0=alpha0, beta0=beta0)
    model = OfferRecommender(strategy='thompson', model_obj=sampler)
    model.fit(X_train, y_train)
    metrics = evaluate_recommender(model, X_test, y_test, name=f'THOMPSON {name} (alpha0={alpha0}, beta0={beta0})')
    metrics['n_segments'] = len(sampler.segments)
    return model, metrics


def main():
    X_train, y_train, X_test, y_test = load_or_prepare_data()

    mlflow.set_tracking_uri(f'sqlite:///{MLFLOW_DB_PATH}')
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f'file:{MLFLOW_ARTIFACT_DIR}')
    mlflow.set_experiment(EXPERIMENT_NAME)

    baseline, baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)
    with mlflow.start_run(run_name='baseline'):
        mlflow.log_param('strategy', 'baseline')
        mlflow.log_metrics(baseline_metrics)

    os.makedirs('models', exist_ok=True)
    baseline.save_json('models/baseline.json')

    best_model, best_metrics, best_name = None, None, None
    for cfg in PRIOR_CONFIGS:
        model, metrics = run_thompson_experiment(
            cfg['alpha0'], cfg['beta0'], X_train, y_train, X_test, y_test, name=cfg['name']
        )
        with mlflow.start_run(run_name=f"thompson-{cfg['name']}"):
            mlflow.log_params(
                {
                    'strategy': 'thompson',
                    'alpha0': cfg['alpha0'],
                    'beta0': cfg['beta0'],
                    'segment_columns': ','.join(ThompsonSampler.SEGMENT_COLUMNS),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_metric('baseline_actual_acceptance', baseline_metrics['actual_acceptance'])

            tmp_path = f"models/_tmp_thompson_{cfg['name']}.json"
            model.save_json(tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path='model')
            os.remove(tmp_path)

        if best_metrics is None or metrics['actual_acceptance'] > best_metrics['actual_acceptance']:
            best_model, best_metrics, best_name = model, metrics, cfg['name']

    best_model.save_json('models/thompson.json')

    print('\n' + '=' * 70)
    print('RESUMO')
    print('=' * 70)
    print(f"Baseline  -> taxa de aceitação real: {baseline_metrics['actual_acceptance']*100:.1f}%")
    print(f"Thompson* -> prior escolhido: {best_name}")
    print(f"Thompson  -> taxa de aceitação real: {best_metrics['actual_acceptance']*100:.1f}%")
    if baseline_metrics['actual_acceptance'] > 0:
        ganho = (best_metrics['actual_acceptance'] - baseline_metrics['actual_acceptance']) / baseline_metrics['actual_acceptance']
        print(f"Ganho relativo: {ganho*100:.1f}%")
    print('\nModelos salvos em models/baseline.json e models/thompson.json')
    print(f"Rode 'mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB_PATH}' para visualizar as execuções registradas.")


if __name__ == '__main__':
    main()
