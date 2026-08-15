from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ready'


def test_recommend():
    # payload de exemplo com categorias em string; não depende do dataset
    # processado (não versionado) para que o teste rode em um clone limpo.
    payload = {
        'features': {
            'age': 35, 'job': 'admin.', 'marital': 'married', 'education': 'secondary',
            'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular',
            'month': 'may', 'day_of_week': 'mon', 'campaign': 1, 'pdays': 999,
            'previous': 0, 'poutcome': 'unknown', 'emp.var.rate': 1.1,
            'cons.price.idx': 93.994, 'cons.conf.idx': -36.4, 'euribor3m': 4.857,
            'nr.employed': 5191.0,
        }
    }
    r = client.post('/api/v1/recommend', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'recommended' in data
    assert 'confidence' in data
    assert data['recommended'] in (0, 1)
    assert 0.0 <= data['confidence'] <= 1.0


def test_recommend_with_raw_categorical_values():
    payload = {'features': {'age': 35, 'job': 'admin.', 'marital': 'married'}}
    r = client.post('/api/v1/recommend', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'recommended' in data
    assert 'confidence' in data
