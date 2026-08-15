from fastapi.testclient import TestClient
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app.main import app
from src.utils import load_model_and_scaler

# ensure model loaded
load_model_and_scaler()

client = TestClient(app)

# health
r = client.get('/health')
print('health', r.status_code, r.json())

# prepare sample
df = pd.read_csv('data/processed/test_clean.csv')
row = df.drop('y', axis=1).iloc[0].to_dict()
payload = {'features': {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in row.items()}}

r = client.post('/api/v1/recommend', json=payload)
print('recommend', r.status_code, r.json())
