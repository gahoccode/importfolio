import pytest
from app import app as flask_app
import json
import pandas as pd

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_load_data_endpoint(client, monkeypatch):
    class DummyLoader:
        @staticmethod
        def fetch():
            return {
                'AAA': pd.DataFrame({'time': ['2024-01-01', '2024-01-02'], 'close': [10, 11]})
            }
        @staticmethod
        def combine_close_prices(data_dict):
            return pd.DataFrame({'time': ['2024-01-01', '2024-01-02'], 'AAA': [10, 11]})
    monkeypatch.setattr('data_loader.DataLoader', DummyLoader)
    response = client.post('/api/load_data', json={
        'symbols': ['AAA'],
        'start_date': '2024-01-01',
        'end_date': '2024-01-02'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'AAA' in data['data'][0] or 'AAA' in data['columns']

def test_optimize_endpoint(client):
    prices = pd.DataFrame({
        'AAA_close': [10, 11, 12, 13],
        'BBB_close': [20, 21, 22, 23]
    })
    response = client.post('/api/optimize', json={
        'prices': prices.to_dict(orient='list'),
        'method': 'mean_variance'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'weights' in data
    assert 'performance' in data

def test_efficient_frontier_endpoint(client):
    prices = pd.DataFrame({
        'AAA_close': [10, 11, 12, 13],
        'BBB_close': [20, 21, 22, 23]
    })
    response = client.post('/api/efficient_frontier', json={
        'prices': prices.to_dict(orient='list')
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'risks' in data
    assert 'returns' in data
