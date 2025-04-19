import pytest
from data_loader import DataLoader

class DummyQuote:
    def __init__(self, symbol):
        self.symbol = symbol
    def history(self, start, end, interval, to_df):
        import pandas as pd
        return pd.DataFrame({
            'time': ['2024-01-01', '2024-01-02'],
            'close': [10, 11],
            'open': [9, 10]
        })

def test_fetch(monkeypatch):
    monkeypatch.setattr('data_loader.Quote', DummyQuote)
    loader = DataLoader(['AAA'], '2024-01-01', '2024-01-02')
    data = loader.fetch()
    assert 'AAA' in data
    assert not data['AAA'].empty

def test_combine_close_prices():
    import pandas as pd
    data_dict = {
        'AAA': pd.DataFrame({
            'time': ['2024-01-01', '2024-01-02'],
            'close': [10, 11]
        }),
        'BBB': pd.DataFrame({
            'time': ['2024-01-01', '2024-01-02'],
            'close': [20, 21]
        })
    }
    combined = DataLoader.combine_close_prices(data_dict)
    assert 'AAA' in combined.columns
    assert 'BBB' in combined.columns
    assert list(combined['AAA']) == [10, 11]
