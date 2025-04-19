import pytest
import numpy as np
import pandas as pd
from optimizer import PortfolioOptimizer

def generate_sample_prices():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10)
    aaa = np.cumsum(np.random.normal(0, 1, 10)) + 100
    bbb = np.cumsum(np.random.normal(0, 1, 10)) + 120
    data = {
        'AAA_close': aaa,
        'BBB_close': bbb
    }
    prices = pd.DataFrame(data, index=dates)
    return prices

def test_returns_calculation():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    returns = optimizer.returns
    assert not returns.isnull().values.any()
    assert returns.shape[0] == prices.shape[0] - 1

def test_mean_variance_optimization():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    weights, performance = optimizer.mean_variance_optimization()
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert performance is not None

def test_min_variance_portfolio():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    weights, performance = optimizer.min_variance_portfolio()
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert performance is not None

def test_max_sharpe_portfolio():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    weights, performance = optimizer.max_sharpe_portfolio()
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1) < 1e-6
    assert performance is not None

def test_cvar_portfolio():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    weights = optimizer.cvar_portfolio()
    assert isinstance(weights, dict)
    assert abs(sum(weights.values()) - 1) < 1e-6

def test_efficient_frontier_points():
    prices = generate_sample_prices()
    optimizer = PortfolioOptimizer(prices)
    risks, rets = optimizer.efficient_frontier_points()
    assert len(risks) > 0
    assert len(risks) == len(rets)
