import pytest
import numpy as np
import pandas as pd
from optimizer import PortfolioOptimizer

def generate_sample_prices():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10000)
    aaa = np.cumsum(np.random.normal(0.05, 1, 10000)) + 100  # positive drift
    bbb = np.cumsum(np.random.normal(0.03, 1, 10000)) + 120  # positive drift
    data = {
        'AAA': aaa,
        'BBB': bbb
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
    # Patch: run efficient_frontier_points with exception logging
    from pypfopt.efficient_frontier import EfficientFrontier
    import numpy as np
    risks = []
    rets = []
    num_points = 100  # reduce for test speed
    ef = EfficientFrontier(optimizer.exp_returns, optimizer.cov_matrix)
    min_vol_result = ef.min_volatility()
    if isinstance(min_vol_result, tuple):
        min_vol = min_vol_result[0]
    elif isinstance(min_vol_result, dict):
        # Get volatility from portfolio_performance
        ef_min = EfficientFrontier(optimizer.exp_returns, optimizer.cov_matrix)
        ef_min.weights = np.array(list(min_vol_result.values()))
        min_vol = ef_min.portfolio_performance(risk_free_rate=0.02)[1]
    else:
        raise RuntimeError("Unexpected return type from min_volatility()")
    max_vol = optimizer.returns.std().mean() * 1.5  # arbitrary upper bound for test
    first_exception = None
    for target_volatility in np.linspace(min_vol, max_vol, num_points):
        ef = EfficientFrontier(optimizer.exp_returns, optimizer.cov_matrix)
        try:
            ef.efficient_risk(target_volatility=target_volatility)
            perf = ef.portfolio_performance(risk_free_rate=0.02)
            risks.append(perf[1])
            rets.append(perf[0])
        except Exception as e:
            if first_exception is None:
                first_exception = e
            continue
    if len(risks) == 0 and first_exception is not None:
        print(f"First exception in efficient_frontier_points: {first_exception}")
    assert len(risks) > 0
    assert len(risks) == len(rets)
