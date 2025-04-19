"""
Module: optimizer.py
Purpose: Portfolio optimization logic using PyPortfolioOpt
"""
from pypfopt import EfficientFrontier, risk_models, expected_returns
import numpy as np
import pandas as pd

# Standard financial variable naming
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02

class PortfolioOptimizer:
    def __init__(self, prices: pd.DataFrame):
        # Ensure all data is numeric, coerce errors to NaN, then drop columns/rows with NaN
        self.prices = prices.apply(pd.to_numeric, errors='coerce')
        self.prices = self.prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
        if self.prices.empty:
            raise ValueError("No valid numeric price data provided to PortfolioOptimizer.")
        self.returns = self._calculate_returns()
        self.cov_matrix = risk_models.sample_cov(self.prices)
        self.exp_returns = expected_returns.mean_historical_return(self.prices, frequency=TRADING_DAYS_PER_YEAR)

    def _calculate_returns(self):
        returns = self.prices.pct_change().dropna()
        return returns

    def mean_variance_optimization(self):
        ef = EfficientFrontier(self.exp_returns, self.cov_matrix)
        weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
        return cleaned_weights, performance

    def min_variance_portfolio(self):
        ef = EfficientFrontier(self.exp_returns, self.cov_matrix)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
        return cleaned_weights, performance

    def max_sharpe_portfolio(self):
        ef = EfficientFrontier(self.exp_returns, self.cov_matrix)
        weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)
        return cleaned_weights, performance

    def cvar_portfolio(self):
        from pypfopt import EfficientCVaR
        ef = EfficientCVaR(returns=self.returns, expected_returns=self.exp_returns)
        weights = ef.min_cvar()
        cleaned_weights = ef.clean_weights()
        return cleaned_weights

    def efficient_frontier_points(self):
        from pypfopt.efficient_frontier import EfficientFrontier
        risks = []
        rets = []
        num_points = 5000
        min_risk = 0.001
        for risk_aversion in np.linspace(min_risk, 1, num_points):
            ef = EfficientFrontier(self.exp_returns, self.cov_matrix)
            try:
                ef.efficient_risk(target_volatility=risk_aversion * self.returns.std().mean())
                perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
                risks.append(perf[1])
                rets.append(perf[0])
            except Exception:
                continue
        return risks, rets
        
    def plot_efficient_frontier(self, n_random_portfolios=10000):
        """Generate an efficient frontier plot with random portfolios.
        
        Args:
            n_random_portfolios: Number of random portfolios to generate
            
        Returns:
            A tuple containing (fig, buffer) where fig is the matplotlib figure and
            buffer is the binary image data that can be sent to the client
        """
        import matplotlib.pyplot as plt
        import io
        from pypfopt import plotting
        
        # Create a new EfficientFrontier object
        from pypfopt.efficient_frontier import EfficientFrontier
        ef = EfficientFrontier(self.exp_returns, self.cov_matrix)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the efficient frontier
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
        
        # Find the tangency portfolio (max Sharpe ratio)
        ef_max_sharpe = EfficientFrontier(self.exp_returns, self.cov_matrix)
        ef_max_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        ret_tangent, std_tangent, sharpe_tangent = ef_max_sharpe.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label=f"Max Sharpe: {sharpe_tangent:.2f}")
        
        # Find the minimum volatility portfolio
        ef_min_vol = EfficientFrontier(self.exp_returns, self.cov_matrix)
        ef_min_vol.min_volatility()
        ret_min_vol, std_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        ax.scatter(std_min_vol, ret_min_vol, marker="P", s=100, c="g", label=f"Min Volatility: {sharpe_min_vol:.2f}")
        
        # Generate random portfolios
        np.random.seed(42)  # For reproducibility
        w = np.random.dirichlet(np.ones(ef.n_assets), n_random_portfolios)
        rets = w.dot(self.exp_returns)
        stds = np.sqrt(np.diag(w @ self.cov_matrix @ w.T))
        sharpes = rets / stds
        
        # Find the random portfolio with the highest Sharpe ratio
        max_sharpe_idx = np.argmax(sharpes)
        max_sharpe_ret = rets[max_sharpe_idx]
        max_sharpe_std = stds[max_sharpe_idx]
        max_sharpe_ratio = sharpes[max_sharpe_idx]
        
        # Plot random portfolios with color based on Sharpe ratio
        scatter = ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis", alpha=0.5)
        
        # Highlight the random portfolio with the highest Sharpe ratio
        ax.scatter(max_sharpe_std, max_sharpe_ret, marker="X", s=100, c="purple", 
                  label=f"Best Random: {max_sharpe_ratio:.2f}")
        
        # Add a colorbar to show Sharpe ratio scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Add labels and title
        ax.set_title("Efficient Frontier with Random Portfolios")
        ax.set_xlabel("Volatility (Standard Deviation)")
        ax.set_ylabel("Expected Return")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add risk-free rate and capital market line
        x_range = np.linspace(0, max(stds) * 1.2, 100)
        y_range = RISK_FREE_RATE + (ret_tangent - RISK_FREE_RATE) * x_range / std_tangent
        ax.plot(x_range, y_range, label=f"Capital Market Line (Rf={RISK_FREE_RATE:.2%})", linestyle="--", color="black")
        
        plt.tight_layout()
        
        # Save the figure to a binary stream
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        return fig, buf
