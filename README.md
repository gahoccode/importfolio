# Portfolio Optimization Tool

## Overview
This tool provides portfolio optimization and analysis using several methods, including Mean-Variance (Max Sharpe), Minimum Variance, and Minimum CVaR. It features both analytical optimization and random simulation approaches for educational and visualization purposes.

## Metric Calculation Clarification

### Analytical Optimizer ("Performance Metrics")
- These metrics are produced by mathematical optimization algorithms (e.g., quadratic programming for mean-variance).
- They represent the true optimal solution for the selected method (e.g., the portfolio with the maximum possible Sharpe ratio).
- Displayed in the UI as "Performance Metrics" or similar.

### Random Portfolio Simulation ("Best Random Portfolio")
- These metrics are calculated by generating thousands of random portfolios and selecting the one with the best Sharpe ratio (or other metric).
- This is an approximate, Monte Carlo approach and may not reach the exact optimal solution.
- Useful for visualizing the efficient frontier and understanding the distribution of possible portfolios.
- Displayed in the UI as "Maximum Sharpe Ratio Portfolio (from Random Simulations)" or similar.

**Note:** The optimizer's metrics are generally more accurate. The random simulation is for illustration and exploration.

## Usage
1. Select tickers and date range.
2. Choose an optimization method.
3. Click "Optimize Portfolio" for analytical results, or "Show Random Portfolios" to see simulation results and visualizations.

## Requirements
- Python 3.x
- See `requirements.txt` for dependencies

## Setup
1. Create and activate a virtual environment:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Run the app:
   ```cmd
   python app.py
   ```

---

For further details, see code comments and docstrings in the respective modules.
