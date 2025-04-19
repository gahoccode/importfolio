from flask import Flask, render_template, request, jsonify, send_file
from data_loader import DataLoader
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/load_data", methods=["POST"])
def load_data():
    data = request.json
    symbols = data.get("symbols", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    loader = DataLoader(symbols, start_date, end_date)
    all_data = loader.fetch()
    combined = DataLoader.combine_close_prices(all_data)
    # Return as dict for test compatibility with columns preserved
    if isinstance(combined, dict):
        return jsonify(combined)
    else:
        result = combined.to_dict(orient="split")
        # Store original column names (tickers) in the response
        result["columns"] = combined.columns.tolist()
        return jsonify(result)

from optimizer import PortfolioOptimizer
import pandas as pd

@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        data = request.json
        
        # Reconstruct DataFrame with original column names (tickers)
        prices_data = data.get("prices")
        columns = data.get("columns", None)
        
        if prices_data is None:
            return jsonify({"error": "No price data provided"}), 400
        
        # Handle different possible data structures
        if isinstance(prices_data, dict):
            if "data" in prices_data and "index" in prices_data:
                # Handle the case where prices is a dict with data/index/columns format
                df_data = prices_data["data"]
                df_index = prices_data["index"]
                df_columns = columns or prices_data.get("columns", None)
                
                # Convert to DataFrame
                prices = pd.DataFrame(df_data, index=df_index, columns=df_columns)
                
                # Set index to datetime if it's not already
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index)
            else:
                # It's a dict but not in the expected format, try direct conversion
                prices = pd.DataFrame(prices_data)
        else:
            # Direct DataFrame construction for other formats
            prices = pd.DataFrame(prices_data)
        
        # Ensure numeric data
        prices = prices.apply(pd.to_numeric, errors='coerce')
        
        # Drop NA values to ensure data consistency
        prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if prices.empty:
            return jsonify({"error": "No valid data after dropping NA values"}), 400
            
        print(f"DataFrame after dropping NA: shape={prices.shape}")
        
        # Get optimization method and create optimizer
        method = data.get("method", "mean_variance")
        optimizer = PortfolioOptimizer(prices)
        
        # Run the selected optimization method
        if method == "mean_variance":
            weights, performance = optimizer.mean_variance_optimization()
        elif method == "min_variance":
            weights, performance = optimizer.min_variance_portfolio()
        elif method == "max_sharpe":
            weights, performance = optimizer.max_sharpe_portfolio()
        elif method == "cvar":
            weights = optimizer.cvar_portfolio()
            performance = None
        else:
            return jsonify({"error": "Unknown optimization method."}), 400
            
        # Format performance metrics with labels if available
        if performance:
            labeled_performance = {
                "expected_return": performance[0],
                "volatility": performance[1],
                "sharpe_ratio": performance[2]
            }
            return jsonify({"weights": weights, "performance": labeled_performance})
        else:
            return jsonify({"weights": weights, "performance": performance})
    except Exception as e:
        print(f"Error in optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/efficient_frontier", methods=["POST"])
def efficient_frontier():
    try:
        data = request.json
        print(f"Received data: {type(data)}")
        
        # Reconstruct DataFrame with original column names (tickers)
        prices_data = data.get("prices")
        columns = data.get("columns", None)
        
        print(f"Prices data type: {type(prices_data)}")
        if prices_data is None:
            return jsonify({"error": "No price data provided"}), 400
        
        # Debug the structure of prices_data
        if isinstance(prices_data, dict):
            print(f"Keys in prices_data: {prices_data.keys()}")
        
        # Handle different possible data structures
        if isinstance(prices_data, dict):
            if "data" in prices_data and "index" in prices_data:
                # Handle the case where prices is a dict with data/index/columns format
                df_data = prices_data["data"]
                df_index = prices_data["index"]
                df_columns = columns or prices_data.get("columns", None)
                
                # Convert to DataFrame
                prices = pd.DataFrame(df_data, index=df_index, columns=df_columns)
                
                # Set index to datetime if it's not already
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index)
            else:
                # It's a dict but not in the expected format, try direct conversion
                prices = pd.DataFrame(prices_data)
        else:
            # Direct DataFrame construction for other formats
            prices = pd.DataFrame(prices_data)
        
        print(f"Created DataFrame with shape: {prices.shape}")
        print(f"DataFrame columns: {prices.columns.tolist()}")
        print(f"DataFrame index type: {type(prices.index)}")
        
        # Ensure numeric data
        prices = prices.apply(pd.to_numeric, errors='coerce')
        
        # Drop NA values to ensure data consistency
        prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if prices.empty:
            return jsonify({"error": "No valid data after dropping NA values"}), 400
            
        print(f"DataFrame after dropping NA: shape={prices.shape}")
        
        # Create optimizer and calculate efficient frontier
        optimizer = PortfolioOptimizer(prices)
        risks, rets = optimizer.efficient_frontier_points()
        
        return jsonify({
            "risks": risks, 
            "returns": rets
        })
    except Exception as e:
        print(f"Error in efficient_frontier: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/random_portfolios", methods=["POST"])
def random_portfolios():
    try:
        import numpy as np
        data = request.json
        
        # Reconstruct DataFrame with original column names (tickers)
        prices_data = data.get("prices")
        columns = data.get("columns", None)
        
        if prices_data is None:
            return jsonify({"error": "No price data provided"}), 400
        
        # Handle different possible data structures
        if isinstance(prices_data, dict):
            if "data" in prices_data and "index" in prices_data:
                # Handle the case where prices is a dict with data/index/columns format
                df_data = prices_data["data"]
                df_index = prices_data["index"]
                df_columns = columns or prices_data.get("columns", None)
                
                # Convert to DataFrame
                prices = pd.DataFrame(df_data, index=df_index, columns=df_columns)
                
                # Set index to datetime if it's not already
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index)
            else:
                # It's a dict but not in the expected format, try direct conversion
                prices = pd.DataFrame(prices_data)
        else:
            # Direct DataFrame construction for other formats
            prices = pd.DataFrame(prices_data)
        
        # Ensure numeric data
        prices = prices.apply(pd.to_numeric, errors='coerce')
        
        # Drop NA values to ensure data consistency
        prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if prices.empty:
            return jsonify({"error": "No valid data after dropping NA values"}), 400
            
        print(f"DataFrame after dropping NA: shape={prices.shape}")
        
        # Create optimizer and generate random portfolios
        optimizer = PortfolioOptimizer(prices)
        n_samples = 5000
        w = np.random.dirichlet(np.ones(optimizer.exp_returns.shape[0]), n_samples)
        rets = w.dot(optimizer.exp_returns)
        stds = np.sqrt(np.diag(w @ optimizer.cov_matrix @ w.T))
        sharpes = rets / stds
        
        # Find the portfolio with the maximum Sharpe ratio
        max_sharpe_idx = np.argmax(sharpes)
        max_sharpe_ratio = sharpes[max_sharpe_idx]
        max_sharpe_return = rets[max_sharpe_idx]
        max_sharpe_risk = stds[max_sharpe_idx]
        max_sharpe_weights = {}
        
        # Create dictionary of weights with ticker symbols as keys
        for i, ticker in enumerate(optimizer.exp_returns.index):
            max_sharpe_weights[ticker] = float(w[max_sharpe_idx, i])
        
        return jsonify({
            "risks": stds.tolist(),
            "returns": rets.tolist(),
            "sharpes": sharpes.tolist(),
            "max_sharpe": {
                "sharpe_ratio": float(max_sharpe_ratio),
                "expected_return": float(max_sharpe_return),
                "risk": float(max_sharpe_risk),
                "weights": max_sharpe_weights
            }
        })
    except Exception as e:
        print(f"Error in random_portfolios: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/plot_efficient_frontier", methods=["POST"])
def plot_efficient_frontier():
    try:
        data = request.json
        
        # Reconstruct DataFrame with original column names (tickers)
        prices_data = data.get("prices")
        columns = data.get("columns", None)
        
        if prices_data is None:
            return jsonify({"error": "No price data provided"}), 400
        
        # Handle different possible data structures
        if isinstance(prices_data, dict):
            if "data" in prices_data and "index" in prices_data:
                # Handle the case where prices is a dict with data/index/columns format
                df_data = prices_data["data"]
                df_index = prices_data["index"]
                df_columns = columns or prices_data.get("columns", None)
                
                # Convert to DataFrame
                prices = pd.DataFrame(df_data, index=df_index, columns=df_columns)
                
                # Set index to datetime if it's not already
                if not isinstance(prices.index, pd.DatetimeIndex):
                    prices.index = pd.to_datetime(prices.index)
            else:
                # It's a dict but not in the expected format, try direct conversion
                prices = pd.DataFrame(prices_data)
        else:
            # Direct DataFrame construction for other formats
            prices = pd.DataFrame(prices_data)
        
        # Ensure numeric data
        prices = prices.apply(pd.to_numeric, errors='coerce')
        
        # Drop NA values to ensure data consistency
        prices = prices.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if prices.empty:
            return jsonify({"error": "No valid data after dropping NA values"}), 400
            
        print(f"DataFrame for plotting: shape={prices.shape}")
        
        # Create optimizer and generate the plot
        optimizer = PortfolioOptimizer(prices)
        n_random_portfolios = int(data.get("n_random_portfolios", 5000))
        _, img_buffer = optimizer.plot_efficient_frontier(n_random_portfolios=n_random_portfolios)
        
        # Return the image as a response
        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        print(f"Error in plot_efficient_frontier: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
