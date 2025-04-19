Create comprehensive tests for a portfolio optimization web application built with Flask and PyPortfolioOpt. Focus on testing:

1. Data preparation functionality
   - Test chronological sorting of time series data
   - Verify proper calculation of returns using pct_change()
   - Ensure NaN values are appropriately handled with dropna()
   - Check data validation for sufficient history and quality

2. Core optimization functionality
   - Test Mean-Variance Optimization calculations 
   - Verify minimum variance portfolio identification
   - Validate maximum Sharpe ratio portfolio calculations
   - Test Conditional Value at Risk (CVaR) computation accuracy

3. Visualization components
   - Test generation of efficient frontier plots
   - Verify correct marking of optimal portfolios on the frontier
   - Ensure asset allocation pie charts accurately reflect portfolio weights
   - Test downloadable report generation (PDF/CSV)

4. Web interface functionality
   - Test stock ticker input and validation
   - Verify date range selection and processing
   - Test parameter adjustment and recalculation functionality
   - Ensure proper error handling for invalid inputs

Include both unit tests for individual components and integration tests for end-to-end functionality. The tests should verify mathematical accuracy of optimization results against known benchmarks and ensure the web interface correctly presents optimization outcomes.