import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer

# Test the current optimization
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
tickers = optimizer.fetch_sp500_tickers(10)
print(f'Testing with tickers: {tickers}')

# Get data
data = optimizer.fetch_historical_data(tickers, '2y')
print(f'Data shape: {data.shape}')
print(f'Data date range: {data.index.min()} to {data.index.max()}')

# Calculate returns
returns = optimizer.calculate_returns(data)
print(f'Returns shape: {returns.shape}')
print('Returns stats:')
print(returns.describe())

# Run optimization
result = optimizer.optimize_portfolio(returns)
if result['status'] == 'optimal':
    weights = result['weights']
    tickers_list = result['tickers']
    
    print('\nOptimization Results:')
    for ticker, weight in zip(tickers_list, weights):
        if weight > 0.001:  # Show only meaningful weights
            print(f'{ticker}: {weight:.4f} ({weight*100:.2f}%)')
    
    print('\nPortfolio Stats:')
    stats = result['stats']
    print(f"Return: {stats['return']:.4f} ({stats['return']*100:.2f}%)")
    print(f"Volatility: {stats['volatility']:.4f} ({stats['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
else:
    print(f'Optimization failed: {result.get("message", "Unknown error")}')
