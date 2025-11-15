from portfolio_optimizer import BacktestEngine, PortfolioOptimizer

print("Testing walk-forward functionality...")
opt = PortfolioOptimizer()
data = opt.fetch_historical_data(['AAPL', 'MSFT', 'GOOGL'], '1y')
print(f"Data loaded: {data.shape}")

engine = BacktestEngine()
result = engine.backtest_portfolio_walk_forward(data, 'Q')
print(f"Walk-forward test successful!")
print(f"Final value: ${result['portfolio_value'].iloc[-1]:.2f}")
print(f"Optimizations: {result.attrs.get('optimization_count', 0)}")
