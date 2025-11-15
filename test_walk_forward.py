"""
Test script for walk-forward analysis functionality
"""
import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine
import matplotlib.pyplot as plt

def test_walk_forward_analysis():
    """Test the walk-forward backtesting functionality"""
    print("Testing Walk-Forward Analysis...")
    
    # Initialize optimizer and fetch data
    optimizer = PortfolioOptimizer()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']  # Small set for testing
    print(f"Testing with tickers: {tickers}")
    
    # Fetch historical data
    data = optimizer.fetch_historical_data(tickers, period="3y")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    if data.empty:
        print("❌ Failed to fetch data")
        return False
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(transaction_cost=0.001)
    
    # Test fixed weights backtesting
    print("\n=== Testing Fixed Weights Backtesting ===")
    returns = optimizer.calculate_returns(data)
    result = optimizer.optimize_portfolio(returns)
    
    if result['status'] == 'optimal':
        print("✅ Initial optimization successful")
        fixed_weights_result = backtest_engine.backtest_portfolio(
            result['weights'], data, 'Q'  # Quarterly rebalancing
        )
        print(f"Fixed weights backtest completed: {fixed_weights_result.shape}")
        print(f"Final portfolio value: ${fixed_weights_result['portfolio_value'].iloc[-1]:,.2f}")
    else:
        print("❌ Initial optimization failed")
        return False
    
    # Test walk-forward backtesting
    print("\n=== Testing Walk-Forward Backtesting ===")
    try:
        walk_forward_result = backtest_engine.backtest_portfolio_walk_forward(
            data, 'Q', min_history_days=252  # Quarterly rebalancing, 1 year min history
        )
        print(f"✅ Walk-forward backtest completed: {walk_forward_result.shape}")
        print(f"Final portfolio value: ${walk_forward_result['portfolio_value'].iloc[-1]:,.2f}")
        
        # Check if optimization_dates column exists and has values
        if 'optimization_dates' in walk_forward_result.columns:
            opt_dates = [d for d in walk_forward_result.get('optimization_dates', []) if d is not None]
            print(f"Number of re-optimizations: {len(opt_dates)}")
            if opt_dates:
                print(f"First optimization: {opt_dates[0]}")
                print(f"Last optimization: {opt_dates[-1]}")
        
    except Exception as e:
        print(f"❌ Walk-forward backtest failed: {e}")
        return False
    
    # Compare performance metrics
    print("\n=== Performance Comparison ===")
    
    # Fixed weights metrics
    fixed_returns = fixed_weights_result['returns']
    fixed_annual_return = fixed_returns.mean() * 252
    fixed_volatility = fixed_returns.std() * np.sqrt(252)
    fixed_sharpe = fixed_annual_return / fixed_volatility
    
    # Walk-forward metrics
    wf_returns = walk_forward_result['returns']
    wf_annual_return = wf_returns.mean() * 252
    wf_volatility = wf_returns.std() * np.sqrt(252)
    wf_sharpe = wf_annual_return / wf_volatility
    
    print(f"Fixed Weights    - Return: {fixed_annual_return:.1%}, Volatility: {fixed_volatility:.1%}, Sharpe: {fixed_sharpe:.2f}")
    print(f"Walk-Forward     - Return: {wf_annual_return:.1%}, Volatility: {wf_volatility:.1%}, Sharpe: {wf_sharpe:.2f}")
    
    # Transaction costs comparison
    fixed_total_costs = fixed_weights_result['rebalancing_costs'].sum()
    wf_total_costs = walk_forward_result['rebalancing_costs'].sum()
    
    print(f"Fixed Weights    - Total transaction costs: {fixed_total_costs:.4f}")
    print(f"Walk-Forward     - Total transaction costs: {wf_total_costs:.4f}")
    
    print("\n✅ All tests completed successfully!")
    return True

if __name__ == "__main__":
    test_walk_forward_analysis()
