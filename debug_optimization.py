"""
Debug script to test portfolio optimization with various parameters
"""

import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer
import warnings
warnings.filterwarnings('ignore')

def test_optimization():
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION DEBUG TEST")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Test with small set of stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    print(f"\n1. Testing with {len(test_tickers)} stocks: {test_tickers}")
    print("-" * 80)
    
    # Fetch data
    print("Fetching data...")
    data = optimizer.fetch_historical_data(test_tickers, period='1y')
    
    if data is None or data.empty:
        print("❌ FAILED: Could not fetch data")
        return
    
    print(f"✓ Data fetched: {len(data)} days, {len(data.columns)} stocks")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Missing values: {data.isna().sum().sum()}")
    
    # Calculate returns
    print("\nCalculating returns...")
    returns = optimizer.calculate_returns(data)
    
    if returns is None or returns.empty:
        print("❌ FAILED: Could not calculate returns")
        return
    
    print(f"✓ Returns calculated: {len(returns)} days")
    print(f"  Missing values: {returns.isna().sum().sum()}")
    print(f"  Returns range: {returns.min().min():.4f} to {returns.max().max():.4f}")
    
    # Clean returns
    returns = returns.dropna()
    print(f"  After cleaning: {len(returns)} days")
    
    # Test different optimization scenarios
    test_cases = [
        {"name": "Basic (no constraints)", "params": {}},
        {"name": "Max weight 20%", "params": {"max_weight": 0.20}},
        {"name": "Max weight 50%", "params": {"max_weight": 0.50}},
        {"name": "Max weight 30% + min 3 positions", "params": {"max_weight": 0.30, "min_positions": 3}},
        {"name": "Max weight 100% (concentrated)", "params": {"max_weight": 1.00}},
    ]
    
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZATION SCENARIOS")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 80)
        print(f"Parameters: {test_case['params']}")
        
        try:
            result = optimizer.optimize_portfolio(returns, **test_case['params'])
            
            if result['status'] == 'optimal':
                print("✓ Optimization SUCCESSFUL")
                print(f"\nPortfolio Composition:")
                weights = result['weights']
                tickers = result['tickers']
                
                # Show all non-zero weights
                for ticker, weight in zip(tickers, weights):
                    if weight > 0.001:  # Show weights > 0.1%
                        print(f"  {ticker:10s}: {weight*100:6.2f}%")
                
                print(f"\nPortfolio Statistics:")
                stats = result['stats']
                print(f"  Expected Return: {stats['expected_return']*100:6.2f}%")
                print(f"  Volatility:      {stats['volatility']*100:6.2f}%")
                print(f"  Sharpe Ratio:    {stats['sharpe_ratio']:6.3f}")
                print(f"  Positions:       {np.sum(weights > 0.001)}")
                print(f"  Weight sum:      {np.sum(weights):.6f}")
                
                # Check for NaN values
                if any(np.isnan(v) for v in stats.values() if isinstance(v, (int, float))):
                    print("\n⚠️ WARNING: Stats contain NaN values!")
                    
            else:
                print(f"❌ Optimization FAILED: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test with target return
    print("\n" + "=" * 80)
    print("TESTING WITH TARGET RETURN")
    print("=" * 80)
    
    # Calculate reasonable target return
    mean_returns = returns.mean() * 252
    target_return = mean_returns.mean() * 1.2  # 20% above average
    
    print(f"\nTarget return: {target_return*100:.2f}%")
    print("-" * 80)
    
    try:
        result = optimizer.optimize_portfolio(returns, target_return=target_return, max_weight=0.30)
        
        if result['status'] == 'optimal':
            print("✓ Optimization SUCCESSFUL")
            stats = result['stats']
            print(f"  Achieved Return: {stats['expected_return']*100:6.2f}%")
            print(f"  Volatility:      {stats['volatility']*100:6.2f}%")
            print(f"  Sharpe Ratio:    {stats['sharpe_ratio']:6.3f}")
        else:
            print(f"❌ Optimization FAILED: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print("DEBUG TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_optimization()
