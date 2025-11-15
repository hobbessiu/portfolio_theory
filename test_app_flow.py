"""
Simple test to mimic the app's optimization flow
"""

import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer
import warnings
warnings.filterwarnings('ignore')

def test_app_flow():
    print("=" * 80)
    print("TESTING APP OPTIMIZATION FLOW")
    print("=" * 80)
    
    optimizer = PortfolioOptimizer()
    
    # Test 1: Fetch S&P 500 tickers
    print("\n1. Fetching S&P 500 tickers...")
    tickers = optimizer.fetch_sp500_tickers(top_n=10)
    print(f"✓ Got {len(tickers)} tickers: {tickers}")
    
    # Test 2: Fetch historical data
    print("\n2. Fetching historical data (1y)...")
    data = optimizer.fetch_historical_data(tickers, period='1y')
    
    if data is None or data.empty:
        print("❌ Failed to fetch data")
        return
    
    print(f"✓ Data: {len(data)} days, {len(data.columns)} stocks")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Missing values: {data.isna().sum().sum()}")
    
    # Test 3: Calculate returns
    print("\n3. Calculating returns...")
    returns = optimizer.calculate_returns(data)
    
    if returns is None or returns.empty:
        print("❌ Failed to calculate returns")
        return
    
    print(f"✓ Returns: {len(returns)} days")
    print(f"  Missing before dropna: {returns.isna().sum().sum()}")
    
    returns = returns.dropna()
    print(f"  Missing after dropna: {returns.isna().sum().sum()}")
    print(f"  Days after cleaning: {len(returns)}")
    
    if len(returns) < 30:
        print(f"❌ Insufficient data: {len(returns)} days")
        return
    
    # Test 4: Run optimization with different parameters
    test_params = [
        {"max_weight": 0.20, "min_positions": None, "label": "Default (20% max)"},
        {"max_weight": 0.30, "min_positions": 4, "label": "30% max, 4 min positions"},
        {"max_weight": 0.50, "min_positions": None, "label": "50% max weight"},
        {"max_weight": 1.00, "min_positions": None, "label": "100% max (concentrated)"},
    ]
    
    for params in test_params:
        label = params.pop("label")
        print(f"\n4. Testing optimization: {label}")
        print(f"   Parameters: {params}")
        
        result = optimizer.optimize_portfolio(returns, **params)
        
        if result['status'] == 'optimal':
            print(f"   ✓ SUCCESS")
            
            # Check for NaN in stats
            has_nan = any(np.isnan(v) for v in result['stats'].values() if isinstance(v, (int, float)))
            if has_nan:
                print(f"   ❌ WARNING: Stats contain NaN!")
                print(f"      Stats: {result['stats']}")
            else:
                print(f"      Return: {result['stats']['return']*100:.2f}%")
                print(f"      Volatility: {result['stats']['volatility']*100:.2f}%")
                print(f"      Sharpe: {result['stats']['sharpe_ratio']:.2f}")
                print(f"      Positions: {np.sum(np.array(result['weights']) > 0.001)}")
        else:
            print(f"   ❌ FAILED: {result.get('message', 'Unknown')}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_app_flow()
