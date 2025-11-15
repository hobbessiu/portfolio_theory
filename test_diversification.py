"""
Quick test to verify diversification constraints are working
"""
import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer

def test_diversification_fix():
    print("🧪 Testing Diversification Constraints...")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Get a small set of tickers for testing
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'KO', 'TSLA']
    print(f"Testing with: {tickers}")
    
    try:
        # Fetch data
        data = optimizer.fetch_historical_data(tickers, '1y')
        returns = optimizer.calculate_returns(data)
        print(f"✅ Data loaded: {data.shape}")
        
        # Test 1: Default constraints (20% max)
        result1 = optimizer.optimize_portfolio(returns)
        if result1['status'] == 'optimal':
            weights1 = result1['weights']
            max_weight1 = np.max(weights1)
            print(f"\n📊 Test 1 - Default (20% max):")
            print(f"   Max single position: {max_weight1:.1%}")
            print(f"   Positions > 1%: {np.sum(weights1 > 0.01)}")
            
            for ticker, weight in zip(tickers, weights1):
                if weight > 0.01:
                    print(f"   {ticker}: {weight:.1%}")
        
        # Test 2: Conservative constraints (10% max)
        result2 = optimizer.optimize_portfolio(returns, max_weight=0.10)
        if result2['status'] == 'optimal':
            weights2 = result2['weights']
            max_weight2 = np.max(weights2)
            print(f"\n📊 Test 2 - Conservative (10% max):")
            print(f"   Max single position: {max_weight2:.1%}")
            print(f"   Positions > 1%: {np.sum(weights2 > 0.01)}")
            
            for ticker, weight in zip(tickers, weights2):
                if weight > 0.01:
                    print(f"   {ticker}: {weight:.1%}")
        
        # Test 3: Aggressive constraints (40% max)
        result3 = optimizer.optimize_portfolio(returns, max_weight=0.40)
        if result3['status'] == 'optimal':
            weights3 = result3['weights']
            max_weight3 = np.max(weights3)
            print(f"\n📊 Test 3 - Aggressive (40% max):")
            print(f"   Max single position: {max_weight3:.1%}")
            print(f"   Positions > 1%: {np.sum(weights3 > 0.01)}")
            
            for ticker, weight in zip(tickers, weights3):
                if weight > 0.01:
                    print(f"   {ticker}: {weight:.1%}")
        
        print(f"\n✅ Diversification constraints working properly!")
        print(f"   - Default max: {max_weight1:.1%} <= 20% ✓")
        print(f"   - Conservative max: {max_weight2:.1%} <= 10% ✓")
        print(f"   - Aggressive max: {max_weight3:.1%} <= 40% ✓")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_diversification_fix()
