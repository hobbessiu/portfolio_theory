"""
Test script to verify efficient frontier functionality
"""
import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer
import plotly.graph_objects as go

def test_efficient_frontier():
    print("🧪 Testing Efficient Frontier Generation...")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Get test tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'KO', 'TSLA']
    print(f"Testing with: {tickers}")
    
    try:
        # Fetch data and calculate returns
        data = optimizer.fetch_historical_data(tickers, '1y')
        returns = optimizer.calculate_returns(data)
        print(f"✅ Data loaded: {data.shape}")
        
        # Generate efficient frontier points
        target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, 20)
        efficient_portfolios = []
        
        print(f"\n📊 Generating efficient frontier with {len(target_returns)} points...")
        
        for i, target_return in enumerate(target_returns):
            try:
                result = optimizer.optimize_portfolio(returns, target_return=target_return)
                if result['status'] == 'optimal':
                    efficient_portfolios.append({
                        'return': result['stats']['return'],
                        'volatility': result['stats']['volatility'],
                        'sharpe_ratio': result['stats']['sharpe_ratio']
                    })
                    if i % 5 == 0:  # Print every 5th point
                        print(f"   Point {i+1}: Return={result['stats']['return']:.2%}, Risk={result['stats']['volatility']:.2%}, Sharpe={result['stats']['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"   ⚠️ Failed at target return {target_return:.2%}: {str(e)}")
                continue
        
        print(f"\n✅ Successfully generated {len(efficient_portfolios)} efficient frontier points")
        
        if len(efficient_portfolios) > 5:
            ef_df = pd.DataFrame(efficient_portfolios)
            print(f"\n📈 Efficient Frontier Statistics:")
            print(f"   Return range: {ef_df['return'].min():.2%} to {ef_df['return'].max():.2%}")
            print(f"   Risk range: {ef_df['volatility'].min():.2%} to {ef_df['volatility'].max():.2%}")
            print(f"   Best Sharpe: {ef_df['sharpe_ratio'].max():.2f}")
            
            # Find optimal portfolio
            max_sharpe_idx = ef_df['sharpe_ratio'].idxmax()
            optimal = ef_df.loc[max_sharpe_idx]
            print(f"\n🎯 Optimal Portfolio (Max Sharpe):")
            print(f"   Expected Return: {optimal['return']:.2%}")
            print(f"   Volatility: {optimal['volatility']:.2%}")
            print(f"   Sharpe Ratio: {optimal['sharpe_ratio']:.2f}")
            
            print(f"\n✅ Efficient frontier generation successful!")
            
        else:
            print(f"❌ Insufficient efficient frontier points generated ({len(efficient_portfolios)})")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_efficient_frontier()
