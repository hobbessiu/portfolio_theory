"""
Simple analysis script to understand NVDA/AVGO allocation using existing modules.
"""

import sys
import os
sys.path.append('.')

from portfolio_optimizer import PortfolioOptimizer

def simple_analysis():
    """Simple analysis of NVDA/AVGO performance."""
    optimizer = PortfolioOptimizer()
    
    # Use a small subset for analysis
    tickers = ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'KO', 'JNJ', 'META']
    
    print("Analyzing NVDA and AVGO portfolio selection...")
    print(f"Tickers: {tickers}")
    
    # Fetch data
    try:
        prices = optimizer.fetch_data(tickers, period='1y')
        if prices is None:
            print("Failed to fetch data")
            return
            
        returns = optimizer.calculate_returns(prices)
        print(f"Returns shape: {returns.shape}")
        print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        
        # Calculate basic stats
        annual_returns = returns.mean() * 252
        annual_vol = returns.std() * (252**0.5)
        sharpe = (annual_returns - 0.02) / annual_vol
        
        print("\nIndividual Asset Performance:")
        for ticker in tickers:
            print(f"{ticker}: Return={annual_returns[ticker]:.4f}, Vol={annual_vol[ticker]:.4f}, Sharpe={sharpe[ticker]:.4f}")
        
        # Try optimization
        result = optimizer.optimize_portfolio(returns, max_weight=0.20)
        if result['status'] == 'optimal':
            print("\nOptimal Portfolio (20% max constraint):")
            for i, ticker in enumerate(result['tickers']):
                weight = result['weights'][i]
                if weight > 0.001:
                    print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
            
            stats = result['stats']
            print(f"\nPortfolio Stats:")
            print(f"Return: {stats['return']:.4f}")
            print(f"Volatility: {stats['volatility']:.4f}")
            print(f"Sharpe: {stats['sharpe_ratio']:.4f}")
        
        # Try unconstrained
        unconstrained = optimizer.optimize_portfolio_unconstrained(returns)
        if unconstrained['status'] == 'optimal':
            print("\nUnconstrained Portfolio (theoretical):")
            for i, ticker in enumerate(unconstrained['tickers']):
                weight = unconstrained['weights'][i]
                if weight > 0.001:
                    print(f"{ticker}: {weight:.4f} ({weight*100:.2f}%)")
            
            stats = unconstrained['stats']
            print(f"\nUnconstrained Stats:")
            print(f"Return: {stats['return']:.4f}")
            print(f"Volatility: {stats['volatility']:.4f}")
            print(f"Sharpe: {stats['sharpe_ratio']:.4f}")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_analysis()
