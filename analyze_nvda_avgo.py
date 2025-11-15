"""
Analyze why NVDA and AVGO are not appearing in optimized portfolios.
"""

import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer
import matplotlib.pyplot as plt

def analyze_asset_performance():
    """Analyze individual asset performance and optimization behavior."""
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Get limited set of stocks including NVDA and AVGO
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'META', 'TSLA', 'KO', 'PEP', 'JNJ']
    
    print("Fetching data for analysis...")
    prices = optimizer.fetch_data(tickers, period='2y')
    
    if prices is None or prices.empty:
        print("Failed to fetch data")
        return
    
    returns = optimizer.calculate_returns(prices)
    
    # Calculate individual asset statistics
    print("\n=== Individual Asset Analysis ===")
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratios = (annual_returns - optimizer.risk_free_rate) / annual_volatility
    
    # Create performance summary
    performance_df = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratios
    }).round(4)
    
    print(performance_df.sort_values('Sharpe Ratio', ascending=False))
    
    # Analyze correlations
    print("\n=== Correlation Analysis for NVDA and AVGO ===")
    correlation_matrix = returns.corr()
    
    print("NVDA correlations:")
    nvda_corr = correlation_matrix['NVDA'].sort_values(ascending=False)
    print(nvda_corr.round(3))
    
    print("\nAVGO correlations:")
    avgo_corr = correlation_matrix['AVGO'].sort_values(ascending=False)
    print(avgo_corr.round(3))
    
    # Test different optimization scenarios
    print("\n=== Portfolio Optimization Analysis ===")
    
    # Scenario 1: Unconstrained optimization
    print("\n1. Unconstrained Optimization (theoretical efficient frontier):")
    unconstrained_result = optimizer.optimize_portfolio_unconstrained(returns)
    if unconstrained_result['status'] == 'optimal':
        weights_df = pd.DataFrame({
            'Ticker': unconstrained_result['tickers'],
            'Weight': unconstrained_result['weights']
        })
        weights_df = weights_df[weights_df['Weight'] > 0.001]  # Show only significant weights
        weights_df = weights_df.sort_values('Weight', ascending=False)
        print(weights_df.round(4))
        print(f"Portfolio Stats: Return={unconstrained_result['stats']['return']:.4f}, "
              f"Volatility={unconstrained_result['stats']['volatility']:.4f}, "
              f"Sharpe={unconstrained_result['stats']['sharpe_ratio']:.4f}")
    
    # Scenario 2: Standard diversified optimization (20% max)
    print("\n2. Diversified Optimization (20% max per asset):")
    diversified_result = optimizer.optimize_portfolio(returns, max_weight=0.20)
    if diversified_result['status'] == 'optimal':
        weights_df = pd.DataFrame({
            'Ticker': diversified_result['tickers'],
            'Weight': diversified_result['weights']
        })
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        weights_df = weights_df.sort_values('Weight', ascending=False)
        print(weights_df.round(4))
        print(f"Portfolio Stats: Return={diversified_result['stats']['return']:.4f}, "
              f"Volatility={diversified_result['stats']['volatility']:.4f}, "
              f"Sharpe={diversified_result['stats']['sharpe_ratio']:.4f}")
    
    # Scenario 3: More relaxed diversification (40% max)
    print("\n3. Relaxed Diversification (40% max per asset):")
    relaxed_result = optimizer.optimize_portfolio(returns, max_weight=0.40)
    if relaxed_result['status'] == 'optimal':
        weights_df = pd.DataFrame({
            'Ticker': relaxed_result['tickers'],
            'Weight': relaxed_result['weights']
        })
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        weights_df = weights_df.sort_values('Weight', ascending=False)
        print(weights_df.round(4))
        print(f"Portfolio Stats: Return={relaxed_result['stats']['return']:.4f}, "
              f"Volatility={relaxed_result['stats']['volatility']:.4f}, "
              f"Sharpe={relaxed_result['stats']['sharpe_ratio']:.4f}")
    
    # Scenario 4: Force include NVDA and AVGO
    print("\n4. Analysis with minimum allocation to NVDA and AVGO:")
    forced_result = optimizer.optimize_portfolio(returns, max_weight=0.20, min_weight=0.0)
    
    # Create a version where we force NVDA and AVGO to have at least 5%
    nvda_idx = list(returns.columns).index('NVDA')
    avgo_idx = list(returns.columns).index('AVGO')
    
    # Test what happens if we manually set minimum weights for these assets
    import cvxpy as cp
    
    n_assets = len(returns.columns)
    weights = cp.Variable(n_assets)
    
    mu = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    
    portfolio_return = mu.T @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix)
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= 0.20,
        weights[nvda_idx] >= 0.05,  # Force 5% minimum in NVDA
        weights[avgo_idx] >= 0.05   # Force 5% minimum in AVGO
    ]
    
    objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status == 'optimal':
        forced_weights = weights.value
        forced_stats = optimizer.calculate_portfolio_stats(forced_weights, returns)
        
        weights_df = pd.DataFrame({
            'Ticker': returns.columns,
            'Weight': forced_weights
        })
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        weights_df = weights_df.sort_values('Weight', ascending=False)
        print(weights_df.round(4))
        print(f"Forced Portfolio Stats: Return={forced_stats['return']:.4f}, "
              f"Volatility={forced_stats['volatility']:.4f}, "
              f"Sharpe={forced_stats['sharpe_ratio']:.4f}")
        
        # Compare to diversified portfolio
        print(f"\nComparison:")
        print(f"Diversified Sharpe: {diversified_result['stats']['sharpe_ratio']:.4f}")
        print(f"Forced NVDA/AVGO Sharpe: {forced_stats['sharpe_ratio']:.4f}")
        print(f"Difference: {forced_stats['sharpe_ratio'] - diversified_result['stats']['sharpe_ratio']:.4f}")
    
    # Risk contribution analysis
    print("\n=== Risk Contribution Analysis ===")
    if diversified_result['status'] == 'optimal':
        weights = diversified_result['weights']
        cov_matrix = returns.cov().values * 252
        
        # Calculate marginal risk contributions
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_risk = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
        risk_contributions = weights * marginal_risk
        
        risk_df = pd.DataFrame({
            'Ticker': returns.columns,
            'Weight': weights,
            'Risk Contribution': risk_contributions / np.sum(risk_contributions),
            'Risk per Unit Weight': marginal_risk
        })
        
        print("Top risk contributors:")
        print(risk_df.sort_values('Risk Contribution', ascending=False).head(10).round(4))

if __name__ == "__main__":
    analyze_asset_performance()
