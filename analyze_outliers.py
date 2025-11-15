"""
Analyze why individual assets appear outside the efficient frontier
This should not happen in proper MPT - investigate the issue
"""
import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer
import plotly.graph_objects as go

def analyze_frontier_outliers():
    print("🔍 Analyzing Efficient Frontier vs Individual Assets...")
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Get data with NVDA and AVGO
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AVGO', 'KO', 'TSLA', 'JPM']
    print(f"Analyzing: {tickers}")
    
    try:
        # Get data and returns
        data = optimizer.fetch_historical_data(tickers, '2y')
        returns = optimizer.calculate_returns(data)
        
        print(f"✅ Data period: {data.index.min()} to {data.index.max()}")
        
        # Calculate individual asset statistics
        individual_stats = []
        for ticker in tickers:
            if ticker in returns.columns:
                annual_return = returns[ticker].mean() * 252
                annual_vol = returns[ticker].std() * np.sqrt(252)
                sharpe = (annual_return - optimizer.risk_free_rate) / annual_vol
                
                individual_stats.append({
                    'ticker': ticker,
                    'return': annual_return,
                    'volatility': annual_vol,
                    'sharpe': sharpe
                })
        
        # Sort by Sharpe ratio
        individual_stats.sort(key=lambda x: x['sharpe'], reverse=True)
        
        print(f"\n📊 Individual Asset Performance (sorted by Sharpe ratio):")
        for stat in individual_stats:
            print(f"   {stat['ticker']}: Return={stat['return']:.2%}, Vol={stat['volatility']:.2%}, Sharpe={stat['sharpe']:.2f}")
        
        # Generate efficient frontier with detailed logging
        print(f"\n🎯 Generating Efficient Frontier...")
        
        # Use wider range of target returns
        min_return = min([s['return'] for s in individual_stats])
        max_return = max([s['return'] for s in individual_stats])
        target_returns = np.linspace(min_return * 0.8, max_return * 1.2, 50)
        
        efficient_portfolios = []
        failed_optimizations = 0
        
        for i, target_return in enumerate(target_returns):
            try:
                # Try optimization with current constraints
                result = optimizer.optimize_portfolio(returns, target_return=target_return)
                
                if result['status'] == 'optimal':
                    efficient_portfolios.append({
                        'target_return': target_return,
                        'actual_return': result['stats']['return'],
                        'volatility': result['stats']['volatility'],
                        'sharpe_ratio': result['stats']['sharpe_ratio'],
                        'weights': result['weights']
                    })
                else:
                    failed_optimizations += 1
                    
            except Exception as e:
                failed_optimizations += 1
                if i % 10 == 0:  # Print occasional failures
                    print(f"   ⚠️ Failed optimization at return {target_return:.2%}: {str(e)}")
        
        print(f"✅ Generated {len(efficient_portfolios)} frontier points ({failed_optimizations} failed)")
        
        if len(efficient_portfolios) > 0:
            # Find the issue: check if any individual asset dominates
            ef_df = pd.DataFrame(efficient_portfolios)
            
            print(f"\n🔍 Efficient Frontier Analysis:")
            print(f"   Frontier return range: {ef_df['actual_return'].min():.2%} to {ef_df['actual_return'].max():.2%}")
            print(f"   Frontier volatility range: {ef_df['volatility'].min():.2%} to {ef_df['volatility'].max():.2%}")
            print(f"   Best Sharpe on frontier: {ef_df['sharpe_ratio'].max():.2f}")
            
            # Check for assets outside frontier
            print(f"\n❗ Assets that appear 'outside' the frontier:")
            frontier_max_sharpe = ef_df['sharpe_ratio'].max()
            
            for stat in individual_stats:
                if stat['sharpe'] > frontier_max_sharpe:
                    print(f"   🚨 {stat['ticker']}: Sharpe {stat['sharpe']:.2f} > Frontier max {frontier_max_sharpe:.2f}")
                    print(f"      This suggests the frontier calculation has issues!")
                
            # Test: Can we achieve the high-performing individual assets?
            print(f"\n🧪 Testing: Can optimizer achieve individual asset performance?")
            
            for stat in individual_stats[:3]:  # Test top 3 performers
                try:
                    # Try to optimize for this specific return level
                    test_result = optimizer.optimize_portfolio(returns, target_return=stat['return'])
                    
                    if test_result['status'] == 'optimal':
                        achieved_vol = test_result['stats']['volatility']
                        achieved_sharpe = test_result['stats']['sharpe_ratio']
                        
                        print(f"   {stat['ticker']} target: Return={stat['return']:.2%}, Vol={stat['volatility']:.2%}")
                        print(f"   Portfolio achieved: Vol={achieved_vol:.2%}, Sharpe={achieved_sharpe:.2f}")
                        
                        if achieved_vol > stat['volatility']:
                            print(f"   ✅ Portfolio properly has higher volatility than individual asset")
                        else:
                            print(f"   ❌ Issue: Portfolio volatility should be >= individual asset volatility")
                    else:
                        print(f"   ❌ Could not optimize for {stat['ticker']}'s return level")
                        
                except Exception as e:
                    print(f"   ❌ Error testing {stat['ticker']}: {str(e)}")
            
            # Test constraints
            print(f"\n🔧 Testing Portfolio Constraints:")
            max_weight_constraint = 0.20  # Current max weight
            
            for stat in individual_stats:
                if stat['sharpe'] > frontier_max_sharpe:
                    print(f"\n   Testing {stat['ticker']} (high Sharpe: {stat['sharpe']:.2f}):")
                    
                    # What if we put max weight in this asset?
                    test_weights = np.zeros(len(tickers))
                    if stat['ticker'] in tickers:
                        ticker_idx = tickers.index(stat['ticker'])
                        test_weights[ticker_idx] = max_weight_constraint
                        # Distribute remaining weight equally
                        remaining_weight = 1 - max_weight_constraint
                        other_weight = remaining_weight / (len(tickers) - 1)
                        for i in range(len(test_weights)):
                            if i != ticker_idx:
                                test_weights[i] = other_weight
                        
                        # Calculate portfolio stats with this allocation
                        portfolio_return = np.sum(returns.mean() * test_weights) * 252
                        portfolio_vol = np.sqrt(np.dot(test_weights.T, np.dot(returns.cov() * 252, test_weights)))
                        portfolio_sharpe = (portfolio_return - optimizer.risk_free_rate) / portfolio_vol
                        
                        print(f"      Max allocation ({max_weight_constraint:.0%}): Return={portfolio_return:.2%}, Vol={portfolio_vol:.2%}, Sharpe={portfolio_sharpe:.2f}")
                        
                        if portfolio_sharpe > frontier_max_sharpe:
                            print(f"      🚨 This allocation beats the frontier! Constraint may be too restrictive.")
        
        else:
            print(f"❌ No efficient frontier points generated!")
            
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

if __name__ == "__main__":
    analyze_frontier_outliers()
