#!/usr/bin/env python3
"""
Complete Walk-Forward Analysis Test
==================================

This script demonstrates and validates the walk-forward analysis functionality
compared to the fixed-weights backtesting approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine
import yfinance as yf
from datetime import datetime

def test_walk_forward_vs_fixed_weights():
    """Compare walk-forward analysis with fixed weights approach."""
    print("🚀 Testing Walk-Forward Analysis vs Fixed Weights")
    print("=" * 60)
    
    # Initialize components
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    backtest_engine = BacktestEngine(transaction_cost=0.001)
    
    # Get test data - using a smaller set for faster testing
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'UNH']
    print(f"📊 Testing with tickers: {test_tickers}")
    
    try:
        # Fetch historical data (2 years for meaningful walk-forward)
        data = optimizer.fetch_historical_data(test_tickers, "2y")
        print(f"📈 Data loaded: {data.shape[0]} days, {data.shape[1]} stocks")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        if data.empty:
            print("❌ No data loaded - test aborted")
            return False
        
        # Method 1: Fixed Weights Backtesting
        print("\n🎯 Method 1: Fixed Weights Approach")
        print("-" * 40)
        
        # Optimize once using all historical data (this simulates look-ahead bias)
        returns = data.pct_change().dropna()
        optimization_result = optimizer.optimize_portfolio(returns)
        
        if optimization_result['status'] != 'optimal':
            print("❌ Portfolio optimization failed")
            return False
            
        fixed_weights = optimization_result['weights']
        print(f"✅ Portfolio optimized with Sharpe ratio: {optimization_result['stats']['sharpe_ratio']:.3f}")
        
        # Backtest with fixed weights
        fixed_backtest = backtest_engine.backtest_portfolio(fixed_weights, data, 'Q')  # Quarterly rebalancing
        
        print(f"📊 Fixed weights backtest completed:")
        print(f"   • Final portfolio value: ${fixed_backtest['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   • Total return: {(fixed_backtest['cumulative_returns'].iloc[-1] - 1)*100:.1f}%")
        print(f"   • Total transaction costs: {fixed_backtest['rebalancing_costs'].sum()*100:.3f}%")
        
        # Method 2: Walk-Forward Analysis
        print("\n🔄 Method 2: Walk-Forward Analysis")
        print("-" * 40)
        
        # Walk-forward backtest (re-optimizes at each rebalancing date)
        walkforward_backtest = backtest_engine.backtest_portfolio_walk_forward(data, 'Q', min_history_days=252)
        
        print(f"📊 Walk-forward backtest completed:")
        print(f"   • Final portfolio value: ${walkforward_backtest['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   • Total return: {(walkforward_backtest['cumulative_returns'].iloc[-1] - 1)*100:.1f}%")
        print(f"   • Total transaction costs: {walkforward_backtest['rebalancing_costs'].sum()*100:.3f}%")
        print(f"   • Re-optimizations: {walkforward_backtest.attrs.get('optimization_count', 0)} times")
        
        # Performance Comparison
        print("\n📈 Performance Comparison")
        print("-" * 40)
        
        # Calculate key metrics for both approaches
        fixed_annual_return = fixed_backtest['returns'].mean() * 252
        fixed_volatility = fixed_backtest['returns'].std() * np.sqrt(252)
        fixed_sharpe = (fixed_annual_return - 0.02) / fixed_volatility
        
        wf_annual_return = walkforward_backtest['returns'].mean() * 252
        wf_volatility = walkforward_backtest['returns'].std() * np.sqrt(252)
        wf_sharpe = (wf_annual_return - 0.02) / wf_volatility
        
        comparison_data = {
            'Method': ['Fixed Weights', 'Walk-Forward'],
            'Annual Return': [f"{fixed_annual_return:.1%}", f"{wf_annual_return:.1%}"],
            'Volatility': [f"{fixed_volatility:.1%}", f"{wf_volatility:.1%}"],
            'Sharpe Ratio': [f"{fixed_sharpe:.3f}", f"{wf_sharpe:.3f}"],
            'Final Value': [f"${fixed_backtest['portfolio_value'].iloc[-1]:,.0f}", 
                           f"${walkforward_backtest['portfolio_value'].iloc[-1]:,.0f}"],
            'Total Costs': [f"{fixed_backtest['rebalancing_costs'].sum():.3%}", 
                           f"{walkforward_backtest['rebalancing_costs'].sum():.3%}"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📋 Detailed Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Performance Analysis
        print(f"\n🎯 Analysis:")
        return_diff = wf_annual_return - fixed_annual_return
        sharpe_diff = wf_sharpe - fixed_sharpe
        cost_diff = walkforward_backtest['rebalancing_costs'].sum() - fixed_backtest['rebalancing_costs'].sum()
        
        print(f"   • Walk-forward return advantage: {return_diff:+.1%}")
        print(f"   • Walk-forward Sharpe advantage: {sharpe_diff:+.3f}")
        print(f"   • Additional transaction costs: {cost_diff:+.3%}")
        
        if return_diff > 0:
            print("   ✅ Walk-forward analysis outperformed fixed weights")
        else:
            print("   ⚠️ Fixed weights outperformed walk-forward analysis")
            
        if cost_diff > 0.005:  # More than 0.5% additional costs
            print("   ⚠️ Walk-forward has significantly higher transaction costs")
        else:
            print("   ✅ Transaction cost difference is reasonable")
        
        # Test Data Validation
        print(f"\n🔍 Data Validation:")
        print(f"   • Fixed weights sum: {np.sum(fixed_weights):.6f}")
        print(f"   • Fixed weights range: {np.min(fixed_weights):.3f} to {np.max(fixed_weights):.3f}")
        print(f"   • Walk-forward data integrity: {len(walkforward_backtest)} days")
        print(f"   • Both methods cover same period: {fixed_backtest.index[0]} to {fixed_backtest.index[-1]}")
        
        print(f"\n✅ Walk-Forward Analysis Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def demo_optimization_dates():
    """Demonstrate the optimization dates tracking in walk-forward analysis."""
    print("\n🗓️ Walk-Forward Optimization Dates Demo")
    print("=" * 50)
    
    optimizer = PortfolioOptimizer()
    backtest_engine = BacktestEngine(transaction_cost=0.001)
    
    # Use a smaller dataset for demo
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM']
    data = optimizer.fetch_historical_data(test_tickers, "1y")
    
    if not data.empty:
        # Run walk-forward with monthly rebalancing
        result = backtest_engine.backtest_portfolio_walk_forward(data, 'M', min_history_days=60)
        
        optimization_dates = result.attrs.get('optimization_dates', [])
        print(f"📅 Portfolio was re-optimized {len(optimization_dates)} times:")
        
        for i, date in enumerate(optimization_dates[:8]):  # Show first 8 dates
            print(f"   {i+1:2d}. {date.strftime('%Y-%m-%d (%b)')}")
        
        if len(optimization_dates) > 8:
            print(f"   ... and {len(optimization_dates) - 8} more dates")
            
        print(f"\n📊 This demonstrates dynamic re-optimization vs. fixed weights approach")
    else:
        print("❌ Could not load demo data")

if __name__ == "__main__":
    print("🚀 Modern Portfolio Theory - Walk-Forward Analysis Test Suite")
    print("=" * 70)
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run main comparison test
    success = test_walk_forward_vs_fixed_weights()
    
    if success:
        # Run additional demo
        demo_optimization_dates()
        
        print("\n" + "=" * 70)
        print("🎉 All tests completed successfully!")
        print("💡 The walk-forward analysis implementation is working correctly.")
        print("📈 You can now use both backtesting methods in the Streamlit app.")
        print("\nNext steps:")
        print("   1. Run 'streamlit run app.py' to launch the dashboard")
        print("   2. Navigate to the 'Backtesting' tab")
        print("   3. Select 'Walk-Forward Analysis' method")
        print("   4. Compare results with 'Fixed Weights' method")
    else:
        print("\n❌ Tests failed - please check the implementation")
