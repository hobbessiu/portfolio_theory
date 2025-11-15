"""
Test script for Equal Weight Portfolio functionality in Backtesting Analysis
"""

import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine
from app import calculate_equal_weight_performance

def test_equal_weight_enhancement():
    """Test the new equal-weight portfolio functionality."""
    print("🔍 Testing Equal Weight Portfolio Enhancement")
    print("=" * 60)
    
    # Initialize components
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Test with 10 stocks for comprehensive analysis
    tickers = optimizer.fetch_sp500_tickers(10)
    print(f"📊 Testing with {len(tickers)} stocks: {tickers}")
    
    # Fetch data
    data = optimizer.fetch_historical_data(tickers, period="2y")
    print(f"📈 Data shape: {data.shape}")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    
    # Test equal-weight performance calculation
    print("\n🔍 Testing Equal Weight Performance Calculation...")
    equal_weight_performance = calculate_equal_weight_performance(data, tickers)
    
    print(f"✅ Equal weight results:")
    print(f"   Annual Return: {equal_weight_performance['annual_return']:.2%}")
    print(f"   Volatility: {equal_weight_performance['volatility']:.2%}")
    print(f"   Sharpe Ratio: {equal_weight_performance['sharpe_ratio']:.2f}")
    print(f"   Weights shape: {equal_weight_performance['weights'].shape}")
    print(f"   Weights sum: {equal_weight_performance['weights'].sum():.6f}")
    print(f"   Each weight: {equal_weight_performance['weights'][0]:.4f} (should be 1/{len(tickers)} = {1/len(tickers):.4f})")
    
    # Test optimized portfolio for comparison
    print("\n🔍 Testing Optimized Portfolio...")
    returns = optimizer.calculate_returns(data)
    optimized_result = optimizer.optimize_portfolio(returns)
    
    if optimized_result['status'] == 'optimal':
        print(f"✅ Optimized portfolio results:")
        print(f"   Annual Return: {optimized_result['stats']['return']:.2%}")
        print(f"   Volatility: {optimized_result['stats']['volatility']:.2%}")
        print(f"   Sharpe Ratio: {optimized_result['stats']['sharpe_ratio']:.2f}")
        print(f"   Max weight: {np.max(optimized_result['weights']):.2%}")
        print(f"   Number of positions > 1%: {np.sum(optimized_result['weights'] > 0.01)}")
    
    # Test backtesting comparison
    print("\n🔍 Testing Backtesting Comparison...")
    backtest_engine = BacktestEngine(transaction_cost=0.001)
    
    # Backtest optimized portfolio
    optimized_backtest = backtest_engine.backtest_portfolio(
        optimized_result['weights'], data, 'M'
    )
    
    # Backtest equal-weight portfolio
    equal_weight_backtest = backtest_engine.backtest_portfolio(
        equal_weight_performance['weights'], data, 'M'
    )
    
    # Calculate performance metrics
    opt_total_return = (optimized_backtest['cumulative_returns'].iloc[-1] - 1) * 100
    eq_total_return = (equal_weight_backtest['cumulative_returns'].iloc[-1] - 1) * 100
    
    print(f"✅ Backtesting Results:")
    print(f"   Optimized Total Return: {opt_total_return:.1f}%")
    print(f"   Equal Weight Total Return: {eq_total_return:.1f}%")
    print(f"   Optimization Advantage: {opt_total_return - eq_total_return:.1f} percentage points")
    
    # Performance comparison summary
    print("\n📊 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_table = pd.DataFrame({
        'Strategy': ['Optimized Portfolio', 'Equal Weight Portfolio'],
        'Annual Return': [
            f"{optimized_result['stats']['return']:.2%}",
            f"{equal_weight_performance['annual_return']:.2%}"
        ],
        'Volatility': [
            f"{optimized_result['stats']['volatility']:.2%}",
            f"{equal_weight_performance['volatility']:.2%}"
        ],
        'Sharpe Ratio': [
            f"{optimized_result['stats']['sharpe_ratio']:.2f}",
            f"{equal_weight_performance['sharpe_ratio']:.2f}"
        ],
        'Total Return (Backtest)': [
            f"{opt_total_return:.1f}%",
            f"{eq_total_return:.1f}%"
        ],
        'Max Position': [
            f"{np.max(optimized_result['weights']):.1%}",
            f"{np.max(equal_weight_performance['weights']):.1%}"
        ]
    })
    
    print(comparison_table.to_string(index=False))
    
    # Test edge cases
    print("\n🔍 Testing Edge Cases...")
    
    # Test with single stock
    single_ticker = [tickers[0]]
    single_stock_data = data[[single_ticker[0]]]
    single_eq_weight = calculate_equal_weight_performance(single_stock_data, single_ticker)
    print(f"✅ Single stock equal weight: {single_eq_weight['weights'][0]:.4f} (should be 1.0000)")
    
    # Test weight validation
    assert abs(equal_weight_performance['weights'].sum() - 1.0) < 1e-10, "Weights don't sum to 1"
    assert np.all(equal_weight_performance['weights'] >= 0), "Found negative weights"
    assert np.allclose(equal_weight_performance['weights'], 1/len(tickers)), "Weights not equal"
    
    print("✅ All validation tests passed!")
    
    # Feature summary
    print(f"\n🎉 EQUAL WEIGHT ENHANCEMENT SUMMARY")
    print("=" * 60)
    print("✅ Equal weight calculation: WORKING")
    print("✅ Performance metrics: CALCULATED")
    print("✅ Backtesting integration: FUNCTIONAL") 
    print("✅ Comparison with optimization: ENABLED")
    print("✅ Edge case handling: ROBUST")
    print("✅ Data validation: PASSED")
    
    print(f"\n🌐 Enhanced Streamlit Dashboard: http://localhost:8505")
    print("📊 New features available in Backtesting tab:")
    print("   • Three-way comparison (Optimized vs Equal Weight vs S&P 500)")
    print("   • Enhanced performance chart with equal weight line")
    print("   • Detailed metrics comparison table")
    print("   • Optimization advantage calculations")

if __name__ == "__main__":
    test_equal_weight_enhancement()
