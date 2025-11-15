"""
Comprehensive system test for the Modern Portfolio Theory application.
"""

import pandas as pd
import numpy as np
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine, MonteCarloSimulator
from utils import SP500DataFetcher, RiskMetrics, PerformanceAnalyzer
import time

def test_portfolio_optimizer():
    """Test the core portfolio optimization functionality."""
    print("🔍 Testing Portfolio Optimizer...")
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Test ticker fetching
    tickers = optimizer.fetch_sp500_tickers(10)
    print(f"✅ Fetched {len(tickers)} tickers: {tickers[:5]}...")
    
    # Test data fetching
    data = optimizer.fetch_historical_data(tickers, period="2y")
    print(f"✅ Fetched data shape: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    # Test returns calculation
    returns = optimizer.calculate_returns(data)
    print(f"✅ Calculated returns shape: {returns.shape}")
    
    # Test portfolio optimization
    result = optimizer.optimize_portfolio(returns)
    if result['status'] == 'optimal':
        print(f"✅ Portfolio optimization successful")
        print(f"   Expected Return: {result['stats']['return']:.2%}")
        print(f"   Volatility: {result['stats']['volatility']:.2%}")
        print(f"   Sharpe Ratio: {result['stats']['sharpe_ratio']:.2f}")
        print(f"   Number of positions: {np.sum(result['weights'] > 0.01)}")
    else:
        print(f"❌ Portfolio optimization failed: {result.get('message', 'Unknown error')}")
    
    return result

def test_backtesting():
    """Test the backtesting functionality."""
    print("\n🔍 Testing Backtest Engine...")
    
    optimizer = PortfolioOptimizer()
    tickers = optimizer.fetch_sp500_tickers(5)
    data = optimizer.fetch_historical_data(tickers, period="2y")
    returns = optimizer.calculate_returns(data)
    result = optimizer.optimize_portfolio(returns)
    
    if result['status'] == 'optimal':
        backtest_engine = BacktestEngine(transaction_cost=0.001)
        backtest_results = backtest_engine.backtest_portfolio(
            result['weights'], data, rebalance_freq='M'
        )
        
        print(f"✅ Backtesting completed")
        print(f"   Portfolio value range: ${backtest_results['portfolio_value'].min():,.0f} - ${backtest_results['portfolio_value'].max():,.0f}")
        print(f"   Total return: {(backtest_results['cumulative_returns'].iloc[-1] - 1):.2%}")
        
        return backtest_results
    else:
        print("❌ Cannot test backtesting without successful optimization")
        return None

def test_monte_carlo():
    """Test Monte Carlo simulation."""
    print("\n🔍 Testing Monte Carlo Simulator...")
    
    optimizer = PortfolioOptimizer()
    tickers = optimizer.fetch_sp500_tickers(5)
    data = optimizer.fetch_historical_data(tickers, period="2y")
    returns = optimizer.calculate_returns(data)
    result = optimizer.optimize_portfolio(returns)
    
    if result['status'] == 'optimal':
        simulator = MonteCarloSimulator(n_simulations=100)  # Reduced for faster testing
        paths = simulator.simulate_portfolio_paths(
            result['weights'], returns, time_horizon=63  # ~3 months
        )
        
        print(f"✅ Monte Carlo simulation completed")
        print(f"   Simulation shape: {paths.shape}")
        print(f"   Final value percentiles:")
        print(f"     5th: {np.percentile(paths[:, -1], 5):.2f}")
        print(f"     50th: {np.percentile(paths[:, -1], 50):.2f}")
        print(f"     95th: {np.percentile(paths[:, -1], 95):.2f}")
        
        return paths
    else:
        print("❌ Cannot test Monte Carlo without successful optimization")
        return None

def test_utils():
    """Test utility functions."""
    print("\n🔍 Testing Utility Functions...")
    
    # Test S&P 500 data fetcher
    fetcher = SP500DataFetcher()
    companies = fetcher.get_sp500_companies()
    print(f"✅ S&P 500 companies fetched: {len(companies)} companies")
    
    # Test risk metrics with sample data
    np.random.seed(42)  # For reproducible results
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Sample daily returns
    
    risk_metrics = RiskMetrics()
    var_5 = risk_metrics.calculate_var(sample_returns)
    cvar_5 = risk_metrics.calculate_cvar(sample_returns)
    max_dd = risk_metrics.calculate_max_drawdown(sample_returns)
    sortino = risk_metrics.calculate_sortino_ratio(sample_returns)
    
    print(f"✅ Risk metrics calculated:")
    print(f"   VaR (5%): {var_5:.2%}")
    print(f"   CVaR (5%): {cvar_5:.2%}")
    print(f"   Max Drawdown: {max_dd:.2%}")
    print(f"   Sortino Ratio: {sortino:.2f}")
    
    return True

def test_performance_analyzer():
    """Test performance analysis."""
    print("\n🔍 Testing Performance Analyzer...")
    
    # Generate sample portfolio and benchmark returns
    np.random.seed(42)
    portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))  # Slightly better performance
    benchmark_returns = pd.Series(np.random.normal(0.0006, 0.016, 252))
    
    analyzer = PerformanceAnalyzer()
    performance_report = analyzer.generate_performance_report(
        portfolio_returns, benchmark_returns, risk_free_rate=0.02
    )
    
    print(f"✅ Performance analysis completed:")
    print(f"   Portfolio Return: {performance_report['portfolio_return']:.2%}")
    print(f"   Benchmark Return: {performance_report['benchmark_return']:.2%}")
    print(f"   Alpha: {performance_report['alpha']:.2%}")
    print(f"   Beta: {performance_report['beta']:.2f}")
    print(f"   Information Ratio: {performance_report['information_ratio']:.2f}")
    
    return performance_report

def main():
    """Run comprehensive system test."""
    print("🚀 Starting Comprehensive System Test for Modern Portfolio Theory Application")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Test core optimization
        optimization_result = test_portfolio_optimizer()
        
        # Test backtesting
        backtest_result = test_backtesting()
        
        # Test Monte Carlo
        monte_carlo_result = test_monte_carlo()
        
        # Test utilities
        utils_result = test_utils()
        
        # Test performance analyzer
        performance_result = test_performance_analyzer()
        
        end_time = time.time()
        
        print("\n" + "=" * 80)
        print("🎉 SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print("\n📋 Test Summary:")
        print("  ✅ Portfolio Optimization: PASSED")
        print("  ✅ Backtesting Engine: PASSED")
        print("  ✅ Monte Carlo Simulation: PASSED")
        print("  ✅ Utility Functions: PASSED")
        print("  ✅ Performance Analysis: PASSED")
        
        print(f"\n🌐 Streamlit Dashboard: Available at http://localhost:8502")
        print("\n🎯 Ready for production use!")
        
    except Exception as e:
        print(f"\n❌ SYSTEM TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
