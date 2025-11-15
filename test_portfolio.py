"""
Test suite for the Modern Portfolio Theory application.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_optimizer import PortfolioOptimizer, BacktestEngine, MonteCarloSimulator
from utils import RiskMetrics, PerformanceAnalyzer


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_assets = 5
        
        # Generate correlated returns
        mean_returns = np.random.normal(0.0008, 0.002, n_assets)  # Daily returns
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) * 0.0001  # Make positive definite
        
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
        
        self.sample_returns = pd.DataFrame(
            returns,
            index=dates,
            columns=[f'STOCK_{i+1}' for i in range(n_assets)]
        )
        
        # Generate price data
        self.sample_prices = (1 + self.sample_returns).cumprod() * 100
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        returns = self.optimizer.calculate_returns(self.sample_prices)
        
        # Check that returns are calculated correctly
        self.assertEqual(len(returns), len(self.sample_prices) - 1)
        self.assertTrue(all(returns.columns == self.sample_prices.columns))
        
        # Verify calculation
        manual_returns = self.sample_prices.pct_change().dropna()
        pd.testing.assert_frame_equal(returns, manual_returns)
    
    def test_portfolio_stats_calculation(self):
        """Test portfolio statistics calculation."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
        stats = self.optimizer.calculate_portfolio_stats(weights, self.sample_returns)
        
        # Check that all required stats are present
        required_keys = ['return', 'volatility', 'sharpe_ratio']
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))
        
        # Sharpe ratio should be reasonable
        self.assertGreater(stats['sharpe_ratio'], -5)
        self.assertLess(stats['sharpe_ratio'], 5)
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        result = self.optimizer.optimize_portfolio(self.sample_returns)
        
        # Check optimization result
        self.assertEqual(result['status'], 'optimal')
        self.assertIn('weights', result)
        self.assertIn('stats', result)
        
        # Weights should sum to 1 and be non-negative
        weights = result['weights']
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        self.assertTrue(np.all(weights >= -1e-6))  # Allow small numerical errors


class TestRiskMetrics(unittest.TestCase):
    """Test cases for RiskMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        var_5 = RiskMetrics.calculate_var(self.returns, 0.05)
        
        # VaR should be negative (loss)
        self.assertLess(var_5, 0)
        
        # Check that approximately 5% of returns are below VaR
        below_var = np.sum(self.returns <= var_5) / len(self.returns)
        self.assertAlmostEqual(below_var, 0.05, delta=0.02)
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation."""
        var_5 = RiskMetrics.calculate_var(self.returns, 0.05)
        cvar_5 = RiskMetrics.calculate_cvar(self.returns, 0.05)
        
        # CVaR should be more negative than VaR
        self.assertLess(cvar_5, var_5)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create a series with known drawdown
        returns = pd.Series([0.1, -0.2, -0.1, 0.05, -0.05])
        max_dd = RiskMetrics.calculate_max_drawdown(returns)
        
        # Should be negative
        self.assertLess(max_dd, 0)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        sortino = RiskMetrics.calculate_sortino_ratio(self.returns)
        
        # Should be a reasonable number
        self.assertIsInstance(sortino, (int, float))
        self.assertGreater(sortino, -10)
        self.assertLess(sortino, 10)


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backtest_engine = BacktestEngine(transaction_cost=0.001)
        
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        n_assets = 3
        
        returns = np.random.multivariate_normal(
            [0.0005, 0.0008, 0.0006],
            [[0.0001, 0.00005, 0.00003],
             [0.00005, 0.0002, 0.00004],
             [0.00003, 0.00004, 0.00015]],
            len(dates)
        )
        
        self.prices = pd.DataFrame(
            (1 + pd.DataFrame(returns)).cumprod().values * 100,
            index=dates,
            columns=['STOCK_A', 'STOCK_B', 'STOCK_C']
        )
        
        self.weights = np.array([0.4, 0.3, 0.3])
    
    def test_backtest_portfolio(self):
        """Test portfolio backtesting."""
        result = self.backtest_engine.backtest_portfolio(
            self.weights, self.prices, rebalance_freq='M'
        )
        
        # Check result structure
        required_columns = ['returns', 'cumulative_returns', 'portfolio_value']
        for col in required_columns:
            self.assertIn(col, result.columns)
        
        # Check data consistency
        self.assertEqual(len(result), len(self.prices) - 1)  # One less due to returns calculation
        
        # Cumulative returns should start near 1
        self.assertAlmostEqual(result['cumulative_returns'].iloc[0], 1.0, delta=0.1)


class TestMonteCarloSimulator(unittest.TestCase):
    """Test cases for MonteCarloSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = MonteCarloSimulator(n_simulations=100)
        
        # Create sample data
        np.random.seed(42)
        self.returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, (252, 3)),
            columns=['A', 'B', 'C']
        )
        self.weights = np.array([0.4, 0.3, 0.3])
    
    def test_simulate_portfolio_paths(self):
        """Test Monte Carlo simulation."""
        paths = self.simulator.simulate_portfolio_paths(
            self.weights, self.returns, time_horizon=100
        )
        
        # Check dimensions
        self.assertEqual(paths.shape, (100, 101))  # n_simulations x (time_horizon + 1)
        
        # All paths should start at 1
        self.assertTrue(np.allclose(paths[:, 0], 1.0))
        
        # Paths should be positive (assuming reasonable parameters)
        self.assertTrue(np.all(paths > 0))


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample return series
        np.random.seed(42)
        self.portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        self.benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, 252))
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        report = self.analyzer.generate_performance_report(
            self.portfolio_returns, self.benchmark_returns
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'portfolio_return', 'portfolio_volatility', 'benchmark_return',
            'benchmark_volatility', 'excess_return', 'sharpe_ratio',
            'sortino_ratio', 'information_ratio', 'tracking_error',
            'beta', 'alpha', 'var_5', 'cvar_5', 'max_drawdown', 'calmar_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, report)
            self.assertIsInstance(report[metric], (int, float))
        
        # Basic sanity checks
        self.assertGreater(report['portfolio_volatility'], 0)
        self.assertGreater(report['benchmark_volatility'], 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPortfolioOptimizer,
        TestRiskMetrics,
        TestBacktestEngine,
        TestMonteCarloSimulator,
        TestPerformanceAnalyzer
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
