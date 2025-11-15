"""
Configuration settings for the Modern Portfolio Theory application.
"""

import os
from typing import Dict, List

class Config:
    """
    Application configuration settings.
    """
    
    # Data settings
    DEFAULT_TOP_N_STOCKS = 30
    MIN_STOCKS = 10
    MAX_STOCKS = 100
    
    DEFAULT_TIME_PERIOD = "5y"
    AVAILABLE_PERIODS = ["1y", "2y", "3y", "5y", "10y", "max"]
    
    # Financial settings
    DEFAULT_RISK_FREE_RATE = 0.02  # 2%
    DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
    
    # Optimization settings
    MIN_WEIGHT = 0.0  # Long-only constraint
    MAX_WEIGHT = 1.0
    WEIGHT_SUM = 1.0
    
    # Monte Carlo settings
    DEFAULT_SIMULATIONS = 1000
    MIN_SIMULATIONS = 500
    MAX_SIMULATIONS = 5000
    
    DEFAULT_TIME_HORIZON = 252  # 1 year in trading days
    MIN_TIME_HORIZON = 30
    MAX_TIME_HORIZON = 1000
    
    # Rebalancing frequencies
    REBALANCE_FREQUENCIES = {
        "Monthly": "M",
        "Quarterly": "Q", 
        "Semi-Annual": "6M",
        "Annual": "Y"
    }
    
    # UI settings
    CHART_HEIGHT = 500
    CHART_WIDTH = 800
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8'
    }
    
    # API settings
    YAHOO_FINANCE_DELAY = 0.1  # Delay between API calls
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Caching settings
    CACHE_TTL = 3600  # 1 hour in seconds
    
    # Display settings
    DECIMAL_PLACES = {
        'percentage': 1,
        'ratio': 2,
        'currency': 2,
        'weight': 3
    }
    
    @classmethod
    def get_display_format(cls, metric_type: str) -> str:
        """
        Get display format string for different metric types.
        
        Args:
            metric_type: Type of metric ('percentage', 'ratio', 'currency', 'weight')
            
        Returns:
            Format string
        """
        decimals = cls.DECIMAL_PLACES.get(metric_type, 2)
        
        if metric_type == 'percentage':
            return f"{{:.{decimals}%}}"
        elif metric_type == 'currency':
            return f"${{:,.{decimals}f}}"
        else:
            return f"{{:.{decimals}f}}"


class FinancialConstants:
    """
    Financial constants and benchmarks.
    """
    
    # Market constants
    TRADING_DAYS_PER_YEAR = 252
    MONTHS_PER_YEAR = 12
    WEEKS_PER_YEAR = 52
    
    # Risk metrics thresholds
    SHARPE_RATIO_EXCELLENT = 2.0
    SHARPE_RATIO_GOOD = 1.0
    SHARPE_RATIO_ACCEPTABLE = 0.5
    
    # Volatility categories
    LOW_VOLATILITY = 0.10  # 10%
    MEDIUM_VOLATILITY = 0.20  # 20%
    HIGH_VOLATILITY = 0.30  # 30%
    
    # Market cap categories (in billions)
    LARGE_CAP = 10_000_000_000  # $10B
    MID_CAP = 2_000_000_000     # $2B
    SMALL_CAP = 300_000_000     # $300M
    
    @classmethod
    def categorize_sharpe_ratio(cls, sharpe_ratio: float) -> str:
        """
        Categorize Sharpe ratio performance.
        
        Args:
            sharpe_ratio: Sharpe ratio value
            
        Returns:
            Performance category string
        """
        if sharpe_ratio >= cls.SHARPE_RATIO_EXCELLENT:
            return "Excellent"
        elif sharpe_ratio >= cls.SHARPE_RATIO_GOOD:
            return "Good"
        elif sharpe_ratio >= cls.SHARPE_RATIO_ACCEPTABLE:
            return "Acceptable"
        else:
            return "Poor"
    
    @classmethod
    def categorize_volatility(cls, volatility: float) -> str:
        """
        Categorize volatility level.
        
        Args:
            volatility: Annual volatility (standard deviation)
            
        Returns:
            Volatility category string
        """
        if volatility <= cls.LOW_VOLATILITY:
            return "Low"
        elif volatility <= cls.MEDIUM_VOLATILITY:
            return "Medium"
        elif volatility <= cls.HIGH_VOLATILITY:
            return "High"
        else:
            return "Very High"
    
    @classmethod
    def categorize_market_cap(cls, market_cap: float) -> str:
        """
        Categorize market capitalization.
        
        Args:
            market_cap: Market capitalization in dollars
            
        Returns:
            Market cap category string
        """
        if market_cap >= cls.LARGE_CAP:
            return "Large Cap"
        elif market_cap >= cls.MID_CAP:
            return "Mid Cap"
        elif market_cap >= cls.SMALL_CAP:
            return "Small Cap"
        else:
            return "Micro Cap"


# Environment-specific settings
class DevConfig(Config):
    """Development configuration."""
    DEBUG = True
    CACHE_TTL = 300  # 5 minutes for development


class ProdConfig(Config):
    """Production configuration."""
    DEBUG = False
    CACHE_TTL = 3600  # 1 hour for production


# Select configuration based on environment
ENV = os.getenv('ENVIRONMENT', 'development').lower()

if ENV == 'production':
    config = ProdConfig()
else:
    config = DevConfig()


# Export commonly used values
DEFAULT_CONFIG = {
    'top_n_stocks': config.DEFAULT_TOP_N_STOCKS,
    'time_period': config.DEFAULT_TIME_PERIOD,
    'risk_free_rate': config.DEFAULT_RISK_FREE_RATE,
    'transaction_cost': config.DEFAULT_TRANSACTION_COST,
    'n_simulations': config.DEFAULT_SIMULATIONS,
    'time_horizon': config.DEFAULT_TIME_HORIZON
}
