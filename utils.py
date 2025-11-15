"""
Utility functions for data fetching and analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time


class SP500DataFetcher:
    """
    Utility class to fetch S&P 500 constituent data and company information.
    """
    
    def __init__(self):
        self.sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self._sp500_companies = None
    
    def get_sp500_companies(self) -> pd.DataFrame:
        """
        Fetch current S&P 500 companies from Wikipedia.
        
        Returns:
            DataFrame with company information including Symbol, Security, GICS Sector, etc.
        """
        if self._sp500_companies is not None:
            return self._sp500_companies
            
        try:
            # Fetch S&P 500 companies from Wikipedia
            tables = pd.read_html(self.sp500_url)
            sp500_companies = tables[0]  # First table contains the companies
            
            # Clean up column names
            sp500_companies.columns = sp500_companies.columns.str.strip()
            
            # Cache the result
            self._sp500_companies = sp500_companies
            return sp500_companies
            
        except Exception as e:
            print(f"Error fetching S&P 500 companies: {e}")
            # Fallback to hardcoded list
            return self._get_fallback_sp500()
    
    def _get_fallback_sp500(self) -> pd.DataFrame:
        """
        Fallback S&P 500 companies list if web scraping fails.
        """
        fallback_companies = {
            'Symbol': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
                'PFE', 'KO', 'AVGO', 'PEP', 'COST', 'WMT', 'TMO', 'DIS', 'ABT',
                'ACN', 'VZ', 'ADBE', 'CRM', 'DHR', 'NFLX', 'NKE', 'TXN', 'PM',
                'RTX', 'NEE', 'QCOM', 'LIN', 'ORCL', 'WFC', 'IBM', 'AMD', 'UPS',
                'T', 'AMGN', 'ELV', 'LOW', 'BA', 'CAT', 'GS', 'SPGI', 'BLK'
            ],
            'Security': [
                'Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com Inc.',
                'NVIDIA Corporation', 'Meta Platforms Inc.', 'Tesla Inc.', 'Berkshire Hathaway Inc.',
                'UnitedHealth Group Inc.', 'Exxon Mobil Corporation', 'Johnson & Johnson',
                'JPMorgan Chase & Co.', 'Visa Inc.', 'Procter & Gamble Co.', 'Mastercard Inc.',
                'The Home Depot Inc.', 'Chevron Corporation', 'AbbVie Inc.', 'Pfizer Inc.',
                'The Coca-Cola Company', 'Broadcom Inc.', 'PepsiCo Inc.', 'Costco Wholesale Corporation',
                'Walmart Inc.', 'Thermo Fisher Scientific Inc.', 'The Walt Disney Company',
                'Abbott Laboratories', 'Accenture plc', 'Verizon Communications Inc.',
                'Adobe Inc.', 'Salesforce Inc.', 'Danaher Corporation', 'Netflix Inc.',
                'NIKE Inc.', 'Texas Instruments Incorporated', 'Philip Morris International Inc.',
                'Raytheon Technologies Corporation', 'NextEra Energy Inc.', 'QUALCOMM Incorporated',
                'Linde plc', 'Oracle Corporation', 'Wells Fargo & Company',
                'International Business Machines Corporation', 'Advanced Micro Devices Inc.',
                'United Parcel Service Inc.', 'AT&T Inc.', 'Amgen Inc.', 'Elevance Health Inc.',
                'Lowe\'s Companies Inc.', 'The Boeing Company', 'Caterpillar Inc.',
                'The Goldman Sachs Group Inc.', 'S&P Global Inc.', 'BlackRock Inc.'
            ]
        }
        
        return pd.DataFrame(fallback_companies)
    
    def get_market_cap_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get market capitalization data for given symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with symbols and market cap information
        """
        market_caps = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                market_caps.append({
                    'Symbol': symbol,
                    'MarketCap': market_cap,
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown')
                })
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                market_caps.append({
                    'Symbol': symbol,
                    'MarketCap': 0,
                    'Sector': 'Unknown',
                    'Industry': 'Unknown'
                })
        
        return pd.DataFrame(market_caps)


class RiskMetrics:
    """
    Calculate various risk and performance metrics for portfolio analysis.
    """
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (default 5%)
            
        Returns:
            VaR value
        """
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (default 5%)
            
        Returns:
            CVaR value
        """
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Maximum drawdown value
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of portfolio returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_deviation = returns[returns < 0].std() * np.sqrt(252)
        
        if downside_deviation == 0:
            return np.inf
        
        return excess_returns / downside_deviation
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * 252
        max_dd = abs(RiskMetrics.calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return np.inf
            
        return annual_return / max_dd


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis tools.
    """
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
    
    def generate_performance_report(self, portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series,
                                  risk_free_rate: float = 0.02) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with performance metrics
        """
        # Basic metrics
        portfolio_annual_return = portfolio_returns.mean() * 252
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_annual_return = benchmark_returns.mean() * 252
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
        sortino_ratio = self.risk_metrics.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
        
        # Risk metrics
        var_5 = self.risk_metrics.calculate_var(portfolio_returns)
        cvar_5 = self.risk_metrics.calculate_cvar(portfolio_returns)
        max_drawdown = self.risk_metrics.calculate_max_drawdown(portfolio_returns)
        calmar_ratio = self.risk_metrics.calculate_calmar_ratio(portfolio_returns)
        
        # Relative performance
        excess_return = portfolio_annual_return - benchmark_annual_return
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error != 0 else np.inf
        
        # Beta calculation
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        beta = covariance / np.var(benchmark_returns)
        
        # Alpha calculation
        alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        return {
            'portfolio_return': portfolio_annual_return,
            'portfolio_volatility': portfolio_volatility,
            'benchmark_return': benchmark_annual_return,
            'benchmark_volatility': benchmark_volatility,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha,
            'var_5': var_5,
            'cvar_5': cvar_5,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }


if __name__ == "__main__":
    # Test the utilities
    fetcher = SP500DataFetcher()
    companies = fetcher.get_sp500_companies()
    print(f"Fetched {len(companies)} S&P 500 companies")
    print(companies.head())
