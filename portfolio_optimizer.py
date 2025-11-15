"""
Modern Portfolio Theory Application
==================================

This application implements Modern Portfolio Theory (MPT) for S&P 500 stocks optimization
with interactive visualizations, backtesting, and Monte Carlo simulation.

Author: Portfolio Theory Team
Date: November 2025
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
import cvxpy as cp
import math

class PortfolioOptimizer:
    """
    Core portfolio optimization class implementing Modern Portfolio Theory.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.tickers = []
        
    def fetch_sp500_tickers(self, top_n: int = 50) -> List[str]:
        """
        Fetch top N S&P 500 tickers by index weight using multi-source strategy.
        
        Priority order:
        1. SlickCharts.com (has actual S&P 500 weights)
        2. Wikipedia + yfinance market cap (proxy for weight)
        3. GitHub CSV (reliable backup)
        4. Hardcoded list (last resort)
        
        Args:
            top_n: Number of top stocks to include
            
        Returns:
            List of ticker symbols ordered by index weight (descending)
        """
        # Strategy 1: Try SlickCharts for actual S&P 500 weights
        try:
            import urllib.request
            
            url = 'https://www.slickcharts.com/sp500'
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req) as response:
                tables = pd.read_html(response.read())
            
            # SlickCharts typically has the weight table as first table
            if len(tables) > 0:
                sp500_table = tables[0]
                
                # Look for Symbol/Ticker and Weight columns
                symbol_col = None
                weight_col = None
                
                for col in sp500_table.columns:
                    col_lower = str(col).lower()
                    if 'symbol' in col_lower or 'ticker' in col_lower:
                        symbol_col = col
                    if 'weight' in col_lower or 'portfolio' in col_lower:
                        weight_col = col
                
                if symbol_col is not None:
                    # Get tickers
                    all_tickers = sp500_table[symbol_col].tolist()
                    
                    # Clean up tickers
                    all_tickers = [
                        str(ticker).replace('.', '-').strip()
                        for ticker in all_tickers 
                        if pd.notna(ticker) and str(ticker).strip() and str(ticker).lower() != 'nan'
                    ]
                    
                    # If we have weights, ensure proper ordering
                    if weight_col is not None:
                        weights = sp500_table[weight_col].tolist()
                        ticker_weight_pairs = [(t, w) for t, w in zip(all_tickers, weights) if t]
                        ticker_weight_pairs.sort(key=lambda x: float(str(x[1]).replace('%', '')) if pd.notna(x[1]) else 0, reverse=True)
                        all_tickers = [t for t, w in ticker_weight_pairs]
                        print(f"Fetched {len(all_tickers)} tickers from SlickCharts (sorted by weight)")
                    else:
                        print(f"Fetched {len(all_tickers)} tickers from SlickCharts")
                    
                    return all_tickers[:top_n]
        except Exception as e:
            print(f"SlickCharts fetch failed: {e}")
        
        # Strategy 2: Try Wikipedia with market cap sorting (proxy for S&P 500 weight)
        try:
            import urllib.request
            
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req) as response:
                tables = pd.read_html(response.read())
            
            # Find the S&P 500 table
            sp500_table = None
            for table in tables:
                if 'Symbol' in table.columns:
                    sp500_table = table
                    break
            
            if sp500_table is None:
                raise ValueError("Could not find S&P 500 table with 'Symbol' column")
            
            # Get tickers
            all_tickers = sp500_table['Symbol'].tolist()
            
            # Try to get market cap data for weight proxy sorting
            if len(all_tickers) > 0:
                try:
                    # Fetch market cap data using yfinance (market cap is proxy for S&P 500 weight)
                    print(f"Fetching market cap data for weight-based sorting...")
                    tickers_obj = yf.Tickers(' '.join(all_tickers[:100]))  # Limit to first 100 for speed
                    
                    market_caps = {}
                    for ticker in all_tickers[:100]:
                        try:
                            info = tickers_obj.tickers[ticker].info
                            market_cap = info.get('marketCap', 0)
                            if market_cap and market_cap > 0:
                                market_caps[ticker] = market_cap
                        except:
                            pass
                    
                    # Sort by market cap (proxy for weight) if we got data
                    if len(market_caps) > 10:
                        sorted_tickers = sorted(market_caps.keys(), key=lambda x: market_caps[x], reverse=True)
                        # Add remaining tickers that we didn't fetch market cap for
                        remaining = [t for t in all_tickers if t not in sorted_tickers]
                        all_tickers = sorted_tickers + remaining
                        print(f"Sorted {len(market_caps)} stocks by market cap (weight proxy)")
                except Exception as e:
                    print(f"Could not fetch market cap data: {e}. Using Wikipedia order.")
            
            # Clean up tickers
            all_tickers = [
                str(ticker).replace('.', '-') 
                for ticker in all_tickers 
                if pd.notna(ticker) and str(ticker).strip() and str(ticker).lower() != 'nan'
            ]
            
            print(f"Fetched {len(all_tickers)} tickers from Wikipedia")
            return all_tickers[:top_n]
            
        except Exception as e:
            print(f"Wikipedia fetch failed: {e}")
        
        # Strategy 2: Try GitHub CSV
        try:
            print("Trying GitHub CSV backup...")
            github_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
            sp500_df = pd.read_csv(github_url)
            
            # Get tickers from Symbol column
            if 'Symbol' in sp500_df.columns:
                all_tickers = sp500_df['Symbol'].tolist()
            else:
                all_tickers = sp500_df.iloc[:, 0].tolist()
            
            # Clean up
            all_tickers = [
                str(ticker).replace('.', '-') 
                for ticker in all_tickers 
                if pd.notna(ticker) and str(ticker).strip()
            ]
            
            print(f"Fetched {len(all_tickers)} tickers from GitHub CSV")
            return all_tickers[:top_n]
            
        except Exception as e:
            print(f"GitHub CSV fetch failed: {e}")
        
        # Strategy 3: Hardcoded fallback (ordered by approximate market cap as of 2024)
        print("Using hardcoded list as last resort")
        top_sp500 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV',
            'PFE', 'KO', 'AVGO', 'PEP', 'COST', 'WMT', 'TMO', 'DIS', 'ABT',
            'ACN', 'VZ', 'ADBE', 'CRM', 'DHR', 'NFLX', 'NKE', 'TXN', 'PM',
            'RTX', 'NEE', 'QCOM', 'LIN', 'ORCL', 'WFC', 'IBM', 'AMD', 'UPS',
            'T', 'AMGN', 'ELV', 'LOW', 'BA'
        ]
        return top_sp500[:top_n]
    
    def fetch_historical_data(self, tickers: List[str], period: str = "5y") -> pd.DataFrame:
        """
        Fetch historical price data for given tickers.
        
        Args:
            tickers: List of stock symbols
            period: Time period for data (1y, 2y, 5y, 10y, max)
            
        Returns:
            DataFrame with adjusted closing prices
        """
        try:
            # Download data with explicit auto_adjust setting
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            
            # Handle single ticker vs multiple tickers
            if len(tickers) == 1:
                # For single ticker, yfinance returns a DataFrame with column names as metrics
                if 'Adj Close' in data.columns:
                    adj_close_data = data[['Adj Close']].copy()
                    adj_close_data.columns = tickers  # Rename column to ticker name
                else:
                    # Fallback to Close if Adj Close not available
                    adj_close_data = data[['Close']].copy()
                    adj_close_data.columns = tickers
            else:
                # For multiple tickers, yfinance returns MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    adj_close_data = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
                else:
                    # Fallback handling
                    adj_close_data = data
            
            return adj_close_data.dropna()
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with daily returns
        """
        return prices.pct_change().dropna()
    
    def calculate_portfolio_stats(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict:
        """
        Calculate portfolio statistics given weights and returns.
        
        Args:        weights: Portfolio weights
            returns: Returns data
            
        Returns:
            Dictionary with portfolio statistics
        """
        # Validate inputs
        if np.any(np.isnan(weights)):
            return {
                'return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            }
        
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Handle division by zero
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio_unconstrained(self, returns: pd.DataFrame, target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio using Modern Portfolio Theory WITHOUT diversification constraints.
        This generates the theoretical efficient frontier for educational purposes.
        
        Args:
            returns: Historical returns data
            target_return: Target return (if None, optimizes for max Sharpe ratio)
            
        Returns:
            Dictionary with optimal weights and statistics
        """
        n_assets = len(returns.columns)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Expected returns and covariance matrix
        mu = returns.mean().values * 252  # Annualized
        cov_matrix = returns.cov().values * 252  # Annualized
        
        # Portfolio return and risk
        portfolio_return = mu.T @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Constraints - ONLY basic constraints (no diversification limits)
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only constraint
        ]
        
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk)
        else:
            # Maximize Sharpe ratio (approximate)
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            optimal_weights = weights.value
            stats = self.calculate_portfolio_stats(optimal_weights, returns)
            
            return {
                'weights': optimal_weights,
                'tickers': returns.columns.tolist(),
                'stats': stats,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': f"Optimization failed: {problem.status}"}

    def optimize_portfolio(self, returns: pd.DataFrame, target_return: Optional[float] = None, 
                          max_weight: float = 0.20, min_weight: float = 0.0, 
                          min_positions: Optional[int] = None) -> Dict:
        """
        Optimize portfolio using Modern Portfolio Theory.
        
        Args:
            returns: Historical returns data
            target_return: Target return (if None, optimizes for max Sharpe ratio)
            max_weight: Maximum weight for any single asset (default 20% for diversification)
            min_weight: Minimum weight for any single asset (default 0%)
            min_positions: Minimum number of positions to hold (default None)
            
        Returns:
            Dictionary with optimal weights and statistics
        """
        n_assets = len(returns.columns)
        
        # Check constraint feasibility
        if min_positions is not None and min_positions > 0:
            # First check: min_positions cannot exceed available assets
            if min_positions > n_assets:
                return {
                    'status': 'failed',
                    'message': f'Infeasible constraints: Minimum positions ({min_positions}) exceeds available assets ({n_assets})'
                }
            
            # Second check: With max_weight < 1.0, we need a MINIMUM number of positions
            # For example, if max_weight = 0.85, we need at least ceil(1/0.85) = 2 positions
            if max_weight < 1.0:
                min_required_positions = math.ceil(1.0 / max_weight)
                if min_positions < min_required_positions:
                    return {
                        'status': 'failed',
                        'message': f'Infeasible constraints: With max weight of {max_weight*100:.0f}%, you need at least {min_required_positions} positions. Increase minimum positions or increase max weight.'
                    }
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Expected returns and covariance matrix
        mu = returns.mean().values * 252  # Annualized
        cov_matrix = returns.cov().values * 252  # Annualized
        
        # Portfolio return and risk
        portfolio_return = mu.T @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Non-negative weights (long-only)
            weights <= max_weight  # Maximum weight constraint for diversification
        ]
        
        # Note: Minimum positions constraint with mixed-integer programming requires
        # specialized solvers (GLPK, GUROBI, MOSEK, CBC). For better compatibility,
        # we enforce minimum positions through post-processing instead.
        
        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk)
        else:
            # Maximize Sharpe ratio (approximate)
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            optimal_weights = weights.value
            
            # Safety check for NaN or invalid weights
            if np.any(np.isnan(optimal_weights)) or np.sum(optimal_weights) == 0:
                return {'status': 'failed', 'message': 'Optimization produced invalid weights (NaN or zero sum)'}
            
            # Post-process: Enforce minimum positions if specified
            if min_positions is not None and min_positions > 0:
                # Count non-zero positions (threshold at 0.1% to avoid numerical noise)
                non_zero_mask = optimal_weights > 0.001
                current_positions = np.sum(non_zero_mask)
                
                if current_positions < min_positions:
                    # Need to add more positions
                    # Strategy: Scale down existing weights and add top-performing unselected assets
                    zero_mask = ~non_zero_mask
                    if np.sum(zero_mask) > 0:
                        # Get returns for unselected assets
                        asset_returns = returns.mean().values * 252
                        
                        # Check for valid returns
                        if not np.any(np.isnan(asset_returns)):
                            # Sort unselected assets by Sharpe ratio (return/risk)
                            asset_volatility = returns.std().values * np.sqrt(252)
                            asset_sharpe = asset_returns / (asset_volatility + 1e-10)  # Add small epsilon to avoid division by zero
                            
                            zero_indices = np.where(zero_mask)[0]
                            zero_sharpe = asset_sharpe[zero_indices]
                            sorted_indices = zero_indices[np.argsort(-zero_sharpe)]
                            
                            # Calculate weight to allocate to new positions
                            positions_to_add = min_positions - current_positions
                            weight_per_new_position = min(0.05, 1.0 / min_positions)  # At least 5% or equal weight
                            total_new_weight = positions_to_add * weight_per_new_position
                            
                            # Scale down existing weights to make room
                            if total_new_weight < 1.0:
                                optimal_weights[non_zero_mask] *= (1.0 - total_new_weight)
                            
                            # Add new positions
                            for i in range(min(positions_to_add, len(sorted_indices))):
                                idx = sorted_indices[i]
                                optimal_weights[idx] = weight_per_new_position
                            
                            # Ensure weights sum to 1
                            weight_sum = np.sum(optimal_weights)
                            if weight_sum > 0 and not np.isclose(weight_sum, 1.0):
                                optimal_weights = optimal_weights / weight_sum
            
            stats = self.calculate_portfolio_stats(optimal_weights, returns)
            
            return {
                'weights': optimal_weights,
                'tickers': returns.columns.tolist(),
                'stats': stats,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': f"Optimization failed: {problem.status}"}


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies with proper rebalancing simulation.
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost
    
    def backtest_portfolio(self, weights: np.ndarray, prices: pd.DataFrame, 
                          rebalance_freq: str = 'M') -> pd.DataFrame:
        """
        Backtest a portfolio strategy with proper rebalancing that accounts for weight drift.
        
        Args:
            weights: Target portfolio weights
            prices: Historical price data
            rebalance_freq: Rebalancing frequency ('M', 'Q', '6M', 'Y')
            
        Returns:
            DataFrame with portfolio performance including rebalancing costs
        """
        returns = prices.pct_change().dropna()
        
        # Handle deprecated frequency aliases
        freq_map = {'M': 'ME', 'Q': 'QE', 'Y': 'YE', 'A': 'YE', '6M': '6ME'}
        actual_freq = freq_map.get(rebalance_freq, rebalance_freq)
        
        # Generate rebalancing dates
        rebalance_dates = set(pd.date_range(start=returns.index[0], 
                                           end=returns.index[-1], 
                                           freq=actual_freq))
        
        # Initialize portfolio tracking
        portfolio_value = 1.0  # Start with $1
        current_weights = weights.copy()  # Current weights (will drift due to price changes)
        target_weights = weights.copy()   # Target weights (constant)
        
        daily_returns = []
        daily_portfolio_values = []
        rebalancing_costs = []
        
        for i, date in enumerate(returns.index):
            # Get today's asset returns
            daily_asset_returns = returns.loc[date].values
            
            # Check if we need to rebalance today (before calculating returns)
            is_rebalance_day = (date in rebalance_dates) or (i == 0)  # Always rebalance on first day
            transaction_cost_today = 0.0
            
            if is_rebalance_day and i > 0:  # Don't charge costs on first day
                # Calculate turnover required to get back to target weights
                weight_differences = np.abs(current_weights - target_weights)
                turnover = np.sum(weight_differences) / 2  # Divide by 2 since buying one asset = selling another
                
                # Apply transaction cost (reduces portfolio value)
                transaction_cost_today = turnover * self.transaction_cost
                portfolio_value *= (1 - transaction_cost_today)
                
                # Reset to target weights after rebalancing
                current_weights = target_weights.copy()
            
            # Calculate portfolio return using current weights
            portfolio_return = np.sum(current_weights * daily_asset_returns)
            daily_returns.append(portfolio_return)
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            daily_portfolio_values.append(portfolio_value)
            
            # Update current weights due to differential asset performance (weight drift)
            if i < len(returns.index) - 1:  # Don't update weights on last day
                asset_multipliers = 1 + daily_asset_returns
                current_weights = current_weights * asset_multipliers
                current_weights = current_weights / np.sum(current_weights)  # Renormalize to sum to 1
                
            rebalancing_costs.append(transaction_cost_today)
          # Convert to pandas Series for consistency
        portfolio_returns_series = pd.Series(daily_returns, index=returns.index)
        cumulative_returns_series = pd.Series(daily_portfolio_values, index=returns.index)
        rebalancing_costs_series = pd.Series(rebalancing_costs, index=returns.index)
        
        return pd.DataFrame({
            'returns': portfolio_returns_series,
            'cumulative_returns': cumulative_returns_series,
            'portfolio_value': cumulative_returns_series * 100000,  # Scale to $100k initial
            'rebalancing_costs': rebalancing_costs_series
        })
    
    def backtest_portfolio_walk_forward(self, prices: pd.DataFrame, 
                                      rebalance_freq: str = 'M',
                                      min_history_days: int = 252) -> pd.DataFrame:
        """
        Backtest a portfolio strategy with walk-forward analysis (dynamic re-optimization).
        
        At each rebalancing date, the portfolio is re-optimized using only historical data
        available up to that point, preventing look-ahead bias.
        
        Args:
            prices: Historical price data
            rebalance_freq: Rebalancing frequency ('M', 'Q', '6M', 'Y')
            min_history_days: Minimum days of history required for optimization
            
        Returns:
            DataFrame with portfolio performance including rebalancing costs
        """
        returns = prices.pct_change().dropna()
        
        # Handle deprecated frequency aliases
        freq_map = {'M': 'ME', 'Q': 'QE', 'Y': 'YE', 'A': 'YE', '6M': '6ME'}
        actual_freq = freq_map.get(rebalance_freq, rebalance_freq)
        
        # Generate rebalancing dates starting after minimum history period
        start_date = returns.index[min_history_days] if len(returns) > min_history_days else returns.index[len(returns)//2]
        rebalance_dates = list(pd.date_range(start=start_date, 
                                           end=returns.index[-1], 
                                           freq=actual_freq))
        
        # Initialize portfolio tracking
        portfolio_value = 1.0
        current_weights = None
        optimizer = PortfolioOptimizer()
        
        daily_returns = []
        daily_portfolio_values = []
        rebalancing_costs = []
        optimization_dates = []
        
        for i, date in enumerate(returns.index):
            # Check if we need to rebalance/optimize today
            is_rebalance_day = date in rebalance_dates or current_weights is None
            transaction_cost_today = 0.0
            
            if is_rebalance_day:
                # Get historical data up to current date for optimization
                if date == returns.index[0]:
                    # First day - use equal weights
                    n_assets = len(returns.columns)
                    new_weights = np.ones(n_assets) / n_assets
                    optimization_dates.append(date)
                else:
                    # Re-optimize using only historical data
                    historical_returns = returns.loc[:date].iloc[:-1]  # Exclude current day
                    
                    if len(historical_returns) >= min_history_days:
                        try:
                            result = optimizer.optimize_portfolio(historical_returns)
                            if result['status'] == 'optimal':
                                new_weights = result['weights']
                                optimization_dates.append(date)
                            else:
                                # Fallback to equal weights if optimization fails
                                n_assets = len(returns.columns)
                                new_weights = np.ones(n_assets) / n_assets
                        except Exception as e:
                            # Fallback to equal weights on error
                            n_assets = len(returns.columns)
                            new_weights = np.ones(n_assets) / n_assets
                    else:
                        # Not enough history - use equal weights
                        n_assets = len(returns.columns)
                        new_weights = np.ones(n_assets) / n_assets
                
                # Calculate transaction costs if we had previous weights
                if current_weights is not None:
                    weight_differences = np.abs(current_weights - new_weights)
                    turnover = np.sum(weight_differences) / 2
                    transaction_cost_today = turnover * self.transaction_cost
                    portfolio_value *= (1 - transaction_cost_today)
                
                # Update current weights
                current_weights = new_weights.copy()
            
            # Calculate portfolio return using current weights
            daily_asset_returns = returns.loc[date].values
            portfolio_return = np.sum(current_weights * daily_asset_returns)
            daily_returns.append(portfolio_return)
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            daily_portfolio_values.append(portfolio_value)
            
            # Update current weights due to differential asset performance (weight drift)
            if i < len(returns.index) - 1:  # Don't update weights on last day
                asset_multipliers = 1 + daily_asset_returns
                current_weights = current_weights * asset_multipliers
                current_weights = current_weights / np.sum(current_weights)
                
            rebalancing_costs.append(transaction_cost_today)
          # Convert to pandas Series for consistency
        portfolio_returns_series = pd.Series(daily_returns, index=returns.index)
        cumulative_returns_series = pd.Series(daily_portfolio_values, index=returns.index)
        rebalancing_costs_series = pd.Series(rebalancing_costs, index=returns.index)
        
        # Create a result DataFrame
        result_df = pd.DataFrame({
            'returns': portfolio_returns_series,
            'cumulative_returns': cumulative_returns_series,
            'portfolio_value': cumulative_returns_series * 100000,  # Scale to $100k initial
            'rebalancing_costs': rebalancing_costs_series
        })
        
        # Store optimization dates as an attribute for reference
        result_df.attrs['optimization_dates'] = optimization_dates
        result_df.attrs['optimization_count'] = len(optimization_dates)
        
        return result_df


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio analysis.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def simulate_portfolio_paths(self, weights: np.ndarray, returns: pd.DataFrame, 
                                time_horizon: int = 252) -> np.ndarray:
        """
        Simulate future portfolio paths using Monte Carlo with Geometric Brownian Motion.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns data
            time_horizon: Number of days to simulate
            
        Returns:
            Array of simulated portfolio paths (n_simulations x time_horizon+1)
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Check for invalid data
        if portfolio_returns.isna().any():
            portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            # Return flat paths if no valid data
            paths = np.ones((self.n_simulations, time_horizon + 1))
            return paths
        
        # Portfolio statistics (daily)
        mu_daily = portfolio_returns.mean()
        sigma_daily = portfolio_returns.std()
        
        # Validate statistics
        if np.isnan(mu_daily) or np.isnan(sigma_daily) or sigma_daily == 0:
            # Return flat paths if invalid statistics
            paths = np.ones((self.n_simulations, time_horizon + 1))
            return paths
        
        # Annualized drift and volatility for GBM
        # Drift adjustment for geometric returns
        drift = mu_daily - 0.5 * sigma_daily**2
        
        # Generate random paths using geometric Brownian motion
        # Use log-space to prevent overflow
        log_paths = np.zeros((self.n_simulations, time_horizon + 1))
        log_paths[:, 0] = 0.0  # log(1.0) = 0
        
        # Generate all random shocks at once for efficiency
        dt = 1  # Daily time step
        random_shocks = np.random.normal(0, 1, (self.n_simulations, time_horizon))
        
        # Simulate paths in log-space (more numerically stable)
        for t in range(1, time_horizon + 1):
            # Log-return for GBM: log(S_t/S_{t-1}) = drift*dt + sigma*sqrt(dt)*Z
            log_return = drift * dt + sigma_daily * np.sqrt(dt) * random_shocks[:, t-1]
            
            # Clip to prevent extreme values
            log_return = np.clip(log_return, -0.5, 0.5)  # Clip daily log returns to ±50%
            
            # Accumulate in log-space
            log_paths[:, t] = log_paths[:, t-1] + log_return
        
        # Convert back from log-space to normal space
        # Use clip to prevent overflow in exp
        log_paths = np.clip(log_paths, -20, 20)  # Prevents exp overflow
        paths = np.exp(log_paths)
        
        # Final safety check
        paths = np.nan_to_num(paths, nan=1.0, posinf=100.0, neginf=0.01)
        
        return paths


if __name__ == "__main__":
    # Example usage
    optimizer = PortfolioOptimizer()
    tickers = optimizer.fetch_sp500_tickers(20)
    print(f"Selected tickers: {tickers}")
