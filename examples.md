# Example configurations and usage scenarios

## Quick Start Examples

### Basic Portfolio Optimization
```python
from portfolio_optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(risk_free_rate=0.02)

# Get top 20 S&P 500 stocks
tickers = optimizer.fetch_sp500_tickers(20)

# Fetch 3 years of historical data
data = optimizer.fetch_historical_data(tickers, period="3y")

# Calculate returns
returns = optimizer.calculate_returns(data)

# Optimize portfolio for maximum Sharpe ratio
result = optimizer.optimize_portfolio(returns)

if result['status'] == 'optimal':
    print(f"Optimal Sharpe Ratio: {result['stats']['sharpe_ratio']:.2f}")
    print(f"Expected Return: {result['stats']['return']:.1%}")
    print(f"Volatility: {result['stats']['volatility']:.1%}")
```

### Backtesting Example
```python
from portfolio_optimizer import BacktestEngine
import yfinance as yf

# Initialize backtest engine
backtest = BacktestEngine(transaction_cost=0.001)

# Run backtest with monthly rebalancing
performance = backtest.backtest_portfolio(
    weights=result['weights'],
    prices=data,
    rebalance_freq='M'
)

# Compare with S&P 500
spy_data = yf.download('SPY', period="3y")['Adj Close']
spy_returns = spy_data.pct_change().dropna()

print(f"Portfolio Annual Return: {performance['returns'].mean() * 252:.1%}")
print(f"S&P 500 Annual Return: {spy_returns.mean() * 252:.1%}")
```

### Monte Carlo Simulation Example
```python
from portfolio_optimizer import MonteCarloSimulator

# Initialize simulator
simulator = MonteCarloSimulator(n_simulations=1000)

# Simulate 1 year of performance
paths = simulator.simulate_portfolio_paths(
    weights=result['weights'],
    returns=returns,
    time_horizon=252
)

# Analyze results
final_values = paths[:, -1]
print(f"Expected Value: {final_values.mean():.2f}")
print(f"95% Confidence Interval: [{np.percentile(final_values, 2.5):.2f}, {np.percentile(final_values, 97.5):.2f}]")
print(f"Probability of Loss: {(final_values < 1).mean() * 100:.1f}%")
```

## Configuration Examples

### Conservative Portfolio (Low Risk)
```python
# Focus on large-cap, low-volatility stocks
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
tickers = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'WMT', 'VZ', 'T']
data = optimizer.fetch_historical_data(tickers, period="5y")
returns = optimizer.calculate_returns(data)

# Optimize for minimum variance
result = optimizer.optimize_portfolio(returns, target_return=0.08)  # Target 8% return
```

### Aggressive Portfolio (High Risk)
```python
# Include growth and tech stocks
tickers = ['NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'NFLX', 'AMD', 'CRM']
data = optimizer.fetch_historical_data(tickers, period="3y")
returns = optimizer.calculate_returns(data)

# Optimize for maximum Sharpe ratio
result = optimizer.optimize_portfolio(returns)
```

### Sector-Diversified Portfolio
```python
# Representative stocks from different sectors
sector_stocks = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL'],
    'Healthcare': ['JNJ', 'UNH', 'PFE'],
    'Finance': ['JPM', 'BAC', 'WFC'],
    'Energy': ['XOM', 'CVX', 'COP'],
    'Consumer': ['WMT', 'PG', 'KO']
}

all_tickers = [stock for stocks in sector_stocks.values() for stock in stocks]
data = optimizer.fetch_historical_data(all_tickers, period="5y")
returns = optimizer.calculate_returns(data)
result = optimizer.optimize_portfolio(returns)
```

## Advanced Usage

### Custom Risk Metrics
```python
from utils import RiskMetrics, PerformanceAnalyzer

# Calculate comprehensive risk metrics
analyzer = PerformanceAnalyzer()
portfolio_returns = (returns * result['weights']).sum(axis=1)

# Get benchmark data
spy_returns = yf.download('SPY', period="5y")['Adj Close'].pct_change().dropna()

# Generate comprehensive report
report = analyzer.generate_performance_report(
    portfolio_returns=portfolio_returns,
    benchmark_returns=spy_returns,
    risk_free_rate=0.02
)

print(f"Alpha: {report['alpha']:.2%}")
print(f"Beta: {report['beta']:.2f}")
print(f"Information Ratio: {report['information_ratio']:.2f}")
print(f"Maximum Drawdown: {report['max_drawdown']:.2%}")
```

### Efficient Frontier Analysis
```python
import numpy as np

# Generate efficient frontier
target_returns = np.linspace(0.05, 0.20, 20)  # 5% to 20% target returns
efficient_portfolios = []

for target in target_returns:
    try:
        result = optimizer.optimize_portfolio(returns, target_return=target)
        if result['status'] == 'optimal':
            efficient_portfolios.append({
                'target_return': target,
                'return': result['stats']['return'],
                'volatility': result['stats']['volatility'],
                'sharpe_ratio': result['stats']['sharpe_ratio'],
                'weights': result['weights']
            })
    except:
        continue

# Find optimal portfolio (max Sharpe ratio)
optimal_portfolio = max(efficient_portfolios, key=lambda x: x['sharpe_ratio'])
print(f"Optimal Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
print(f"Optimal Return: {optimal_portfolio['return']:.1%}")
print(f"Optimal Risk: {optimal_portfolio['volatility']:.1%}")
```

## Performance Comparison Strategies

### Strategy 1: Equal Weight Portfolio
```python
n_assets = len(returns.columns)
equal_weights = np.ones(n_assets) / n_assets
equal_weight_stats = optimizer.calculate_portfolio_stats(equal_weights, returns)
```

### Strategy 2: Market Cap Weighted (Proxy)
```python
# Use inverse volatility as proxy for market cap weighting
volatilities = returns.std()
market_cap_weights = (1 / volatilities) / (1 / volatilities).sum()
market_cap_stats = optimizer.calculate_portfolio_stats(market_cap_weights.values, returns)
```

### Strategy 3: Minimum Variance Portfolio
```python
# Optimize for minimum variance (no return constraint)
min_var_result = optimizer.optimize_portfolio(returns, target_return=returns.mean().min() * 252)
```

## Streamlit Dashboard Customization

### Custom Metrics Display
```python
# Add custom metrics to the dashboard
def display_custom_metrics(result, returns):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        concentration = 1 / np.sum(result['weights']**2)
        st.metric("Concentration Ratio", f"{concentration:.1f}")
    
    with col2:
        portfolio_returns = (returns * result['weights']).sum(axis=1)
        skewness = portfolio_returns.skew()
        st.metric("Skewness", f"{skewness:.2f}")
    
    with col3:
        kurtosis = portfolio_returns.kurtosis()
        st.metric("Kurtosis", f"{kurtosis:.2f}")
```

### Custom Chart Styling
```python
# Custom color scheme for charts
CUSTOM_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D'
}

def create_custom_pie_chart(weights, tickers):
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        marker_colors=[CUSTOM_COLORS['primary'], CUSTOM_COLORS['secondary'], 
                      CUSTOM_COLORS['accent'], CUSTOM_COLORS['success']] * 10
    )])
    return fig
```

## Troubleshooting Common Issues

### Data Fetching Issues
```python
# Handle API rate limiting
import time

def fetch_data_with_retry(tickers, period, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, period=period, progress=False)
            return data['Adj Close'].dropna()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return pd.DataFrame()
```

### Optimization Issues
```python
# Handle optimization failures
def robust_optimization(returns, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            result = optimizer.optimize_portfolio(returns)
            if result['status'] == 'optimal':
                return result
        except Exception as e:
            print(f"Optimization attempt {attempt + 1} failed: {e}")
    
    # Fallback to equal weights
    n_assets = len(returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    stats = optimizer.calculate_portfolio_stats(equal_weights, returns)
    
    return {
        'weights': equal_weights,
        'tickers': returns.columns.tolist(),
        'stats': stats,
        'status': 'fallback'
    }
```
