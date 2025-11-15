"""
Portfolio Manager Pro - Modern Portfolio Theory Application
=========================================================

Professional portfolio optimization platform with advanced analytics and backtesting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf

from portfolio_optimizer import PortfolioOptimizer, BacktestEngine, MonteCarloSimulator
from quick_improvements import apply_quick_improvements, EnhancedErrorHandler, LoadingManager, MODERN_THEME_CSS

# Enhanced page configuration with modern theme
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://portfoliomanager.com/help',
        'Report a bug': 'https://portfoliomanager.com/bugs',
        'About': """
        # Portfolio Manager Pro v2.0
        
        Advanced portfolio optimization using Modern Portfolio Theory with walk-forward analysis.
        
        **Features:**
        - MPT-based optimization with CVXPY
        - Walk-forward analysis capability
        - Comprehensive risk metrics
        - Interactive visualizations
        - Professional backtesting engine
        
        Built with ❤️ using Streamlit and Python.
        """
    }
)

# Apply modern theme and initialize enhanced components
st.markdown(MODERN_THEME_CSS, unsafe_allow_html=True)
error_handler = EnhancedErrorHandler()
loading_manager = LoadingManager()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sp500_data(top_n, period):
    """Load S&P 500 data with caching."""
    try:
        optimizer = PortfolioOptimizer()
        tickers = optimizer.fetch_sp500_tickers(top_n)
        
        if not tickers:
            raise ValueError("No tickers fetched")
              data = optimizer.fetch_historical_data(tickers, period)
        
        if data.empty:
            raise ValueError("Empty data received from yfinance")
            
        return data, tickers, optimizer
    except Exception as e:
        st.error(f"Error in load_sp500_data: {str(e)}")
        # Return fallback data
        optimizer = PortfolioOptimizer()
        fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        try:
            data = optimizer.fetch_historical_data(fallback_tickers, period)
            return data, fallback_tickers, optimizer
        except:
            return pd.DataFrame(), [], optimizer

@st.cache_data
def calculate_benchmark_performance(period):
    """Calculate S&P 500 ETF (SPY) performance."""
    try:
        # Download SPY data with explicit auto_adjust setting
        spy_data = yf.download('SPY', period=period, progress=False, auto_adjust=True)
        
        # Handle the data structure - for single ticker, check if it's MultiIndex
        if isinstance(spy_data.columns, pd.MultiIndex):
            if 'Adj Close' in spy_data.columns.levels[0]:
                spy_prices = spy_data['Adj Close']['SPY']
            else:
                spy_prices = spy_data['Close']['SPY']
        else:
            # Single ticker returns simple DataFrame
            if 'Adj Close' in spy_data.columns:
                spy_prices = spy_data['Adj Close']
            else:
                spy_prices = spy_data['Close']
        
        spy_returns = spy_prices.pct_change().dropna()
        spy_cumulative = (1 + spy_returns).cumprod()
        
        return {
            'returns': spy_returns,
            'cumulative_returns': spy_cumulative,
            'annual_return': spy_returns.mean() * 252,
            'volatility': spy_returns.std() * np.sqrt(252),
            'sharpe_ratio': (spy_returns.mean() * 252 - 0.02) / (spy_returns.std() * np.sqrt(252))
        }
    except Exception as e:
        st.error(f"Error fetching benchmark data: {e}")
        # Return dummy data to prevent crashes
        dummy_returns = pd.Series([0.001] * 252, name='SPY')
        return {
            'returns': dummy_returns,
            'cumulative_returns': (1 + dummy_returns).cumprod(),
            'annual_return': 0.05,
            'volatility': 0.15,
            'sharpe_ratio': 0.20
        }

@st.cache_data
def calculate_equal_weight_performance(data, tickers):
    """Calculate equal-weight portfolio performance for top N stocks."""
    try:
        # Create equal weights for all selected stocks
        n_stocks = len(tickers)
        equal_weights = np.ones(n_stocks) / n_stocks
        
        # Calculate returns using equal weights
        returns = data.pct_change().dropna()
        portfolio_returns = (returns * equal_weights).sum(axis=1)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        return {
            'returns': portfolio_returns,
            'cumulative_returns': portfolio_cumulative,
            'annual_return': portfolio_returns.mean() * 252,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)),
            'weights': equal_weights
        }
    except Exception as e:
        st.error(f"Error calculating equal-weight performance: {e}")
        # Return dummy data to prevent crashes
        dummy_returns = pd.Series([0.0005] * len(data), index=data.index, name='Equal Weight')
        return {
            'returns': dummy_returns,
            'cumulative_returns': (1 + dummy_returns).cumprod(),
            'annual_return': 0.03,
            'volatility': 0.12,
            'sharpe_ratio': 0.08,
            'weights': np.ones(len(tickers)) / len(tickers)
        }

def create_portfolio_pie_chart(weights, tickers):
    """Create interactive pie chart for portfolio allocation."""
    # Filter out very small weights for better visualization
    threshold = 0.01
    significant_weights = []
    significant_tickers = []
    other_weight = 0
    
    for i, (weight, ticker) in enumerate(zip(weights, tickers)):
        if weight >= threshold:
            significant_weights.append(weight)
            significant_tickers.append(ticker)
        else:
            other_weight += weight
    
    if other_weight > 0:
        significant_weights.append(other_weight)
        significant_tickers.append('Others')
    
    fig = go.Figure(data=[go.Pie(
        labels=significant_tickers,
        values=significant_weights,
        hole=.3,
        textinfo='label+percent',
        textposition='outside',
        marker_colors=px.colors.qualitative.Set3
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        title_x=0.5,
        font=dict(size=14),
        showlegend=True,
        height=500
    )
    
    return fig

def create_efficient_frontier(optimizer, returns):
    """Create efficient frontier plot."""
    target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, 50)
    efficient_portfolios = []
    
    for target_return in target_returns:
        try:
            result = optimizer.optimize_portfolio(returns, target_return=target_return)
            if result['status'] == 'optimal':
                efficient_portfolios.append({
                    'return': result['stats']['return'],
                    'volatility': result['stats']['volatility'],
                    'sharpe_ratio': result['stats']['sharpe_ratio']
                })
        except:
            continue
    
    if efficient_portfolios:
        ef_df = pd.DataFrame(efficient_portfolios)
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=ef_df['volatility'],
            y=ef_df['return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Individual assets
        individual_returns = returns.mean() * 252
        individual_volatility = returns.std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=individual_volatility,
            y=individual_returns,
            mode='markers',
            name='Individual Assets',
            marker=dict(size=8, color='red', symbol='diamond'),
            text=returns.columns,
            textposition='top center'
        ))
        
        # Optimal portfolio
        max_sharpe_idx = ef_df['sharpe_ratio'].idxmax()
        fig.add_trace(go.Scatter(
            x=[ef_df.loc[max_sharpe_idx, 'volatility']],
            y=[ef_df.loc[max_sharpe_idx, 'return']],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=12, color='green', symbol='star')
        ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            showlegend=True,
            height=600
        )
        
        return fig
    else:
        return None

def create_backtest_comparison_chart(portfolio_performance, benchmark_performance, equal_weight_performance=None, strategy_name="Optimized Portfolio"):
    """Create comparison chart for backtesting results."""
    fig = go.Figure()
    
    # Portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_performance.index,
        y=portfolio_performance['cumulative_returns'],
        mode='lines',
        name=strategy_name,
        line=dict(color='blue', width=3)
    ))
    
    # Benchmark performance
    fig.add_trace(go.Scatter(
        x=benchmark_performance['cumulative_returns'].index,
        y=benchmark_performance['cumulative_returns'],
        mode='lines',
        name='S&P 500 (SPY)',
        line=dict(color='red', width=2)
    ))
    
    # Equal-weight portfolio performance
    if equal_weight_performance is not None:
        fig.add_trace(go.Scatter(
            x=equal_weight_performance['cumulative_returns'].index,
            y=equal_weight_performance['cumulative_returns'],
            mode='lines',
            name='Equal Weight Portfolio',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        showlegend=True,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns series."""
    try:
        # Calculate running maximum (peak)
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Return the maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
        
        return max_drawdown
    except Exception:
        return 0.0

def create_monte_carlo_chart(simulation_paths):
    """Create Monte Carlo simulation visualization."""
    fig = go.Figure()
    
    # Plot sample paths (not all to avoid clutter)
    sample_indices = np.random.choice(len(simulation_paths), size=min(100, len(simulation_paths)), replace=False)
    
    for i in sample_indices:
        fig.add_trace(go.Scatter(
            y=simulation_paths[i],
            mode='lines',
            line=dict(color='lightblue', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add percentiles
    percentiles = np.percentile(simulation_paths, [5, 50, 95], axis=0)
    
    fig.add_trace(go.Scatter(
        y=percentiles[1],
        mode='lines',
        name='Median (50th percentile)',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        y=percentiles[0],
        mode='lines',
        name='5th percentile',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        y=percentiles[2],
        mode='lines',
        name='95th percentile',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Monte Carlo Simulation - Portfolio Value Paths",
        xaxis_title="Days",
        yaxis_title="Portfolio Value (Normalized)",
        showlegend=True,
        height=500
    )
    
    return fig

# Main application
def main():
    # Header
    # Professional header with enhanced styling
    st.markdown('<h1 class="main-header">📊 Portfolio Manager Pro</h1>', unsafe_allow_html=True)
    
    # Professional info box
    st.markdown("""
    <div class="info-box">
        <strong>🚀 Professional Portfolio Optimization Platform</strong><br>
        Create optimal portfolios using Modern Portfolio Theory with advanced risk analysis, 
        walk-forward backtesting, and comprehensive performance metrics.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # Configuration parameters
    top_n_stocks = st.sidebar.slider("Top N S&P 500 Stocks", min_value=10, max_value=100, value=30, step=5)
    time_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "3y", "5y", "10y"], index=3)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
    transaction_cost = st.sidebar.number_input("Transaction Cost (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.01) / 100
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Actions")    run_optimization = st.sidebar.button("🚀 Run Portfolio Optimization", type="primary")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Optimization", "📊 Backtesting", "🎲 Monte Carlo Simulation", "📋 Performance Metrics"])
    
    # Enhanced data loading with better UX
    if 'data_loaded' not in st.session_state or st.sidebar.button("🔄 Reload Data"):
        try:
            # Create loading progress
            progress_placeholder = st.empty()
            with progress_placeholder:
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; margin: 2rem 0;">
                    <div class="loading-spinner"></div>
                    <span style="margin-left: 10px; font-weight: 500;">Loading S&P 500 market data...</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Load data
            data, tickers, optimizer = load_sp500_data(top_n_stocks, time_period)
            
            # Clear loading and validate data
            progress_placeholder.empty()
            
            if data.empty or len(tickers) == 0:
                error_handler.show_error('insufficient_data', 
                    f"No data loaded for period: {time_period}, stocks: {top_n_stocks}")
                return
            
            # Store data and show success
            st.session_state.data = data
            st.session_state.tickers = tickers
            st.session_state.optimizer = optimizer
            st.session_state.data_loaded = True
            
            # Success notification
            st.success(f"✅ Successfully loaded {len(tickers)} stocks with {len(data)} days of data!")
            
        except Exception as e:
            error_handler.show_error('network_error', str(e))
            # Provide fallback empty data to prevent crashes
            st.session_state.data = pd.DataFrame()
            st.session_state.tickers = []
            st.session_state.optimizer = PortfolioOptimizer()
            st.session_state.data_loaded = True
            return
      data = st.session_state.data
    tickers = st.session_state.tickers
    optimizer = st.session_state.optimizer
    optimizer.risk_free_rate = risk_free_rate
    
    if data.empty:
        error_handler.show_error('network_error', 
            f"Data loading failed. Period: {time_period}, Stocks: {top_n_stocks}")
        return
    
    # Professional status display in sidebar
    st.sidebar.markdown("### 📊 **Data Status**")
    st.sidebar.success(f"✅ **{len(tickers)} stocks** | **{len(data)} trading days**")
    st.sidebar.info(f"📅 Period: **{time_period.upper()}** | 🔄 Risk-free rate: **{risk_free_rate:.1%}**")
    
    # Quick stats in sidebar
    if len(data) > 0:
        date_range = f"{data.index[0].strftime('%b %Y')} - {data.index[-1].strftime('%b %Y')}"
        st.sidebar.caption(f"📈 Data range: {date_range}")
    
    # Tab 1: Portfolio Optimization
    with tab1:
        st.header("🎯 Portfolio Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📋 Portfolio Summary")
            st.info(f"**Stocks Selected:** {len(tickers)}")
            st.info(f"**Data Period:** {time_period}")        st.info(f"**Risk-Free Rate:** {risk_free_rate:.1%}")
        
        if run_optimization or 'optimization_results' not in st.session_state:
            try:
                # Enhanced loading with progress steps
                progress_placeholder = st.empty()
                
                # Step 1: Calculate returns
                with progress_placeholder:
                    st.markdown("""
                    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                        <div class="loading-spinner"></div>
                        <span style="margin-left: 10px; font-weight: 500;">Calculating historical returns...</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                returns = optimizer.calculate_returns(data)
                
                # Step 2: Run optimization
                with progress_placeholder:
                    st.markdown("""
                    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                        <div class="loading-spinner"></div>
                        <span style="margin-left: 10px; font-weight: 500;">Running MPT optimization algorithm...</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                result = optimizer.optimize_portfolio(returns)
                
                # Clear loading
                progress_placeholder.empty()
                
                if result['status'] == 'optimal':
                    st.session_state.optimization_results = result
                    st.session_state.returns_data = returns
                    
                    # Success message with key metrics
                    st.success(f"✅ **Portfolio Optimized!** Sharpe Ratio: **{result['stats']['sharpe_ratio']:.3f}** | "
                             f"Expected Return: **{result['stats']['return']:.1%}** | "
                             f"Risk: **{result['stats']['volatility']:.1%}**")
                else:
                    error_handler.show_error('optimization_failed', 
                        f"Optimization status: {result.get('message', 'Unknown error')}")
                    return
                    
            except Exception as e:
                error_handler.show_error('optimization_failed', str(e))
                return
        
        if 'optimization_results' in st.session_state:
            result = st.session_state.optimization_results
            
            with col1:
                # Portfolio allocation pie chart
                pie_chart = create_portfolio_pie_chart(result['weights'], result['tickers'])
                st.plotly_chart(pie_chart, width='stretch')
              # Enhanced performance metrics with modern styling
            st.markdown("### 📊 **Portfolio Performance Metrics**")
            
            # Create enhanced metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--primary-blue);">Expected Return</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-dark);">{:.1%}</h2>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">Annualized</p>
                </div>
                """.format(result['stats']['return']), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--secondary-purple);">Volatility</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-dark);">{:.1%}</h2>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">Risk measure</p>
                </div>
                """.format(result['stats']['volatility']), unsafe_allow_html=True)
                
            with col3:
                sharpe_color = "var(--success-green)" if result['stats']['sharpe_ratio'] > 1.0 else "var(--accent-orange)"
                st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0; color: {};">Sharpe Ratio</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-dark);">{:.3f}</h2>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">Risk-adjusted return</p>
                </div>
                """.format(sharpe_color, result['stats']['sharpe_ratio']), unsafe_allow_html=True)
                
            with col4:
                positions = np.sum(result['weights'] > 0.01)
                st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0; color: var(--accent-orange);">Positions</h4>
                    <h2 style="margin: 0.5rem 0; color: var(--text-dark);">{}</h2>
                    <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">Active holdings</p>
                </div>
                """.format(positions), unsafe_allow_html=True)
            
            # Efficient Frontier
            st.subheader("🎯 Efficient Frontier")
            if 'returns_data' in st.session_state:
                ef_chart = create_efficient_frontier(optimizer, st.session_state.returns_data)
                if ef_chart:
                    st.plotly_chart(ef_chart, width='stretch')
                else:
                    st.warning("Could not generate efficient frontier. Try adjusting parameters.")
            
            # Detailed allocation table
            st.subheader("📈 Detailed Portfolio Allocation")
            allocation_df = pd.DataFrame({
                'Ticker': result['tickers'],
                'Weight': result['weights'],
                'Weight (%)': result['weights'] * 100
            })
            allocation_df = allocation_df[allocation_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            st.dataframe(allocation_df, width='stretch')
      # Tab 2: Backtesting
    with tab2:
        st.header("📊 Backtesting Analysis")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first.")
        else:            # Backtesting configuration
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("🔧 Backtest Settings")
                rebalance_freq = st.selectbox("Rebalancing Frequency", 
                                            ["Monthly", "Quarterly", "Semi-Annual", "Annual"], 
                                            index=0)
                freq_map = {"Monthly": "M", "Quarterly": "Q", "Semi-Annual": "6M", "Annual": "Y"}
                  # Backtesting methodology selection
                backtest_method = st.radio("Backtesting Method", 
                                         ["Fixed Weights", "Walk-Forward Analysis"], 
                                         index=0,
                                         help="""
                                         • **Fixed Weights**: Use initial optimization weights throughout, rebalance periodically
                                         • **Walk-Forward**: Re-optimize portfolio at each rebalancing date using only historical data
                                         """)
                
                if backtest_method == "Fixed Weights":
                    st.info(f"**Comparing 3 strategies:**\n"
                           f"• **Optimized**: MPT-optimized weights (fixed)\n"
                           f"• **Equal Weight**: 1/N allocation across {len(tickers)} stocks\n"
                           f"• **S&P 500**: Market benchmark (SPY)")
                else:
                    st.info(f"**Comparing 3 strategies:**\n"
                           f"• **Walk-Forward**: Re-optimized at each rebalance\n"
                           f"• **Equal Weight**: 1/N allocation across {len(tickers)} stocks\n"
                           f"• **S&P 500**: Market benchmark (SPY)")
                  run_backtest = st.button("🏃‍♂️ Run Backtest", type="primary")
                
                if run_backtest or 'backtest_results' not in st.session_state:
                    try:
                    # Enhanced backtesting with progress indication
                    progress_placeholder = st.empty()
                    
                    with progress_placeholder:
                        st.markdown("""
                        <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                            <div class="loading-spinner"></div>
                            <span style="margin-left: 10px; font-weight: 500;">Initializing backtest engine...</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Portfolio backtest
                    backtest_engine = BacktestEngine(transaction_cost=transaction_cost)
                    
                    if backtest_method == "Fixed Weights":
                        # Use fixed weights from initial optimization
                        result = st.session_state.optimization_results
                        portfolio_performance = backtest_engine.backtest_portfolio(
                            result['weights'], data, freq_map[rebalance_freq]
                        )
                        strategy_name = "Optimized (Fixed)"
                    else:
                        # Use walk-forward analysis with dynamic re-optimization
                        portfolio_performance = backtest_engine.backtest_portfolio_walk_forward(
                            data, freq_map[rebalance_freq]
                        )
                        strategy_name = "Walk-Forward"
                    
                    # Benchmark performance
                    benchmark_performance = calculate_benchmark_performance(time_period)
                    
                    # Equal-weight portfolio performance
                    equal_weight_performance = calculate_equal_weight_performance(data, tickers)
                      # Equal-weight backtest with same rebalancing
                    equal_weight_backtest = backtest_engine.backtest_portfolio(
                        equal_weight_performance['weights'], data, freq_map[rebalance_freq]
                    )
                      # Clear progress and store results
                    progress_placeholder.empty()
                    
                    st.session_state.backtest_results = {
                        'portfolio': portfolio_performance,
                        'benchmark': benchmark_performance,
                        'equal_weight': equal_weight_backtest,
                        'method': backtest_method,
                        'strategy_name': strategy_name
                    }
                    
                    # Success message
                    st.success(f"✅ **Backtest Complete!** {strategy_name} strategy analyzed with {rebalance_freq.lower()} rebalancing.")
                    
                except Exception as e:
                    error_handler.show_error('optimization_failed', f"Backtesting error: {str(e)}")
                    return
            
            if 'backtest_results' in st.session_state:
                backtest_results = st.session_state.backtest_results
                portfolio_perf = backtest_results['portfolio']
                benchmark_perf = backtest_results['benchmark']
                equal_weight_perf = backtest_results.get('equal_weight')
                
                with col1:
                    # Performance comparison chart
                    strategy_name = backtest_results.get('strategy_name', 'Optimized Portfolio')
                    comparison_chart = create_backtest_comparison_chart(portfolio_perf, benchmark_perf, equal_weight_perf, strategy_name)
                    st.plotly_chart(comparison_chart, width='stretch')
                
                # Performance metrics comparison
                st.subheader("📈 Performance Comparison")
                
                portfolio_annual_return = portfolio_perf['returns'].mean() * 252
                portfolio_volatility = portfolio_perf['returns'].std() * np.sqrt(252)
                portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
                
                # Equal-weight portfolio metrics
                if equal_weight_perf is not None:
                    equal_weight_annual_return = equal_weight_perf['returns'].mean() * 252
                    equal_weight_volatility = equal_weight_perf['returns'].std() * np.sqrt(252)
                    equal_weight_sharpe = (equal_weight_annual_return - risk_free_rate) / equal_weight_volatility
                  # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    strategy_display = backtest_results.get('strategy_name', 'Optimized Portfolio')
                    st.markdown(f"**🎯 {strategy_display}**")
                    st.metric("Annual Return", f"{portfolio_annual_return:.1%}")
                    st.metric("Volatility", f"{portfolio_volatility:.1%}")
                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
                      # Add walk-forward specific info
                    if backtest_results.get('method') == "Walk-Forward Analysis":
                        opt_count = portfolio_perf.attrs.get('optimization_count', 0)
                        if opt_count > 0:
                            st.caption(f"📈 Re-optimized {opt_count} times")
                
                with col2:
                    st.markdown("**⚖️ Equal Weight Portfolio**")
                    if equal_weight_perf is not None:
                        st.metric("Annual Return", f"{equal_weight_annual_return:.1%}")
                        st.metric("Volatility", f"{equal_weight_volatility:.1%}")
                        st.metric("Sharpe Ratio", f"{equal_weight_sharpe:.2f}")
                    else:
                        st.info("Run backtest to see results")
                
                with col3:
                    st.markdown("**📊 S&P 500 Benchmark**")
                    st.metric("Annual Return", f"{benchmark_perf['annual_return']:.1%}")
                    st.metric("Volatility", f"{benchmark_perf['volatility']:.1%}")
                    st.metric("Sharpe Ratio", f"{benchmark_perf['sharpe_ratio']:.2f}")
                
                with col4:
                    st.markdown("**🚀 Optimization Advantage**")
                    return_diff = portfolio_annual_return - benchmark_perf['annual_return']
                    
                    if equal_weight_perf is not None:
                        eq_weight_diff = portfolio_annual_return - equal_weight_annual_return
                        st.metric("vs Equal Weight", f"{eq_weight_diff:.1%}")
                    
                    st.metric("vs S&P 500", f"{return_diff:.1%}")
                    
                    if equal_weight_perf is not None:
                        sharpe_diff_eq = portfolio_sharpe - equal_weight_sharpe
                        st.metric("Sharpe Advantage", f"{sharpe_diff_eq:.2f}")
                
                # Detailed comparison table
                st.subheader("📋 Detailed Performance Metrics")
                
                comparison_data = {
                    'Strategy': ['Optimized Portfolio', 'S&P 500 Benchmark'],
                    'Annual Return': [f"{portfolio_annual_return:.1%}", f"{benchmark_perf['annual_return']:.1%}"],
                    'Volatility': [f"{portfolio_volatility:.1%}", f"{benchmark_perf['volatility']:.1%}"],
                    'Sharpe Ratio': [f"{portfolio_sharpe:.2f}", f"{benchmark_perf['sharpe_ratio']:.2f}"],
                    'Max Drawdown': [f"{calculate_max_drawdown(portfolio_perf['cumulative_returns']):.1%}", f"{calculate_max_drawdown(benchmark_perf['cumulative_returns']):.1%}"],
                    'Total Return': [
                        f"{(portfolio_perf['cumulative_returns'].iloc[-1] - 1):.1%}",
                        f"{(benchmark_perf['cumulative_returns'].iloc[-1] - 1):.1%}"
                    ]                }
                
                if equal_weight_perf is not None:
                    comparison_data['Strategy'].insert(1, 'Equal Weight Portfolio')
                    comparison_data['Annual Return'].insert(1, f"{equal_weight_annual_return:.1%}")
                    comparison_data['Volatility'].insert(1, f"{equal_weight_volatility:.1%}")
                    comparison_data['Sharpe Ratio'].insert(1, f"{equal_weight_sharpe:.2f}")
                    comparison_data['Max Drawdown'].insert(1, f"{calculate_max_drawdown(equal_weight_perf['cumulative_returns']):.1%}")
                    comparison_data['Total Return'].insert(1, f"{(equal_weight_perf['cumulative_returns'].iloc[-1] - 1):.1%}")
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch', hide_index=True)
                
                # Method explanation
                st.subheader("ℹ️ Backtesting Methodology")
                method_used = backtest_results.get('method', 'Fixed Weights')
                
                if method_used == "Fixed Weights":
                    st.info("""
                    **Fixed Weights Method:**
                    - Portfolio weights are optimized once at the beginning using all historical data
                    - These target weights remain constant throughout the backtest period
                    - Portfolio is rebalanced to target weights at specified intervals
                    - Simulates a "buy and hold with periodic rebalancing" strategy
                    - More practical for individual investors, lower transaction costs
                    """)
                else:
                    st.info("""
                    **Walk-Forward Analysis Method:**
                    - Portfolio is re-optimized at each rebalancing date
                    - Uses only historical data available up to that point (no look-ahead bias)
                    - Weights change dynamically based on evolving market conditions
                    - More computationally intensive but potentially more adaptive
                    - Higher transaction costs due to more frequent weight changes
                    """)
                      # Show optimization frequency for walk-forward
                    opt_dates = portfolio_perf.attrs.get('optimization_dates', [])
                    if opt_dates:
                        st.write(f"**Re-optimization occurred {len(opt_dates)} times during the backtest period**")
                        with st.expander("View Re-optimization Dates"):
                            for i, date in enumerate(opt_dates[:10]):  # Show first 10 dates
                                st.write(f"{i+1}. {date.strftime('%Y-%m-%d')}")
                            if len(opt_dates) > 10:
                                st.write(f"... and {len(opt_dates) - 10} more dates")
    
    # Tab 3: Monte Carlo Simulation
    with tab3:
        st.header("🎲 Monte Carlo Simulation")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first.")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("⚙️ Simulation Settings")
                n_simulations = st.selectbox("Number of Simulations", [500, 1000, 2000, 5000], index=1)
                time_horizon = st.slider("Time Horizon (Days)", 30, 1000, 252)
                
                run_simulation = st.button("🎯 Run Simulation", type="primary")
            
            if run_simulation or 'monte_carlo_results' not in st.session_state:
                with st.spinner("Running Monte Carlo simulation..."):
                    simulator = MonteCarloSimulator(n_simulations=n_simulations)
                    result = st.session_state.optimization_results
                    returns = st.session_state.returns_data
                    
                    simulation_paths = simulator.simulate_portfolio_paths(
                        result['weights'], returns, time_horizon
                    )
                    
                    st.session_state.monte_carlo_results = {
                        'paths': simulation_paths,
                        'n_simulations': n_simulations,
                        'time_horizon': time_horizon
                    }
            
            if 'monte_carlo_results' in st.session_state:
                mc_results = st.session_state.monte_carlo_results
                
                with col1:
                    # Monte Carlo visualization
                    mc_chart = create_monte_carlo_chart(mc_results['paths'])
                    st.plotly_chart(mc_chart, width='stretch')
                
                # Simulation statistics
                st.subheader("📊 Simulation Results")
                
                final_values = mc_results['paths'][:, -1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expected Final Value", f"{np.mean(final_values):.2f}x")
                with col2:
                    st.metric("Median Final Value", f"{np.median(final_values):.2f}x")
                with col3:
                    st.metric("5th Percentile", f"{np.percentile(final_values, 5):.2f}x")
                with col4:
                    st.metric("95th Percentile", f"{np.percentile(final_values, 95):.2f}x")
                
                # Risk metrics
                st.subheader("🚨 Risk Analysis")
                probability_loss = np.mean(final_values < 1) * 100
                value_at_risk_5 = (1 - np.percentile(final_values, 5)) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability of Loss", f"{probability_loss:.1f}%")
                with col2:
                    st.metric("Value at Risk (5%)", f"{value_at_risk_5:.1f}%")
    
    # Tab 4: Performance Metrics
    with tab4:
        st.header("📋 Comprehensive Performance Analysis")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first.")
        else:
            # Summary dashboard
            result = st.session_state.optimization_results
            
            # Key metrics overview
            st.subheader("🎯 Key Performance Indicators")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Expected Return",
                    f"{result['stats']['return']:.1%}",
                    help="Annualized expected return based on historical data"
                )
            
            with col2:
                st.metric(
                    "Risk (Volatility)",
                    f"{result['stats']['volatility']:.1%}",
                    help="Annualized standard deviation of returns"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{result['stats']['sharpe_ratio']:.2f}",
                    help="Risk-adjusted return metric"
                )
            
            with col4:
                diversification = 1 / np.sum(result['weights']**2)
                st.metric(
                    "Diversification Ratio",
                    f"{diversification:.1f}",
                    help="Effective number of positions"
                )
            
            with col5:
                max_weight = np.max(result['weights'])
                st.metric(
                    "Max Position Size",
                    f"{max_weight:.1%}",
                    help="Largest single position weight"
                )
            
            # Detailed analysis
            st.subheader("📈 Detailed Analysis")
            
            # Risk-Return Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Risk-Return Profile**")
                returns = st.session_state.returns_data
                portfolio_returns = (returns * result['weights']).sum(axis=1)
                
                # Calculate additional metrics
                sortino_ratio = (result['stats']['return'] - risk_free_rate) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252))
                max_drawdown = ((portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()) * 100
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
                    'Value': [
                        f"{result['stats']['return']:.2%}",
                        f"{result['stats']['volatility']:.2%}",
                        f"{result['stats']['sharpe_ratio']:.2f}",
                        f"{sortino_ratio:.2f}",
                        f"{max_drawdown:.2f}%"
                    ]
                })
                st.dataframe(metrics_df, width='stretch', hide_index=True)
            
            with col2:
                st.markdown("**📊 Portfolio Composition**")
                
                # Sector analysis (simplified)
                top_holdings = pd.DataFrame({
                    'Ticker': result['tickers'],
                    'Weight': result['weights']
                }).sort_values('Weight', ascending=False).head(10)
                
                st.dataframe(top_holdings.style.format({'Weight': '{:.1%}'}), width='stretch', hide_index=True)
            
            # Performance attribution (simplified)
            st.subheader("🔍 Performance Attribution")
            
            individual_contributions = returns.mean() * 252 * result['weights']
            contribution_df = pd.DataFrame({
                'Ticker': result['tickers'],
                'Weight': result['weights'],
                'Expected Return': returns.mean() * 252,
                'Contribution': individual_contributions
            }).sort_values('Contribution', ascending=False)
              # Filter significant contributions
            contribution_df = contribution_df[contribution_df['Weight'] > 0.01]
            
            st.dataframe(
                contribution_df.style.format({
                    'Weight': '{:.1%}',
                    'Expected Return': '{:.1%}',
                    'Contribution': '{:.2%}'
                }),
                width='stretch',
                hide_index=True
            )

if __name__ == "__main__":
    main()
