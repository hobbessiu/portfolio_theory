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
from utils import RiskMetrics

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.25rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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
        st.error(f"Error fetching SPY data: {e}")
        # Return dummy data to prevent crashes
        dummy_returns = pd.Series([0.0004] * 252, name='SPY')
        return {
            'returns': dummy_returns,
            'cumulative_returns': (1 + dummy_returns).cumprod(),
            'annual_return': 0.10,  # Typical S&P 500 return
            'volatility': 0.16,     # Typical S&P 500 volatility
            'sharpe_ratio': 0.5
        }

@st.cache_data
def calculate_equal_weight_performance(data, tickers):
    """Calculate equal-weight portfolio performance for comparison."""
    try:
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
        textposition='auto',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        )
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_performance_comparison_chart(portfolio_cumulative, benchmark_cumulative, equal_weight_cumulative):
    """Create performance comparison chart."""
    fig = go.Figure()
    
    # Portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_cumulative.index,
        y=portfolio_cumulative.values,
        mode='lines',
        name='Optimized Portfolio',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Benchmark performance
    fig.add_trace(go.Scatter(
        x=benchmark_cumulative.index,
        y=benchmark_cumulative.values,
        mode='lines',
        name='S&P 500 (SPY)',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Equal weight performance
    fig.add_trace(go.Scatter(
        x=equal_weight_cumulative.index,
        y=equal_weight_cumulative.values,
        mode='lines',
        name='Equal Weight',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Portfolio Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_risk_return_scatter(results_df):
    """Create risk-return scatter plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['volatility'],
        y=results_df['return'],
        mode='markers+text',
        text=results_df['strategy'],
        textposition="top center",
        marker=dict(
            size=15,
            color=results_df['sharpe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Return: %{y:.2%}<br>'
            'Volatility: %{x:.2%}<br>'
            'Sharpe: %{marker.color:.2f}'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        height=400
    )
    
    return fig

def create_efficient_frontier(optimizer, returns):
    """
    Create efficient frontier plot showing both theoretical and practical frontiers.
    
    This addresses the issue where individual assets appear "outside" the efficient frontier
    by showing two frontiers:
    1. Theoretical (unconstrained) - what MPT theory predicts
    2. Practical (constrained) - what you get with diversification limits
    """
    try:
        target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, 30)
        
        # Generate THEORETICAL efficient frontier (unconstrained - true MPT)
        theoretical_portfolios = []
        for target_return in target_returns:
            try:
                result = optimizer.optimize_portfolio_unconstrained(returns, target_return=target_return)
                if result['status'] == 'optimal':
                    theoretical_portfolios.append({
                        'return': result['stats']['return'],
                        'volatility': result['stats']['volatility'],
                        'sharpe_ratio': result['stats']['sharpe_ratio']
                    })
            except:
                continue
        
        # Generate PRACTICAL efficient frontier (with diversification constraints)
        practical_portfolios = []
        for target_return in target_returns:
            try:
                result = optimizer.optimize_portfolio(returns, target_return=target_return, max_weight=0.20)
                if result['status'] == 'optimal':
                    practical_portfolios.append({
                        'return': result['stats']['return'],
                        'volatility': result['stats']['volatility'],
                        'sharpe_ratio': result['stats']['sharpe_ratio']
                    })
            except:
                continue
        
        if not practical_portfolios and not theoretical_portfolios:
            return None
        
        fig = go.Figure()
        
        # Theoretical efficient frontier (unconstrained - true MPT)
        if theoretical_portfolios:
            theoretical_df = pd.DataFrame(theoretical_portfolios)
            fig.add_trace(go.Scatter(
                x=theoretical_df['volatility'],
                y=theoretical_df['return'],
                mode='lines',
                name='📏 Theoretical Frontier (True MPT)',
                line=dict(color='#2E86AB', width=2, dash='dot'),
                hovertemplate=(
                    '<b>Theoretical Frontier</b><br>'
                    'Return: %{y:.2%}<br>'
                    'Volatility: %{x:.2%}<br>'
                    '<i>Allows up to 100% in single asset</i><br>'
                    '<extra></extra>'
                )
            ))
        
        # Practical efficient frontier (with 20% max constraint)
        if practical_portfolios:
            practical_df = pd.DataFrame(practical_portfolios)
            fig.add_trace(go.Scatter(
                x=practical_df['volatility'],
                y=practical_df['return'],
                mode='lines+markers',
                name='🛡️ Practical Frontier (Max 20% per stock)',
                line=dict(color='#C73E1D', width=3),
                marker=dict(size=4, color='#C73E1D'),
                hovertemplate=(
                    '<b>Practical Frontier</b><br>'
                    'Return: %{y:.2%}<br>'
                    'Volatility: %{x:.2%}<br>'
                    '<i>Diversification constraints applied</i><br>'
                    '<extra></extra>'
                )
            ))
            
            # Your optimal portfolio (from practical frontier)
            max_sharpe_idx = practical_df['sharpe_ratio'].idxmax()
            fig.add_trace(go.Scatter(
                x=[practical_df.loc[max_sharpe_idx, 'volatility']],
                y=[practical_df.loc[max_sharpe_idx, 'return']],
                mode='markers',
                name='⭐ Your Optimal Portfolio',
                marker=dict(size=15, color='#C73E1D', symbol='star'),
                hovertemplate=(
                    '<b>Your Optimal Portfolio</b><br>'
                    'Expected Return: %{y:.2%}<br>'
                    'Volatility: %{x:.2%}<br>'
                    f'Sharpe Ratio: {practical_df.loc[max_sharpe_idx, "sharpe_ratio"]:.2f}<br>'
                    '<i>Maximum Sharpe ratio with constraints</i><br>'
                    '<extra></extra>'
                )
            ))
        
        # Individual assets - these may appear "outside" the practical frontier
        individual_returns = returns.mean() * 252
        individual_volatility = returns.std() * np.sqrt(252)
        
        # Color code individual assets by performance
        individual_sharpe = (individual_returns - 0.02) / individual_volatility
        
        fig.add_trace(go.Scatter(
            x=individual_volatility,
            y=individual_returns,
            mode='markers+text',
            name='💎 Individual Assets',
            marker=dict(
                size=12, 
                color=individual_sharpe,
                colorscale='Viridis',
                symbol='diamond',
                showscale=True,
                colorbar=dict(title="Individual<br>Sharpe Ratio", x=1.02)
            ),
            text=returns.columns,
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Expected Return: %{y:.2%}<br>'
                'Volatility: %{x:.2%}<br>'
                'Individual Sharpe: %{marker.color:.2f}<br>'
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title="📈 Efficient Frontier: Theory vs Practice",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Annual Return",
            showlegend=True,
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font=dict(size=16, color='#2E86AB'),
            annotations=[
                dict(
                    text="💡 If individual assets appear above the red line,<br>it means diversification constraints limit optimal allocation",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, 
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    align='left'
                )
            ]
        )
        
        # Add grid
        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', gridwidth=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating efficient frontier: {str(e)}")
        return None

def main():
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
    
    # Info box explaining Modern Portfolio Theory
    st.info("""
    **What is Modern Portfolio Theory (MPT)?**

    Modern Portfolio Theory is a framework for building investment portfolios that balance risk and return. It helps investors choose a mix of assets that aims to maximize returns for a given level of risk, or minimize risk for a given level of expected return. MPT uses diversification to reduce risk by combining assets that behave differently.

    For beginners: MPT is a useful starting point for understanding how to diversify and manage risk, but it’s important to be aware of its assumptions and limitations.
    """)
    
    st.info("""
**Modern Portfolio Theory (MPT) & Sharpe Ratio - Simple Guide**

Modern Portfolio Theory helps you build a portfolio that balances risk and reward by mixing different investments. The Sharpe Ratio shows how much extra return you get for the risk you take. Both are useful tools for comparing investments and managing risk.

**Key Assumptions:**
- Past data can help predict future returns
- Markets are fair and prices reflect all info
- Risk is measured by ups and downs (volatility)
- Diversification lowers risk

**Limitations & Tips:**
- Relies on historical data, which may not predict future performance
- Assumes returns are normally distributed and markets are efficient
- Sensitive to estimation errors in returns and covariances
- May not account for real-world constraints (taxes, liquidity, transaction costs)
- Returns, risks, and correlations change over time
- Use these tools as a starting point, but always do extra research and consider your own goals
""")
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    # Add beginner-friendly help
    with st.sidebar.expander("❓ Help for Beginners", expanded=False):
        st.markdown("""
        **🎯 Quick Guide to Settings:**
        
        **Top N Stocks**: How many companies to include
        - More stocks = more diversification
        - Fewer stocks = simpler portfolio
        - **Tip**: Start with 20-30 stocks
        
        **Historical Period**: How far back to analyze
        - Longer = more data, captures different market cycles
        - Shorter = more recent market conditions
        - **Tip**: 3-5 years is a good balance
        
        **Risk-Free Rate**: Safe investment return (like T-Bills)
        - Used to calculate risk-adjusted returns
        - Currently ~4-5% in 2025
        - **Tip**: Use 2-5% for realistic analysis
        
        **Transaction Cost**: Trading fees per transaction
        - Includes broker fees and slippage
        - Typical: 0.1% for most brokers
        - **Tip**: Lower is better (use discount brokers!)
        
        **Max Weight per Asset**: Maximum % in one stock
        - Lower = more diversification, safer
        - Higher = can concentrate in best performers
        - **Tip**: 15-20% prevents over-concentration
        
        **Minimum Positions**: Least number of stocks to hold
        - Ensures you don't put all eggs in one basket
        - More positions = more diversification
        - **Tip**: At least 5-10 stocks for safety
        """)
    
    # Configuration parameters
    top_n_stocks = st.sidebar.slider("Top N S&P 500 Stocks", min_value=10, max_value=100, value=30, step=5,
                                      help="Number of largest S&P 500 companies to consider. More stocks = better diversification.")
    time_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "3y", "5y", "10y"], index=3,
                                        help="How many years of price history to analyze. Longer periods capture more market conditions.")
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                                              help="Expected return from safe investments (e.g., Treasury Bills). Used to calculate Sharpe ratio.") / 100
    transaction_cost = st.sidebar.number_input("Transaction Cost (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.01,
                                                help="Cost per trade as % of transaction value. Includes broker fees and market impact.") / 100
    
    # Diversification controls
    st.sidebar.markdown("### 🎯 Diversification")
    max_weight = st.sidebar.slider("Max Weight per Asset (%)", min_value=5, max_value=100, value=20, step=5,
                                     help="Maximum percentage allocated to any single stock. Lower values force more diversification.") / 100
    min_positions = st.sidebar.number_input("Minimum Positions", min_value=1, max_value=100, value=5,
                                             help="Minimum number of stocks to hold. Ensures portfolio isn't too concentrated.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Quick Actions")
    run_optimization = st.sidebar.button("🚀 Run Portfolio Optimization", type="primary")
    
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
        st.header("📈 Modern Portfolio Theory Optimization")
        
        # Portfolio Wizard - Step-by-step guide
        with st.expander("🧙‍♂️ Portfolio Creation Wizard", expanded=False):
            st.markdown("""
            ### Welcome to the Portfolio Optimization Wizard!
            Follow these steps to create your optimal portfolio:
            
            **Step 1: 📊 Data Configuration**
            - Adjust the number of stocks in the sidebar (10-100)
            - Select your preferred time period for historical analysis
            - Set the risk-free rate (current default: US Treasury rate)
            
            **Step 2: 🚀 Run Optimization**
            - Click "Run Portfolio Optimization" in the sidebar
            - Our algorithm will find the optimal risk-return balance
            - Review the performance metrics and allocation
            
            **Step 3: 📊 Analyze Results**
            - Check the expected return, volatility, and Sharpe ratio
            - Review the portfolio allocation pie chart
            - Examine the top holdings table
            
            **Step 4: 🧪 Test with Backtesting**
            - Go to the "Backtesting" tab
            - Choose between Fixed Weights or Walk-Forward Analysis
            - Compare your portfolio against benchmarks
            
            **Step 5: 🎲 Risk Analysis**
            - Use Monte Carlo simulation to understand potential outcomes
            - Review Value at Risk (VaR) and Expected Shortfall metrics
            - Analyze the distribution of possible returns
            """)
            
            # Quick action buttons
            wizard_col1, wizard_col2, wizard_col3 = st.columns(3)
            
            with wizard_col1:
                if st.button("📊 Load Sample Data", help="Load 30 top S&P 500 stocks for 5 years"):
                    st.session_state.wizard_step = 1
                    st.rerun()
            
            with wizard_col2:
                if st.button("🚀 Quick Optimize", help="Run optimization with current settings"):
                    st.session_state.wizard_step = 2
                    st.rerun()
            
            with wizard_col3:
                if st.button("📈 View Tutorial", help="Open detailed tutorial"):
                    st.info("📚 **Tutorial**: Check out our comprehensive guide in the sidebar configuration!")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.info(f"**Stocks:** {len(tickers)}")
            st.info(f"**Data Period:** {time_period}")
            st.info(f"**Risk-Free Rate:** {risk_free_rate:.1%}")
        
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
                
                # Validate returns data
                if returns.empty or returns.isna().all().all():
                    error_handler.show_error('optimization_failed', 
                        "Unable to calculate returns from price data. Please check your data selection.")
                    return
                
                # Remove any remaining NaN values
                returns = returns.dropna()
                
                if len(returns) < 30:
                    error_handler.show_error('optimization_failed', 
                        f"Insufficient valid data points ({len(returns)} days). Need at least 30 days of clean data.")
                    return
                
                # Step 2: Run optimization
                with progress_placeholder:
                    st.markdown("""
                    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                        <div class="loading-spinner"></div>
                        <span style="margin-left: 10px, font-weight: 500;">Running MPT optimization algorithm...</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                result = optimizer.optimize_portfolio(returns, max_weight=max_weight, min_positions=min_positions)
                
                # Clear loading
                progress_placeholder.empty()
                
                if result['status'] == 'optimal':
                    # Validate result stats
                    if any(np.isnan(v) for v in result['stats'].values() if isinstance(v, (int, float))):
                        error_handler.show_error('optimization_failed', 
                            "Optimization produced invalid statistics (NaN values). This may be due to insufficient data or extreme correlations.")
                        return
                    
                    st.session_state.optimization_results = result
                    st.success("✅ Portfolio optimization completed successfully!")
                else:
                    error_handler.show_error('optimization_failed', 
                        f"Optimization failed: {result.get('message', 'Unknown error')}")
                    return
                    
            except Exception as e:
                error_handler.show_error('optimization_failed', f"Optimization error: {str(e)}")
                return
        
        # Display results if available
        if 'optimization_results' in st.session_state:
            result = st.session_state.optimization_results
            
            # Enhanced performance metrics with styled cards
            st.subheader("📊 Portfolio Performance")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                return_color = "🟢" if result['stats']['return'] > 0 else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{return_color} Expected Return</h4>
                    <h2>{result['stats']['return']:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col2:
                vol_color = "🟡" if result['stats']['volatility'] < 0.2 else "🟠"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{vol_color} Volatility</h4>
                    <h2>{result['stats']['volatility']:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col3:
                sharpe_color = "🟢" if result['stats']['sharpe_ratio'] > 1 else "🟡" if result['stats']['sharpe_ratio'] > 0.5 else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{sharpe_color} Sharpe Ratio</h4>
                    <h2>{result['stats']['sharpe_ratio']:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col4:
                positions = np.sum(result['weights'] > 0.01)
                pos_color = "🎯"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{pos_color} Positions</h4>
                    <h2>{positions}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Portfolio allocation chart
            st.subheader("🥧 Portfolio Allocation")
            # Use tickers from optimization result to match the weights
            result_tickers = result.get('tickers', tickers)
            fig_pie = create_portfolio_pie_chart(result['weights'], result_tickers)
            st.plotly_chart(fig_pie, width='stretch')
            
            # Efficient Frontier
            st.subheader("🎯 Efficient Frontier")
            
            # Calculate returns for efficient frontier
            returns = optimizer.calculate_returns(data)
            
            # Store returns data for efficient frontier calculation
            if 'returns_data' not in st.session_state:
                st.session_state.returns_data = returns
            
            with st.spinner("Generating efficient frontier..."):
                ef_chart = create_efficient_frontier(optimizer, returns)
                
                if ef_chart:
                    st.plotly_chart(ef_chart, width='stretch')
                    
                    st.info("""
                    📊 **Understanding the Efficient Frontier:**
                    
                    **🔵 Theoretical Frontier (Dotted Blue)**: True MPT frontier allowing up to 100% in any single asset
                    
                    **🔴 Practical Frontier (Solid Red)**: Your constrained frontier with max 20% per stock for diversification
                    
                    **💎 Individual Assets**: Colored by their Sharpe ratios - high-performing assets (NVDA, AVGO) may appear "above" the practical frontier
                    
                    **⭐ Your Portfolio**: Optimal allocation within diversification constraints
                    
                    **🚨 Key Insight**: If individual assets appear above the red line, it means diversification constraints prevent concentrating in these high performers. This is normal and expected for risk management!
                    """)
                else:
                    st.warning("⚠️ Could not generate efficient frontier. This may happen with insufficient data or optimization constraints.")
            
            # Top holdings table
            st.subheader("📋 Top Holdings")
            # Use tickers from result if available, otherwise use the selected tickers
            result_tickers = result.get('tickers', tickers)
            weights_df = pd.DataFrame({
                'Ticker': result_tickers,
                'Weight': result['weights']
            }).sort_values('Weight', ascending=False)
            
            # Show only holdings with non-zero weight (>0.01%)
            significant_holdings = weights_df[weights_df['Weight'] > 0.0001].head(15)
            
            # Add formatting
            display_df = significant_holdings.copy()
            display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
            
            st.dataframe(display_df, width='stretch')
            
            # Show summary
            total_positions = len(significant_holdings)
            st.info(f"**Total Positions:** {total_positions} | **Largest Position:** {weights_df.iloc[0]['Weight']:.2%} ({weights_df.iloc[0]['Ticker']})")
    
    # Tab 2: Backtesting
    with tab2:
        st.header("📊 Portfolio Backtesting")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first in the 'Portfolio Optimization' tab.")
        else:
            # Backtesting configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                backtest_method = st.selectbox(
                    "Backtesting Method",
                    ["Fixed Weights", "Walk-Forward Analysis"],
                    help="Choose between fixed weights or dynamic rebalancing"
                )
                
            with col2:
                rebalance_freq = st.selectbox(
                    "Rebalancing Frequency",
                    ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
                    index=1
                )
            
            with col3:
                equal_weight_stocks = st.selectbox(
                    "Equal Weight Benchmark",
                    ["All Selected Stocks", "Top 10", "Top 20", "Top 30"],
                    help="Choose how many stocks to include in the equal weight benchmark"
                )
            
            freq_map = {
                "Monthly": "M",
                "Quarterly": "Q",
                "Semi-Annual": "6M",
                "Annual": "A"
            }
            
            st.info(f"""
            **Backtesting Configuration:**
            • **Method**: {backtest_method}
            • **Rebalancing**: {rebalance_freq}
            • **Transaction Cost**: {transaction_cost:.2%}
            • **Benchmark**: S&P 500 (SPY)
            • **Equal Weight**: Market benchmark (SPY)""")
            
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
                    result = st.session_state.optimization_results
                    
                    # Get the tickers and data that match the optimization result
                    result_tickers = result.get('tickers', tickers)
                    result_data = data[result_tickers]
                    
                    if backtest_method == "Fixed Weights":
                        # Use fixed weights from initial optimization
                        portfolio_performance = backtest_engine.backtest_portfolio(
                            result['weights'], result_data, freq_map[rebalance_freq]
                        )
                        strategy_name = "Optimized (Fixed)"
                    else:
                        # Use walk-forward analysis with dynamic re-optimization
                        # Pass the same constraints used in initial optimization
                        portfolio_performance = backtest_engine.backtest_portfolio_walk_forward(
                            result_data, 
                            freq_map[rebalance_freq],
                            max_weight=max_weight,
                            min_positions=min_positions
                        )
                        strategy_name = "Optimized (Walk-Forward)"
                    
                    progress_placeholder.empty()
                    
                    # Determine equal weight tickers based on selection
                    equal_weight_map = {
                        "All Selected Stocks": len(tickers),
                        "Top 10": min(10, len(tickers)),
                        "Top 20": min(20, len(tickers)),
                        "Top 30": min(30, len(tickers))
                    }
                    n_equal_weight = equal_weight_map[equal_weight_stocks]
                    equal_weight_tickers = tickers[:n_equal_weight]
                    equal_weight_data = data[equal_weight_tickers]
                    
                    # Calculate benchmarks
                    benchmark_performance = calculate_benchmark_performance(time_period)
                    equal_weight_performance = calculate_equal_weight_performance(equal_weight_data, equal_weight_tickers)
                    
                    # Calculate performance metrics from portfolio returns
                    portfolio_returns = portfolio_performance['returns']
                    portfolio_metrics = {
                        'returns': portfolio_returns,
                        'cumulative_returns': portfolio_performance['cumulative_returns'],
                        'annual_return': portfolio_returns.mean() * 252,
                        'volatility': portfolio_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))
                    }
                    
                    # Store results
                    st.session_state.backtest_results = {
                        'portfolio': portfolio_metrics,
                        'benchmark': benchmark_performance,
                        'equal_weight': equal_weight_performance,
                        'strategy_name': strategy_name,
                        'equal_weight_tickers': equal_weight_tickers,
                        'equal_weight_n': n_equal_weight,
                        'portfolio_performance': portfolio_performance,  # Store full performance data
                        'backtest_method': backtest_method
                    }
                    
                    st.success("✅ Backtesting completed successfully!")
                    
                except Exception as e:
                    error_handler.show_error('optimization_failed', f"Backtesting error: {str(e)}")
                    return
            
            # Display backtest results
            if 'backtest_results' in st.session_state:
                results = st.session_state.backtest_results
                
                # Performance comparison chart
                st.subheader("📈 Performance Comparison")
                fig_perf = create_performance_comparison_chart(
                    results['portfolio']['cumulative_returns'],
                    results['benchmark']['cumulative_returns'],
                    results['equal_weight']['cumulative_returns']
                )
                st.plotly_chart(fig_perf, width='stretch')
                
                # Performance metrics comparison
                st.subheader("📊 Performance Metrics Comparison")
                
                # Calculate max drawdown for each strategy
                portfolio_max_dd = RiskMetrics.calculate_max_drawdown(results['portfolio']['returns'])
                benchmark_max_dd = RiskMetrics.calculate_max_drawdown(results['benchmark']['returns'])
                equal_weight_max_dd = RiskMetrics.calculate_max_drawdown(results['equal_weight']['returns'])
                
                # Calculate Sortino ratio for each strategy
                portfolio_sortino = RiskMetrics.calculate_sortino_ratio(results['portfolio']['returns'], risk_free_rate)
                benchmark_sortino = RiskMetrics.calculate_sortino_ratio(results['benchmark']['returns'], risk_free_rate)
                equal_weight_sortino = RiskMetrics.calculate_sortino_ratio(results['equal_weight']['returns'], risk_free_rate)
                
                # Calculate Calmar ratio (annual return / max drawdown)
                portfolio_calmar = results['portfolio']['annual_return'] / abs(portfolio_max_dd) if portfolio_max_dd != 0 else 0
                benchmark_calmar = results['benchmark']['annual_return'] / abs(benchmark_max_dd) if benchmark_max_dd != 0 else 0
                equal_weight_calmar = results['equal_weight']['annual_return'] / abs(equal_weight_max_dd) if equal_weight_max_dd != 0 else 0
                
                # Calculate cumulative return
                portfolio_cum_return = results['portfolio']['cumulative_returns'].iloc[-1] - 1
                benchmark_cum_return = results['benchmark']['cumulative_returns'].iloc[-1] - 1
                equal_weight_cum_return = results['equal_weight']['cumulative_returns'].iloc[-1] - 1
                
                metrics_data = {
                    'Strategy': [results['strategy_name'], 'S&P 500 (SPY)', 'Equal Weight'],
                    'Cumulative Return': [portfolio_cum_return, benchmark_cum_return, equal_weight_cum_return],
                    'Annual Return': [
                        results['portfolio']['annual_return'],
                        results['benchmark']['annual_return'],
                        results['equal_weight']['annual_return']
                    ],
                    'Volatility': [
                        results['portfolio']['volatility'],
                        results['benchmark']['volatility'],
                        results['equal_weight']['volatility']
                    ],
                    'Max Drawdown': [portfolio_max_dd, benchmark_max_dd, equal_weight_max_dd],
                    'Sharpe Ratio': [
                        results['portfolio']['sharpe_ratio'],
                        results['benchmark']['sharpe_ratio'],
                        results['equal_weight']['sharpe_ratio']
                    ],
                    'Sortino Ratio': [portfolio_sortino, benchmark_sortino, equal_weight_sortino],
                    'Calmar Ratio': [portfolio_calmar, benchmark_calmar, equal_weight_calmar]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                
                # Format the dataframe
                display_df = metrics_df.copy()
                display_df['Cumulative Return'] = display_df['Cumulative Return'].map('{:.2%}'.format)
                display_df['Annual Return'] = display_df['Annual Return'].map('{:.2%}'.format)
                display_df['Volatility'] = display_df['Volatility'].map('{:.2%}'.format)
                display_df['Max Drawdown'] = display_df['Max Drawdown'].map('{:.2%}'.format)
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].map('{:.2f}'.format)
                display_df['Sortino Ratio'] = display_df['Sortino Ratio'].map('{:.2f}'.format)
                display_df['Calmar Ratio'] = display_df['Calmar Ratio'].map('{:.2f}'.format)
                
                st.dataframe(display_df, width='stretch')
                
                # Add explanation of metrics
                with st.expander("📖 Understanding Performance Metrics"):
                    st.markdown("""
                    **Cumulative Return**: Total gain/loss over the entire period
                    - Shows overall investment growth
                    - Example: 50% means $10,000 became $15,000
                    
                    **Annual Return**: Average yearly return (annualized)
                    - Standardized measure for comparing different time periods
                    - Higher is better for long-term wealth building
                    
                    **Volatility (Risk)**: Standard deviation of returns
                    - Measures price fluctuation and uncertainty
                    - Lower is better for risk-averse investors
                    
                    **Max Drawdown**: Largest peak-to-trough decline
                    - Shows worst-case loss experience
                    - Important for understanding downside risk
                    - Lower (less negative) is better
                    
                    **Sharpe Ratio**: Risk-adjusted return (excess return / volatility)
                    - Measures return per unit of total risk
                    - Higher is better (>1 is good, >2 is very good)
                    
                    **Sortino Ratio**: Risk-adjusted return using downside deviation
                    - Similar to Sharpe but only penalizes downside volatility
                    - Better measure for investors who don't mind upside volatility
                    - Higher is better
                    
                    **Calmar Ratio**: Annual return / max drawdown
                    - Measures return relative to worst drawdown
                    - Higher is better (shows strong recovery from losses)
                    - Useful for assessing risk of ruin
                    """)
                
                # Equal Weight Composition
                st.subheader("⚖️ Equal Weight Portfolio Composition")
                if 'equal_weight_tickers' in results and 'equal_weight_n' in results:
                    eq_tickers = results['equal_weight_tickers']
                    eq_n = results['equal_weight_n']
                    eq_weight = 1.0 / eq_n
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create dataframe of equal weight holdings - showing tickers only
                        # since all weights are identical (shown in the metrics)
                        eq_df = pd.DataFrame({
                            '#': range(1, eq_n + 1),
                            'Ticker': eq_tickers
                        })
                        
                        st.dataframe(eq_df, hide_index=True, width='stretch')
                    
                    with col2:
                        st.metric("Number of Stocks", eq_n)
                        st.metric("Weight per Stock", f"{eq_weight:.2%}")
                        st.info(f"""
                        **Equal Weight Strategy:**
                        
                        Simple 1/N allocation across {eq_n} stocks. 
                        Each position gets exactly {eq_weight:.2%} of the portfolio.
                        
                        This benchmark helps evaluate if optimization adds value over naive diversification.
                        """)
                
                # Walk-Forward Rebalancing Details (only shown for walk-forward method)
                if results.get('backtest_method') == "Walk-Forward Analysis":
                    st.subheader("🔄 Walk-Forward Rebalancing History")
                    
                    portfolio_perf = results.get('portfolio_performance')
                    if portfolio_perf is not None and hasattr(portfolio_perf, 'attrs'):
                        compositions = portfolio_perf.attrs.get('portfolio_compositions', [])
                        
                        if compositions:
                            # Calculate diagnostic metrics
                            total_turnover = 0
                            for idx in range(1, len(compositions)):
                                prev_comp = compositions[idx-1]['composition']
                                curr_comp = compositions[idx]['composition']
                                all_tickers = set(prev_comp.keys()) | set(curr_comp.keys())
                                turnover = sum(abs(curr_comp.get(t, 0) - prev_comp.get(t, 0)) for t in all_tickers) / 2
                                total_turnover += turnover
                            
                            avg_turnover = total_turnover / max(len(compositions) - 1, 1) if len(compositions) > 1 else 0
                            
                            # Calculate average portfolio concentration
                            avg_concentration = np.mean([
                                sum(w**2 for w in comp['composition'].values()) 
                                for comp in compositions
                            ])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Rebalances", len(compositions))
                            with col2:
                                st.metric("Avg Turnover per Rebalance", f"{avg_turnover:.1%}",
                                         help="Average portfolio turnover at each rebalancing. Higher = more drastic changes")
                            with col3:
                                st.metric("Avg Concentration", f"{avg_concentration:.3f}",
                                         help="Herfindahl index: Lower = more diversified. 1.0 = all in one stock")
                            with col4:
                                # Calculate time between rebalances
                                if len(compositions) > 1:
                                    days_between = (compositions[-1]['date'] - compositions[0]['date']).days / (len(compositions) - 1)
                                    st.metric("Avg Days Between", f"{days_between:.0f}")
                                else:
                                    st.metric("Avg Days Between", "N/A")
                            
                            st.info(f"""
                            **Walk-Forward Analysis Insights**:
                            
                            At each rebalancing date, the optimizer used only historical data available up to that point, 
                            preventing look-ahead bias and simulating real-world portfolio management.
                            
                            **🔬 Survivorship-Bias-Free Analysis**: The optimizer only considers stocks with sufficient historical 
                            data at each rebalancing date. This excludes:
                            - Stocks that weren't publicly traded yet
                            - Stocks with insufficient trading history
                            - Stocks with excessive missing data
                            
                            This simulates realistic portfolio construction at each point in time, though it approximates historical 
                            S&P 500 composition rather than using exact historical index membership (which requires proprietary data).
                            
                            **Why different frequencies perform differently:**
                            - **Monthly**: More frequent rebalancing but may overfit to recent noise with limited lookback data
                            - **Quarterly**: Balances stability with adaptability; aligns with earnings cycles
                            - **Semi-Annual/Yearly**: Fewer rebalances may miss important market shifts or have insufficient data points
                            
                            High turnover suggests the optimizer is chasing recent performance (possible overfitting).
                            Low turnover suggests stable, consistent allocations.
                            """)
                            
                            # Show optimization quality summary
                            failed_optimizations = sum(1 for comp in compositions if not comp.get('optimization_success', True))
                            if failed_optimizations > 0:
                                st.warning(f"⚠️ **Warning**: {failed_optimizations} out of {len(compositions)} optimizations failed and fell back to equal weights. "
                                          f"This may indicate insufficient historical data or numerical issues. Consider using a longer time period or quarterly/semi-annual rebalancing.")
                            
                            # Show historical data availability for each rebalancing
                            with st.expander("📊 Historical Data Availability at Each Rebalancing"):
                                history_data = [(comp['date'].strftime('%Y-%m-%d'), comp.get('history_days', 0)) for comp in compositions if comp.get('history_days', 0) > 0]
                                if history_data:
                                    history_df = pd.DataFrame(history_data, columns=['Date', 'Days of History'])
                                    st.dataframe(history_df, use_container_width=True)
                                    
                                    min_history = min(h[1] for h in history_data) if history_data else 0
                                    avg_history = np.mean([h[1] for h in history_data]) if history_data else 0
                                    
                                    st.info(f"""
                                    **Data Availability Analysis:**
                                    - Minimum history used: {min_history} days (~{min_history/252:.1f} years)
                                    - Average history used: {avg_history:.0f} days (~{avg_history/252:.1f} years)
                                    
                                    **Rule of Thumb**: For robust optimization, you typically want at least 252 days (1 year) of history, 
                                    preferably 2-3 years. Less data may lead to overfitting where the optimizer latches onto recent noise rather than true patterns.
                                    """)
                            
                            # Create expandable sections for each rebalancing
                            for idx, rebal in enumerate(compositions, 1):
                                date = rebal['date']
                                composition = rebal['composition']
                                history_days = rebal.get('history_days', 0)
                                opt_success = rebal.get('optimization_success', True)
                                stocks_available = rebal.get('stocks_available', len(composition))
                                
                                # Sort by weight descending
                                sorted_composition = sorted(composition.items(), key=lambda x: x[1], reverse=True)
                                
                                # Calculate changes from previous rebalancing
                                prev_composition = compositions[idx-2]['composition'] if idx > 1 else {}
                                
                                # Create title with diagnostic info
                                title = f"📅 Rebalancing #{idx} - {date.strftime('%B %d, %Y')}"
                                if history_days > 0:
                                    title += f" ({history_days} days history, {stocks_available} stocks available)"
                                if not opt_success:
                                    title += " ⚠️ Optimization Failed - Using Equal Weights"
                                
                                with st.expander(title, expanded=(idx <= 2)):
                                    # Show changes from previous rebalancing (if not first)
                                    if idx > 1:
                                        st.markdown("##### 📊 Changes from Previous Rebalancing")
                                        
                                        # Identify additions, removals, and changes
                                        all_tickers = set(composition.keys()) | set(prev_composition.keys())
                                        additions = []
                                        removals = []
                                        increases = []
                                        decreases = []
                                        
                                        for ticker in all_tickers:
                                            new_weight = composition.get(ticker, 0)
                                            old_weight = prev_composition.get(ticker, 0)
                                            change = new_weight - old_weight
                                            
                                            if old_weight == 0 and new_weight > 0:
                                                additions.append((ticker, new_weight))
                                            elif old_weight > 0 and new_weight == 0:
                                                removals.append((ticker, old_weight))
                                            elif abs(change) > 0.005:  # More than 0.5% change
                                                if change > 0:
                                                    increases.append((ticker, old_weight, new_weight, change))
                                                else:
                                                    decreases.append((ticker, old_weight, new_weight, change))
                                        
                                        # Display changes in a user-friendly way
                                        change_col1, change_col2 = st.columns(2)
                                        
                                        with change_col1:
                                            if additions:
                                                st.markdown("**✅ Added to Portfolio:**")
                                                for ticker, weight in sorted(additions, key=lambda x: x[1], reverse=True):
                                                    st.markdown(f"- **{ticker}**: {weight:.2%}")
                                            
                                            if increases:
                                                st.markdown("**📈 Increased Positions:**")
                                                for ticker, old_w, new_w, change in sorted(increases, key=lambda x: x[3], reverse=True):
                                                    st.markdown(f"- **{ticker}**: {old_w:.2%} → {new_w:.2%} <span style='color:green'>(+{change:.2%})</span>", unsafe_allow_html=True)
                                        
                                        with change_col2:
                                            if removals:
                                                st.markdown("**❌ Removed from Portfolio:**")
                                                for ticker, weight in sorted(removals, key=lambda x: x[1], reverse=True):
                                                    st.markdown(f"- **{ticker}**: {weight:.2%}")
                                            
                                            if decreases:
                                                st.markdown("**📉 Decreased Positions:**")
                                                for ticker, old_w, new_w, change in sorted(decreases, key=lambda x: x[3]):
                                                    st.markdown(f"- **{ticker}**: {old_w:.2%} → {new_w:.2%} <span style='color:red'>({change:.2%})</span>", unsafe_allow_html=True)
                                        
                                        if not (additions or removals or increases or decreases):
                                            st.info("No significant changes from previous rebalancing.")
                                        
                                        st.divider()
                                    
                                    # Portfolio composition after rebalancing
                                    st.markdown("##### 💼 Portfolio Composition After Rebalancing")
                                    
                                    # Create a two-column layout
                                    col1, col2 = st.columns([3, 2])
                                    
                                    with col1:
                                        # Portfolio composition table
                                        comp_df = pd.DataFrame([
                                            {'#': i+1, 'Ticker': ticker, 'Weight': f"{weight:.2%}"}
                                            for i, (ticker, weight) in enumerate(sorted_composition)
                                        ])
                                        st.dataframe(comp_df, hide_index=True, use_container_width=True)
                                    
                                    with col2:
                                        # Summary metrics
                                        st.metric("Number of Holdings", len(composition))
                                        st.metric("Largest Position", f"{sorted_composition[0][1]:.2%}" if sorted_composition else "N/A")
                                        st.metric("Smallest Position", f"{sorted_composition[-1][1]:.2%}" if sorted_composition else "N/A")
                                        
                                        # Concentration measure
                                        weights_array = np.array(list(composition.values()))
                                        herfindahl = np.sum(weights_array ** 2)
                                        st.metric("Portfolio Concentration", f"{herfindahl:.3f}", 
                                                 help="Herfindahl index: 1.0 = concentrated in one stock, lower = more diversified")
                        else:
                            st.info("No rebalancing data available for this backtest.")
                
                # Risk-return scatter plot
                st.subheader("🎯 Risk-Return Analysis")
                results_df = pd.DataFrame({
                    'strategy': metrics_data['Strategy'],
                    'return': metrics_data['Annual Return'],
                    'volatility': metrics_data['Volatility'],
                    'sharpe_ratio': metrics_data['Sharpe Ratio']
                })
                
                fig_scatter = create_risk_return_scatter(results_df)
                st.plotly_chart(fig_scatter, width='stretch')
    
    # Tab 3: Monte Carlo Simulation
    with tab3:
        st.header("🎲 Monte Carlo Simulation")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first.")
        else:
            # Monte Carlo configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
            
            with col2:
                time_horizon = st.number_input("Time Horizon (Years)", min_value=1, max_value=30, value=5)
            
            with col3:
                confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
            
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000, max_value=10000000, value=10000, step=1000)
            
            if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
                try:
                    with st.spinner("Running Monte Carlo simulation..."):
                        # Initialize Monte Carlo simulator
                        returns = optimizer.calculate_returns(data)
                        result = st.session_state.optimization_results
                        
                        mc_simulator = MonteCarloSimulator(n_simulations=n_simulations)
                        
                        # Run simulation
                        simulation_results = mc_simulator.simulate_portfolio_paths(
                            weights=result['weights'],
                            returns=returns,
                            time_horizon=time_horizon * 252  # Convert years to trading days
                        )
                        
                        # Calculate risk metrics (scale by initial investment)
                        final_values = simulation_results[:, -1] * initial_investment
                        
                        # Value at Risk and Expected Shortfall
                        var_level = (100 - confidence_level) / 100
                        var = np.percentile(final_values, var_level * 100)
                        es = np.mean(final_values[final_values <= var])
                        
                        # Display results
                        st.success("✅ Monte Carlo simulation completed!")
                        
                        # Summary statistics
                        st.subheader("📊 Simulation Summary")
                        
                        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
                        
                        with sim_col1:
                            st.metric("Expected Value", f"${np.mean(final_values):,.0f}")
                        
                        with sim_col2:
                            st.metric("Standard Deviation", f"${np.std(final_values):,.0f}")
                        
                        with sim_col3:
                            st.metric(f"VaR ({confidence_level}%)", f"${var:,.0f}")
                        
                        with sim_col4:
                            st.metric("Expected Shortfall", f"${es:,.0f}")
                        
                        # Distribution plot
                        st.subheader("📈 Portfolio Value Distribution")
                        
                        fig_hist = go.Figure()
                        
                        fig_hist.add_trace(go.Histogram(
                            x=final_values,
                            nbinsx=50,
                            name='Portfolio Values',
                            opacity=0.7
                        ))
                        
                        # Add VaR line
                        fig_hist.add_vline(
                            x=var,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"VaR ({confidence_level}%)"
                        )
                        
                        fig_hist.update_layout(
                            title=f"Distribution of Portfolio Values after {time_horizon} Years",
                            xaxis_title="Portfolio Value ($)",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig_hist, width='stretch')
                        
                        # Percentile analysis
                        st.subheader("📊 Percentile Analysis")
                        
                        percentiles = [5, 10, 25, 50, 75, 90, 95]
                        percentile_values = [np.percentile(final_values, p) for p in percentiles]
                        
                        percentile_df = pd.DataFrame({
                            'Percentile': [f"{p}th" for p in percentiles],
                            'Portfolio Value': [f"${v:,.0f}" for v in percentile_values],
                            'Return': [f"{(v/initial_investment - 1)*100:.1f}%" for v in percentile_values]
                        })
                        
                        st.dataframe(percentile_df, width='stretch')
                        
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")
    
    # Tab 4: Performance Metrics
    with tab4:
        st.header("📋 Comprehensive Performance Analysis")
        
        if 'backtest_results' not in st.session_state:
            st.warning("⚠️ Please run backtesting first to see detailed performance metrics.")
        else:
            results = st.session_state.backtest_results
            portfolio_returns = results['portfolio']['returns']
            
            # Advanced risk metrics
            st.subheader("🎯 Advanced Risk Metrics")
            
            try:
                risk_calc = RiskMetrics()
                
                # Calculate all available metrics
                max_drawdown = risk_calc.calculate_max_drawdown(portfolio_returns)
                sortino_ratio = risk_calc.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                calmar_ratio = risk_calc.calculate_calmar_ratio(portfolio_returns)
                var_5 = risk_calc.calculate_var(portfolio_returns, confidence_level=0.05)
                cvar_5 = risk_calc.calculate_cvar(portfolio_returns, confidence_level=0.05)
                var_1 = risk_calc.calculate_var(portfolio_returns, confidence_level=0.01)
                cvar_1 = risk_calc.calculate_cvar(portfolio_returns, confidence_level=0.01)
                
                # Display all metrics in a grid
                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                
                with risk_col1:
                    st.metric("Maximum Drawdown", f"{max_drawdown:.2%}", 
                             help="Largest peak-to-trough decline in portfolio value")
                    st.metric("VaR (95%)", f"{var_5:.2%}",
                             help="Value at Risk: Maximum expected loss on 95% of days")
                
                with risk_col2:
                    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}",
                             help="Risk-adjusted return using downside deviation only")
                    st.metric("CVaR (95%)", f"{cvar_5:.2%}",
                             help="Conditional VaR: Average loss when losses exceed VaR")
                
                with risk_col3:
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}",
                             help="Annual return divided by maximum drawdown")
                    st.metric("VaR (99%)", f"{var_1:.2%}",
                             help="Value at Risk: Maximum expected loss on 99% of days")
                
                with risk_col4:
                    st.metric("Annual Return", f"{results['portfolio']['annual_return']:.2%}",
                             help="Average yearly return (annualized)")
                    st.metric("CVaR (99%)", f"{cvar_1:.2%}",
                             help="Conditional VaR: Average loss in worst 1% of days")
                
                # Add beginner-friendly explanation
                with st.expander("📖 Understanding Advanced Risk Metrics for Beginners"):
                    st.markdown("""
                    ### 📊 Risk & Return Metrics Explained
                    
                    **Maximum Drawdown (Max DD)**
                    - What it is: The biggest drop from a peak to a trough in your portfolio value
                    - Why it matters: Shows the worst loss you would have experienced
                    - Example: -20% means if you had $10,000 at peak, it dropped to $8,000 at worst
                    - **Lower is better** (less negative means smaller losses)
                    
                    **Sortino Ratio**
                    - What it is: Similar to Sharpe Ratio, but only counts downside volatility as risk
                    - Why it matters: Better for investors who don't mind upside volatility
                    - Calculation: (Return - Risk-Free Rate) / Downside Deviation
                    - **Higher is better** (>2 is excellent, >1 is good)
                    
                    **Calmar Ratio**
                    - What it is: Annual return divided by maximum drawdown
                    - Why it matters: Shows how much return you get per unit of worst-case risk
                    - Example: 1.5 means you earned 1.5% return for every 1% of max drawdown
                    - **Higher is better** (>1 is good, means you recover well from losses)
                    
                    **Value at Risk (VaR)**
                    - What it is: Maximum loss expected on X% of days (95% or 99%)
                    - Why it matters: Helps you understand typical bad days
                    - Example: VaR 95% = -2% means on 95% of days, you won't lose more than 2%
                    - **Less negative is better** (smaller typical losses)
                    
                    **Conditional VaR (CVaR) / Expected Shortfall**
                    - What it is: Average loss when losses are worse than VaR
                    - Why it matters: Shows how bad the really bad days are
                    - Example: CVaR 95% = -3% means when you lose more than VaR, average loss is 3%
                    - **Less negative is better** (indicates tail risk is controlled)
                    
                    **Annual Return**
                    - What it is: Average yearly return, standardized for comparison
                    - Why it matters: Shows wealth-building potential
                    - **Higher is better** (10% means you'd expect to grow $10,000 to $11,000 yearly)
                    
                    ### 💡 How to Use These Metrics
                    
                    1. **Compare drawdown across strategies**: Lower drawdown = easier to stick with during tough times
                    2. **Use Sortino if you like upside volatility**: Better than Sharpe for growth investors
                    3. **Check Calmar for recovery ability**: High Calmar means good bounce-back from losses
                    4. **VaR/CVaR for daily risk**: Helps set stop-losses and understand typical losses
                    5. **Balance return with risk**: High return with low drawdown is ideal
                    
                    ### ⚠️ Important Notes
                    - All metrics use historical data - past performance doesn't guarantee future results
                    - Use multiple metrics together for a complete picture
                    - Consider your personal risk tolerance and investment timeline
                    - These are tools for comparison, not predictions
                    """)
                
                # Rolling metrics
                st.subheader("📈 Rolling Performance Analysis")
                
                # Calculate rolling Sharpe ratio (3-month window)
                rolling_window = 63  # ~3 months
                rolling_sharpe = portfolio_returns.rolling(rolling_window).apply(
                    lambda x: (x.mean() * 252 - risk_free_rate) / (x.std() * np.sqrt(252))
                ).dropna()
                
                fig_rolling = go.Figure()
                
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='3-Month Rolling Sharpe Ratio',
                    line=dict(color='blue', width=2)
                ))
                
                fig_rolling.add_hline(
                    y=rolling_sharpe.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {rolling_sharpe.mean():.2f}"
                )
                
                fig_rolling.update_layout(
                    title="Rolling Sharpe Ratio (3-Month Window)",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    height=400
                )
                
                st.plotly_chart(fig_rolling, width='stretch')
                
                # Monthly returns heatmap
                st.subheader("🗓️ Monthly Returns Heatmap")
                
                monthly_returns = portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_pivot = monthly_returns.groupby([
                    monthly_returns.index.year,
                    monthly_returns.index.month
                ]).first().unstack()
                
                if len(monthly_returns_pivot) > 0:
                    # Convert to percentage and handle NaN values
                    heatmap_data = monthly_returns_pivot.values * 100
                    
                    # Create text with conditional formatting (hide NaN)
                    text_data = []
                    for row in heatmap_data:
                        text_row = []
                        for val in row:
                            if np.isnan(val):
                                text_row.append('')  # Empty string for NaN
                            else:
                                text_row.append(f'{val:.1f}%')
                        text_data.append(text_row)
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        y=monthly_returns_pivot.index,
                        colorscale='RdYlGn',
                        text=text_data,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False,
                        zmid=0  # Center colorscale at 0%
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Monthly Returns (%)",
                        xaxis_title="Month",
                        yaxis_title="Year",
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap, width='stretch')
                
            except ImportError:
                st.info("Advanced risk metrics require the utils module with RiskMetrics class.")
            except Exception as e:
                st.error(f"Error calculating advanced metrics: {str(e)}")

    # Sharpe Ratio info box
    st.subheader("📊 Portfolio Metrics")
    st.info("""
**What is Sharpe Ratio?**

The Sharpe Ratio measures how much extra return you receive for the risk you take. It is calculated as:

    (Portfolio Return - Risk-Free Rate) / Portfolio Volatility

A higher Sharpe Ratio means better risk-adjusted performance. For beginners: it helps you compare investments by showing how much reward you get for each unit of risk. A Sharpe Ratio above 1 is considered good, above 2 is excellent.
""")

if __name__ == "__main__":
    main()
