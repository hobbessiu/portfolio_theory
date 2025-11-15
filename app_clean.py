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
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    # Configuration parameters
    top_n_stocks = st.sidebar.slider("Top N S&P 500 Stocks", min_value=10, max_value=100, value=30, step=5)
    time_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "3y", "5y", "10y"], index=3)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
    transaction_cost = st.sidebar.number_input("Transaction Cost (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.01) / 100
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Actions")
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
            fig_pie = create_portfolio_pie_chart(result['weights'], tickers)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Top holdings table
            st.subheader("📋 Top Holdings")
            weights_df = pd.DataFrame({
                'Ticker': tickers,
                'Weight': result['weights']
            }).sort_values('Weight', ascending=False)
            
            # Show only significant holdings (>1%)
            significant_holdings = weights_df[weights_df['Weight'] > 0.01].head(10)
            significant_holdings['Weight'] = significant_holdings['Weight'].map('{:.2%}'.format)
            st.dataframe(significant_holdings, use_container_width=True)
    
    # Tab 2: Backtesting
    with tab2:
        st.header("📊 Portfolio Backtesting")
        
        if 'optimization_results' not in st.session_state:
            st.warning("⚠️ Please run portfolio optimization first in the 'Portfolio Optimization' tab.")
        else:
            # Backtesting configuration
            col1, col2 = st.columns(2)
            
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
                        strategy_name = "Optimized (Walk-Forward)"
                    
                    progress_placeholder.empty()
                    
                    # Calculate benchmarks
                    benchmark_performance = calculate_benchmark_performance(time_period)
                    equal_weight_performance = calculate_equal_weight_performance(data, tickers)
                    
                    # Store results
                    st.session_state.backtest_results = {
                        'portfolio': portfolio_performance,
                        'benchmark': benchmark_performance,
                        'equal_weight': equal_weight_performance,
                        'strategy_name': strategy_name
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
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Performance metrics comparison
                st.subheader("📊 Performance Metrics Comparison")
                
                metrics_data = {
                    'Strategy': [results['strategy_name'], 'S&P 500 (SPY)', 'Equal Weight'],
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
                    'Sharpe Ratio': [
                        results['portfolio']['sharpe_ratio'],
                        results['benchmark']['sharpe_ratio'],
                        results['equal_weight']['sharpe_ratio']
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df['Annual Return'] = metrics_df['Annual Return'].map('{:.2%}'.format)
                metrics_df['Volatility'] = metrics_df['Volatility'].map('{:.2%}'.format)
                metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].map('{:.2f}'.format)
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Risk-return scatter plot
                st.subheader("🎯 Risk-Return Analysis")
                results_df = pd.DataFrame({
                    'strategy': metrics_data['Strategy'],
                    'return': [float(x.strip('%'))/100 for x in metrics_df['Annual Return']],
                    'volatility': [float(x.strip('%'))/100 for x in metrics_df['Volatility']],
                    'sharpe_ratio': [float(x) for x in metrics_df['Sharpe Ratio']]
                })
                
                fig_scatter = create_risk_return_scatter(results_df)
                st.plotly_chart(fig_scatter, use_container_width=True)
    
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
            
            if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
                try:
                    with st.spinner("Running Monte Carlo simulation..."):
                        # Initialize Monte Carlo simulator
                        returns = optimizer.calculate_returns(data)
                        result = st.session_state.optimization_results
                        
                        mc_simulator = MonteCarloSimulator(
                            returns=returns,
                            weights=result['weights'],
                            initial_value=10000
                        )
                        
                        # Run simulation
                        simulation_results = mc_simulator.simulate_portfolio(
                            n_simulations=n_simulations,
                            time_horizon=time_horizon
                        )
                        
                        # Calculate risk metrics
                        final_values = simulation_results[:, -1]
                        
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
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Percentile analysis
                        st.subheader("📊 Percentile Analysis")
                        
                        percentiles = [5, 10, 25, 50, 75, 90, 95]
                        percentile_values = [np.percentile(final_values, p) for p in percentiles]
                        
                        percentile_df = pd.DataFrame({
                            'Percentile': [f"{p}th" for p in percentiles],
                            'Portfolio Value': [f"${v:,.0f}" for v in percentile_values],
                            'Return': [f"{(v/10000 - 1)*100:.1f}%" for v in percentile_values]
                        })
                        
                        st.dataframe(percentile_df, use_container_width=True)
                        
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
                from utils import RiskMetrics
                risk_calc = RiskMetrics()
                
                # Calculate additional metrics
                max_drawdown = risk_calc.calculate_max_drawdown(results['portfolio']['cumulative_returns'])
                sortino_ratio = risk_calc.calculate_sortino_ratio(portfolio_returns)
                calmar_ratio = results['portfolio']['annual_return'] / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # Display advanced metrics
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
                
                with risk_col2:
                    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                
                with risk_col3:
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
                
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
                
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Monthly returns heatmap
                st.subheader("🗓️ Monthly Returns Heatmap")
                
                monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_pivot = monthly_returns.groupby([
                    monthly_returns.index.year,
                    monthly_returns.index.month
                ]).first().unstack()
                
                if len(monthly_returns_pivot) > 0:
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=monthly_returns_pivot.values * 100,  # Convert to percentage
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        y=monthly_returns_pivot.index,
                        colorscale='RdYlGn',
                        text=monthly_returns_pivot.values * 100,
                        texttemplate="%{text:.1f}%",
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Monthly Returns (%)",
                        xaxis_title="Month",
                        yaxis_title="Year",
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
            except ImportError:
                st.info("Advanced risk metrics require the utils module with RiskMetrics class.")
            except Exception as e:
                st.error(f"Error calculating advanced metrics: {str(e)}")

if __name__ == "__main__":
    main()
