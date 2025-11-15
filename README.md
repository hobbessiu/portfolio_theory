# Modern Portfolio Theory Dashboard

A comprehensive Python application implementing Modern Portfolio Theory (MPT) for S&P 500 stock portfolio optimization with interactive visualizations, backtesting, and Monte Carlo simulation.

## 🚀 Features

### Core Functionality
- **Portfolio Optimization**: Implements Modern Portfolio Theory using mean-variance optimization
- **S&P 500 Integration**: Fetches top N stocks from S&P 500 by market cap
- **Interactive Dashboard**: Beautiful Streamlit web interface with real-time updates
- **Backtesting Engine**: Compare portfolio performance against benchmarks
- **Monte Carlo Simulation**: Risk analysis and future performance projections

### Advanced Analytics
- **Efficient Frontier**: Visualize risk-return trade-offs
- **Performance Metrics**: Comprehensive risk and return analysis
- **Benchmark Comparison**: Compare against S&P 500 ETF (SPY)
- **Risk Analysis**: VaR, Sharpe ratio, Sortino ratio, and more
- **Transaction Costs**: Configurable trading costs in backtesting

### Visualizations
- Interactive portfolio allocation pie charts
- Efficient frontier plots
- Performance comparison charts
- Monte Carlo simulation paths
- Risk-return scatter plots

## 🛠️ Technology Stack

- **Python 3.8+**
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive charting
- **pandas & numpy**: Data manipulation and analysis
- **scipy & cvxpy**: Optimization algorithms
- **yfinance**: Financial data API
- **scikit-learn**: Additional analytics

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd portfolio_theory
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🎯 Usage Guide

### Getting Started

1. **Configure Parameters**: Use the sidebar to set:
   - Number of top S&P 500 stocks (10-100)
   - Historical data period (1y-10y)
   - Risk-free rate
   - Transaction costs

2. **Run Optimization**: Click "Run Portfolio Optimization" to:
   - Fetch latest S&P 500 data
   - Calculate optimal portfolio weights
   - Generate efficient frontier
   - Display allocation and metrics

### Dashboard Tabs

#### 📈 Portfolio Optimization
- View optimal portfolio allocation
- Interactive pie chart of holdings
- Efficient frontier visualization
- Key performance metrics

#### 📊 Backtesting
- Compare portfolio vs S&P 500 benchmark
- Configurable rebalancing frequency
- Performance attribution analysis
- Risk-adjusted return metrics

#### 🎲 Monte Carlo Simulation
- Future portfolio path projections
- Configurable simulation parameters
- Risk analysis and VaR calculations
- Probability distributions

#### 📋 Performance Metrics
- Comprehensive performance dashboard
- Detailed risk analytics
- Portfolio composition analysis
- Performance attribution

## 🔧 Configuration Options

### Portfolio Settings
- **Top N Stocks**: Select 10-100 top S&P 500 stocks by market cap
- **Time Period**: Choose historical data period (1y, 2y, 3y, 5y, 10y)
- **Risk-Free Rate**: Set benchmark risk-free rate (default: 2%)

### Trading Settings
- **Transaction Costs**: Configure trading costs for backtesting
- **Rebalancing**: Choose frequency (Monthly, Quarterly, Semi-Annual, Annual)

### Simulation Settings
- **Monte Carlo Runs**: 500-5000 simulations
- **Time Horizon**: 30-1000 days projection period

## 📊 Key Metrics Explained

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside deviation adjusted return
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Maximum Drawdown**: Largest peak-to-trough decline

### Portfolio Metrics
- **Expected Return**: Annualized expected portfolio return
- **Volatility**: Annualized standard deviation of returns
- **Diversification Ratio**: Effective number of positions
- **Beta**: Portfolio sensitivity to market movements

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time data integration
- [ ] Additional optimization algorithms (Black-Litterman, Risk Parity)
- [ ] Sector and geographic constraints
- [ ] ESG factor integration
- [ ] Options and derivatives support
- [ ] Multi-asset class optimization

### Technical Improvements
- [ ] Database integration for historical storage
- [ ] API rate limiting and caching
- [ ] Advanced charting features
- [ ] Export functionality (PDF reports)
- [ ] User authentication and portfolios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

## 🙏 Acknowledgments

- Modern Portfolio Theory by Harry Markowitz
- S&P 500 data provided by Yahoo Finance
- Streamlit community for excellent documentation
- Open source Python financial libraries

---

**Built with ❤️ using Python and Modern Portfolio Theory**
