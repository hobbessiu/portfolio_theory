<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Modern Portfolio Theory Project - Copilot Instructions

## Project Overview
This is a Modern Portfolio Theory (MPT) implementation project using Python and Streamlit for portfolio optimization of S&P 500 stocks. The project includes interactive visualizations, backtesting, and Monte Carlo simulation capabilities.

## Code Style Guidelines
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Write comprehensive docstrings for classes and methods
- Use descriptive variable and function names
- Maintain modular code structure with separation of concerns

## Architecture Principles
- **portfolio_optimizer.py**: Core MPT algorithms and data fetching
- **app.py**: Streamlit dashboard and user interface
- Separate concerns: optimization logic, visualization, and data handling
- Use caching for expensive operations (data fetching, calculations)
- Implement error handling for external API calls

## Financial Domain Knowledge
- Focus on Modern Portfolio Theory implementation
- Use appropriate financial metrics (Sharpe ratio, Sortino ratio, VaR, etc.)
- Implement proper risk-return optimization
- Consider transaction costs in backtesting
- Use annualized returns and volatility for consistency

## Data Handling
- Cache expensive data operations using Streamlit caching
- Handle missing data and API failures gracefully
- Use pandas for data manipulation and numpy for numerical calculations
- Ensure data consistency across different time periods

## Visualization Standards
- Use Plotly for interactive charts
- Maintain consistent color schemes across visualizations
- Include proper titles, axis labels, and legends
- Make charts responsive and user-friendly
- Add hover information and tooltips where appropriate

## Performance Considerations
- Cache data fetching operations
- Optimize portfolio optimization algorithms
- Use vectorized operations with numpy/pandas
- Consider computational complexity for Monte Carlo simulations
- Implement progress indicators for long-running operations

## Error Handling
- Gracefully handle API failures and network issues
- Validate user inputs and configuration parameters
- Provide helpful error messages to users
- Implement fallback options when possible

## Testing Considerations
- Focus on financial calculation accuracy
- Test optimization algorithms with known datasets
- Validate risk metric calculations
- Test edge cases (empty data, invalid parameters)

## Security Notes
- Be cautious with external API calls and rate limiting
- Validate all user inputs
- Don't expose sensitive configuration in the code
- Use environment variables for API keys if needed
