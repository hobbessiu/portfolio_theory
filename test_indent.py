# Test file to verify correct indentation structure
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
