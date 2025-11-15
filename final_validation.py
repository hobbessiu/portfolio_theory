#!/usr/bin/env python3
"""
Final Validation: Both Backtesting Methods
==========================================

This script validates that both Fixed Weights and Walk-Forward Analysis 
methods are working correctly in the portfolio optimization system.
"""

import numpy as np
import pandas as pd
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine
from datetime import datetime

def comprehensive_validation():
    """Validate both backtesting methodologies work correctly."""
    
    print("🎯 COMPREHENSIVE BACKTESTING VALIDATION")
    print("=" * 55)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    engine = BacktestEngine(transaction_cost=0.001)
    
    # Test with a representative set of stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    print(f"📊 Testing with: {', '.join(tickers)}")
    
    try:
        # Load data
        data = optimizer.fetch_historical_data(tickers, "1y")
        print(f"📈 Data loaded: {data.shape[0]} days, {data.shape[1]} stocks")
        
        if data.empty:
            print("❌ No data available - validation failed")
            return False
        
        # Test 1: Fixed Weights Method
        print(f"\n1️⃣ TESTING FIXED WEIGHTS METHOD")
        print("-" * 35)
        
        # Initial optimization
        returns = data.pct_change().dropna()
        opt_result = optimizer.optimize_portfolio(returns)
        
        if opt_result['status'] != 'optimal':
            print("❌ Initial optimization failed")
            return False
            
        print(f"✅ Optimization successful (Sharpe: {opt_result['stats']['sharpe_ratio']:.3f})")
        
        # Fixed weights backtest
        fixed_result = engine.backtest_portfolio(
            opt_result['weights'], data, 'M'  # Monthly rebalancing
        )
        
        fixed_return = (fixed_result['cumulative_returns'].iloc[-1] - 1) * 100
        fixed_costs = fixed_result['rebalancing_costs'].sum() * 100
        
        print(f"📊 Fixed Weights Results:")
        print(f"   • Total Return: {fixed_return:.2f}%")
        print(f"   • Transaction Costs: {fixed_costs:.3f}%")
        print(f"   • Final Value: ${fixed_result['portfolio_value'].iloc[-1]:,.0f}")
        
        # Test 2: Walk-Forward Analysis Method
        print(f"\n2️⃣ TESTING WALK-FORWARD ANALYSIS")
        print("-" * 35)
        
        # Walk-forward backtest
        wf_result = engine.backtest_portfolio_walk_forward(
            data, 'M', min_history_days=60  # Monthly with 60-day minimum history
        )
        
        wf_return = (wf_result['cumulative_returns'].iloc[-1] - 1) * 100
        wf_costs = wf_result['rebalancing_costs'].sum() * 100
        wf_optimizations = wf_result.attrs.get('optimization_count', 0)
        
        print(f"📊 Walk-Forward Results:")
        print(f"   • Total Return: {wf_return:.2f}%")
        print(f"   • Transaction Costs: {wf_costs:.3f}%")
        print(f"   • Final Value: ${wf_result['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   • Re-optimizations: {wf_optimizations}")
        
        # Test 3: Results Validation
        print(f"\n3️⃣ VALIDATION CHECKS")
        print("-" * 25)
        
        # Check data integrity
        data_integrity = (
            len(fixed_result) == len(wf_result) and
            fixed_result.index[0] == wf_result.index[0] and
            fixed_result.index[-1] == wf_result.index[-1]
        )
        
        print(f"✅ Data integrity: {'PASS' if data_integrity else 'FAIL'}")
        
        # Check reasonable performance bounds
        reasonable_returns = (
            -50 < fixed_return < 200 and
            -50 < wf_return < 200
        )
        
        print(f"✅ Reasonable returns: {'PASS' if reasonable_returns else 'FAIL'}")
        
        # Check transaction costs are positive
        positive_costs = fixed_costs >= 0 and wf_costs >= 0
        
        print(f"✅ Positive costs: {'PASS' if positive_costs else 'FAIL'}")
        
        # Check walk-forward had optimizations
        had_optimizations = wf_optimizations > 0
        
        print(f"✅ Walk-forward optimized: {'PASS' if had_optimizations else 'FAIL'}")
        
        # Test 4: Comparative Analysis
        print(f"\n4️⃣ COMPARATIVE ANALYSIS")
        print("-" * 30)
        
        return_diff = wf_return - fixed_return
        cost_diff = wf_costs - fixed_costs
        
        print(f"📈 Performance Comparison:")
        print(f"   • Return difference: {return_diff:+.2f}% (WF vs Fixed)")
        print(f"   • Cost difference: {cost_diff:+.3f}% (WF vs Fixed)")
        
        if abs(return_diff) < 0.1:  # Very similar returns
            print(f"   💡 Returns are very similar between methods")
        elif return_diff > 0:
            print(f"   🚀 Walk-forward outperformed by {return_diff:.2f}%")
        else:
            print(f"   🎯 Fixed weights outperformed by {abs(return_diff):.2f}%")
            
        if cost_diff > 0.1:  # Significant cost difference
            print(f"   ⚠️ Walk-forward has {cost_diff:.3f}% higher costs")
        else:
            print(f"   ✅ Transaction cost difference is minimal")
        
        # Final validation
        all_checks_pass = (
            data_integrity and reasonable_returns and 
            positive_costs and had_optimizations
        )
        
        if all_checks_pass:
            print(f"\n✅ VALIDATION SUCCESSFUL!")
            print(f"🎉 Both backtesting methods are working correctly")
            print(f"\n📋 Summary:")
            print(f"   • Fixed Weights: {fixed_return:+.2f}% return, {fixed_costs:.3f}% costs")
            print(f"   • Walk-Forward: {wf_return:+.2f}% return, {wf_costs:.3f}% costs, {wf_optimizations} optimizations")
            print(f"\n🚀 Ready for production use in Streamlit app!")
            return True
        else:
            print(f"\n❌ VALIDATION FAILED - Some checks did not pass")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = comprehensive_validation()
    
    if success:
        print(f"\n" + "=" * 55)
        print(f"🎯 WALK-FORWARD IMPLEMENTATION: COMPLETE & VALIDATED ✅")
        print(f"=" * 55)
        print(f"\nNext steps:")
        print(f"1. Launch Streamlit app: streamlit run app.py")
        print(f"2. Navigate to 'Backtesting' tab")
        print(f"3. Select between 'Fixed Weights' and 'Walk-Forward Analysis'")
        print(f"4. Compare the methodologies with your own portfolio!")
    else:
        print(f"\n❌ Implementation needs review")
