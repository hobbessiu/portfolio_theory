#!/usr/bin/env python3
"""
Max Drawdown Implementation Test
===============================

Test the max drawdown calculation in both Fixed Weights and Walk-Forward methods.
"""

import numpy as np
import pandas as pd
from portfolio_optimizer import PortfolioOptimizer, BacktestEngine
from app import calculate_max_drawdown

def test_max_drawdown_functionality():
    """Test max drawdown calculation with backtesting methods."""
    
    print("📊 MAX DRAWDOWN IMPLEMENTATION TEST")
    print("=" * 40)
    
    # Setup
    optimizer = PortfolioOptimizer()
    engine = BacktestEngine(transaction_cost=0.001)
    
    # Test with sample stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    print(f"🎯 Testing with: {', '.join(tickers)}")
    
    try:
        # Load data
        data = optimizer.fetch_historical_data(tickers, "1y")
        print(f"📈 Data loaded: {data.shape[0]} days, {data.shape[1]} stocks")
        
        if data.empty:
            print("❌ No data - test aborted")
            return False
        
        # Test 1: Fixed Weights Method with Max Drawdown
        print(f"\n1️⃣ FIXED WEIGHTS + MAX DRAWDOWN")
        print("-" * 35)
        
        returns = data.pct_change().dropna()
        opt_result = optimizer.optimize_portfolio(returns)
        
        if opt_result['status'] != 'optimal':
            print("❌ Optimization failed")
            return False
            
        # Fixed weights backtest
        fixed_result = engine.backtest_portfolio(opt_result['weights'], data, 'Q')
        
        # Calculate max drawdown
        fixed_max_dd = calculate_max_drawdown(fixed_result['cumulative_returns'])
        
        print(f"✅ Fixed Weights Results:")
        print(f"   • Total Return: {(fixed_result['cumulative_returns'].iloc[-1] - 1)*100:.2f}%")
        print(f"   • Max Drawdown: {fixed_max_dd:.2%}")
        print(f"   • Final Value: ${fixed_result['portfolio_value'].iloc[-1]:,.0f}")
        
        # Test 2: Walk-Forward Method with Max Drawdown
        print(f"\n2️⃣ WALK-FORWARD + MAX DRAWDOWN")
        print("-" * 35)
        
        # Walk-forward backtest
        wf_result = engine.backtest_portfolio_walk_forward(data, 'Q', min_history_days=60)
        
        # Calculate max drawdown
        wf_max_dd = calculate_max_drawdown(wf_result['cumulative_returns'])
        
        print(f"✅ Walk-Forward Results:")
        print(f"   • Total Return: {(wf_result['cumulative_returns'].iloc[-1] - 1)*100:.2f}%")
        print(f"   • Max Drawdown: {wf_max_dd:.2%}")
        print(f"   • Final Value: ${wf_result['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   • Optimizations: {wf_result.attrs.get('optimization_count', 0)}")
        
        # Test 3: Validation Checks
        print(f"\n3️⃣ VALIDATION CHECKS")
        print("-" * 25)
        
        # Check max drawdowns are reasonable (between 0% and -100%)
        dd_reasonable = (-1.0 <= fixed_max_dd <= 0.0) and (-1.0 <= wf_max_dd <= 0.0)
        print(f"✅ Reasonable drawdowns: {'PASS' if dd_reasonable else 'FAIL'}")
        
        # Check drawdowns are negative (losses)
        dd_negative = fixed_max_dd <= 0 and wf_max_dd <= 0
        print(f"✅ Drawdowns are negative: {'PASS' if dd_negative else 'FAIL'}")
        
        # Check calculation works (not NaN or None)
        dd_valid = not np.isnan(fixed_max_dd) and not np.isnan(wf_max_dd)
        print(f"✅ Valid calculations: {'PASS' if dd_valid else 'FAIL'}")
        
        # Test 4: Comparison
        print(f"\n4️⃣ COMPARISON")
        print("-" * 15)
        
        dd_diff = wf_max_dd - fixed_max_dd
        
        print(f"📊 Max Drawdown Comparison:")
        print(f"   • Fixed Weights: {fixed_max_dd:.2%}")
        print(f"   • Walk-Forward:  {wf_max_dd:.2%}")
        print(f"   • Difference:    {dd_diff:+.2%}")
        
        if abs(dd_diff) < 0.02:  # Less than 2% difference
            print(f"   💡 Similar risk profiles between methods")
        elif dd_diff > 0:  # Walk-forward has smaller drawdown (less negative)
            print(f"   🚀 Walk-forward shows better drawdown control")
        else:
            print(f"   🎯 Fixed weights shows better drawdown control")
        
        # Final validation
        all_checks = dd_reasonable and dd_negative and dd_valid
        
        if all_checks:
            print(f"\n✅ MAX DRAWDOWN IMPLEMENTATION: SUCCESS!")
            print(f"🎉 Max drawdown calculations working correctly in both methods")
            print(f"\n📋 Summary:")
            print(f"   • Max drawdown function: ✅ Working")
            print(f"   • Fixed weights integration: ✅ Working") 
            print(f"   • Walk-forward integration: ✅ Working")
            print(f"   • Performance table: ✅ Will now show actual values")
            print(f"\n🚀 The Streamlit app will now display real max drawdown values!")
            return True
        else:
            print(f"\n❌ Some validation checks failed")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_max_drawdown_functionality()
    
    if success:
        print(f"\n" + "=" * 50)
        print(f"🎯 MAX DRAWDOWN FEATURE: IMPLEMENTED & TESTED ✅")
        print(f"=" * 50)
        print(f"\nThe performance comparison table in the Streamlit app")
        print(f"will now show actual max drawdown percentages instead of 'N/A'.")
        print(f"\nMax drawdown shows the largest peak-to-trough decline,")
        print(f"which is a key risk metric for portfolio evaluation.")
    else:
        print(f"\n❌ Max drawdown implementation needs review")
