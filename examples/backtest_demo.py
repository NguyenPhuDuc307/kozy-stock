#!/usr/bin/env python3
"""
📋 BACKTESTING EXAMPLE - Demo các chiến lược giao dịch
======================================================

Script demo backtesting với các chiến lược khác nhau
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

try:
    from src.backtesting.backtesting_engine import BacktestEngine, Portfolio
    from src.backtesting.strategies import (
        MovingAverageCrossoverStrategy, 
        MeanReversionStrategy, 
        MomentumStrategy, 
        BollingerBandsStrategy
    )
    from src.data.data_provider import DataProvider
    import pandas as pd
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)

def demo_moving_average_crossover():
    """
    Demo MA Crossover strategy trên VCB
    """
    print("🔄 Testing Moving Average Crossover Strategy on VCB...")
    
    try:
        # Get data
        data_provider = DataProvider()
        
        # Test với VCB - 1 năm gần đây
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = data_provider.get_historical_data(
            symbol='VCB',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            resolution='1D'
        )
        
        if data is None or data.empty:
            print("❌ Không thể lấy dữ liệu VCB")
            return None
        
        print(f"📊 Data: {len(data)} rows từ {data.index[0]} đến {data.index[-1]}")
        
        # Create strategy - MA nhanh 10 ngày, MA chậm 30 ngày
        strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=30)
        
        # Create portfolio - 100M VND
        portfolio = Portfolio(
            initial_capital=100_000_000,  # 100M VND
            position_size_method='fixed_percentage',
            position_size_value=0.2,  # 20% mỗi lệnh
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.1,  # 10% take profit
            max_drawdown_pct=0.2  # 20% max drawdown
        )
        
        # Run backtest
        engine = BacktestEngine(portfolio)
        results = engine.backtest(strategy, data)
        
        # Print results
        print("\n📊 MOVING AVERAGE CROSSOVER RESULTS")
        print("=" * 60)
        
        metrics = results['performance_metrics']
        print(f"💰 Total Return: {metrics['total_return']:.2%}")
        print(f"📈 Annual Return: {metrics['annual_return']:.2%}")
        print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"🔢 Total Trades: {metrics['total_trades']}")
        print(f"💹 Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"📊 Volatility: {metrics['volatility']:.2%}")
        
        # Show some trades
        if results['trades']:
            print(f"\n📋 Một số giao dịch đầu:")
            for i, trade in enumerate(results['trades'][:3], 1):
                action = trade['action'].upper()
                entry_date = trade['entry_date']
                entry_price = trade['entry_price']
                exit_price = trade.get('exit_price', 'Đang mở')
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                
                print(f"  {i}. {action} ngày {entry_date} @ {entry_price:.0f} → "
                      f"{exit_price if isinstance(exit_price, str) else f'{exit_price:.0f}'} = "
                      f"{pnl:,.0f} VND ({pnl_pct:.1%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None

def demo_bollinger_bands():
    """
    Demo Bollinger Bands strategy trên VNM
    """
    print("\n🔄 Testing Bollinger Bands Strategy on VNM...")
    
    try:
        # Get data
        data_provider = DataProvider()
        
        # Test với VNM - 6 tháng
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        data = data_provider.get_historical_data(
            symbol='VNM',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            resolution='1D'
        )
        
        if data is None or data.empty:
            print("❌ Không thể lấy dữ liệu VNM")
            return None
        
        print(f"📊 Data: {len(data)} rows từ {data.index[0]} đến {data.index[-1]}")
        
        # Create strategy
        strategy = BollingerBandsStrategy(period=20, std_multiplier=2.0)
        
        # Create portfolio
        portfolio = Portfolio(
            initial_capital=80_000_000,  # 80M VND
            position_size_method='fixed_percentage',
            position_size_value=0.25,  # 25% mỗi lệnh
            stop_loss_pct=0.07,  # 7% stop loss
            take_profit_pct=0.12,  # 12% take profit
            max_drawdown_pct=0.2
        )
        
        # Run backtest
        engine = BacktestEngine(portfolio)
        results = engine.backtest(strategy, data)
        
        # Print results
        print("\n📊 BOLLINGER BANDS RESULTS")
        print("=" * 60)
        
        metrics = results['performance_metrics']
        print(f"💰 Total Return: {metrics['total_return']:.2%}")
        print(f"📈 Annual Return: {metrics['annual_return']:.2%}")
        print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"🔢 Total Trades: {metrics['total_trades']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None

def demo_mean_reversion():
    """
    Demo Mean Reversion strategy trên HPG
    """
    print("\n🔄 Testing Mean Reversion Strategy on HPG...")
    
    try:
        # Get data
        data_provider = DataProvider()
        
        # Test với HPG - 9 tháng
        end_date = datetime.now()
        start_date = end_date - timedelta(days=270)
        
        data = data_provider.get_historical_data(
            symbol='HPG',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            resolution='1D'
        )
        
        if data is None or data.empty:
            print("❌ Không thể lấy dữ liệu HPG")
            return None
        
        print(f"📊 Data: {len(data)} rows từ {data.index[0]} đến {data.index[-1]}")
        
        # Create strategy
        strategy = MeanReversionStrategy(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=1.0
        )
        
        # Create portfolio
        portfolio = Portfolio(
            initial_capital=60_000_000,  # 60M VND
            position_size_method='risk_parity',
            position_size_value=0.15,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
            max_drawdown_pct=0.25
        )
        
        # Run backtest
        engine = BacktestEngine(portfolio)
        results = engine.backtest(strategy, data)
        
        # Print results
        print("\n📊 MEAN REVERSION RESULTS")
        print("=" * 60)
        
        metrics = results['performance_metrics']
        print(f"💰 Total Return: {metrics['total_return']:.2%}")
        print(f"📈 Annual Return: {metrics['annual_return']:.2%}")
        print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"🔢 Total Trades: {metrics['total_trades']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None

def compare_strategies():
    """
    So sánh các strategies
    """
    print("\n🏆 STRATEGY COMPARISON")
    print("=" * 80)
    
    strategies = [
        ("MA Crossover (VCB)", demo_moving_average_crossover),
        ("Bollinger Bands (VNM)", demo_bollinger_bands),
        ("Mean Reversion (HPG)", demo_mean_reversion)
    ]
    
    results_summary = []
    
    for name, demo_func in strategies:
        try:
            result = demo_func()
            if result and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                results_summary.append({
                    'Strategy': name,
                    'Total Return': metrics['total_return'],
                    'Annual Return': metrics['annual_return'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Win Rate': metrics['win_rate'],
                    'Total Trades': metrics['total_trades']
                })
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    
    # Print comparison table
    if results_summary:
        print("\n📊 SUMMARY COMPARISON")
        print("=" * 80)
        
        # Headers
        print(f"{'Strategy':<25} {'Total Ret':<10} {'Annual':<8} {'Sharpe':<8} {'Max DD':<8} {'Win Rate':<9} {'Trades':<6}")
        print("-" * 80)
        
        # Data rows
        for summary in results_summary:
            print(f"{summary['Strategy']:<25} "
                  f"{summary['Total Return']:<10.1%} "
                  f"{summary['Annual Return']:<8.1%} "
                  f"{summary['Sharpe Ratio']:<8.2f} "
                  f"{summary['Max Drawdown']:<8.1%} "
                  f"{summary['Win Rate']:<9.1%} "
                  f"{summary['Total Trades']:<6}")
        
        # Find best strategy
        if results_summary:
            best_by_return = max(results_summary, key=lambda x: x['Total Return'])
            best_by_sharpe = max(results_summary, key=lambda x: x['Sharpe Ratio'])
            
            print(f"\n🏆 Best by Total Return: {best_by_return['Strategy']} ({best_by_return['Total Return']:.1%})")
            print(f"🏆 Best by Sharpe Ratio: {best_by_sharpe['Strategy']} ({best_by_sharpe['Sharpe Ratio']:.2f})")

def main():
    """
    Main function
    """
    print("🚀 BACKTESTING DEMO")
    print("=" * 50)
    print("Testing different trading strategies on Vietnamese stocks...")
    print()
    
    try:
        # Run comparison
        compare_strategies()
        
        print("\n✅ Backtesting demo completed!")
        print("\n💡 Key Takeaways:")
        print("- Different strategies work better on different stocks")
        print("- Risk management is crucial (stop loss, position sizing)")
        print("- Past performance doesn't guarantee future results")
        print("- Always test on different time periods")
        print("- Consider transaction costs in real trading")
        
        print("\n🌐 Next Steps:")
        print("- Try the web interface: 📋 Backtest trong main app")
        print("- Test with different parameters")
        print("- Combine multiple strategies")
        print("- Add more sophisticated risk management")
        
    except Exception as e:
        print(f"❌ Error during backtesting demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
