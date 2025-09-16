"""
🔄 BACKTESTING EXAMPLE - Ví dụ về backtesting chiến lược
=====================================================

Demo backtesting các chiến lược giao dịch với dữ liệu thực
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.backtesting.backtesting_engine import BacktestingEngine, OrderType
from src.backtesting.strategies import (
    MovingAverageCrossoverStrategy,
    RSIStrategy, 
    MACDStrategy,
    BollingerBandsStrategy,
    MultiSignalStrategy,
    StrategyOptimizer
)
from src.data.data_provider import DataProvider
from src.utils.config import ConfigManager
from src.analysis.indicators import TechnicalIndicators

def setup_logging():
    """Cấu hình logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_sample_data(symbol: str = "VCB", days: int = 365) -> pd.DataFrame:
    """
    Lấy dữ liệu mẫu để backtesting
    
    Args:
        symbol: Mã cổ phiếu
        days: Số ngày dữ liệu
        
    Returns:
        DataFrame chứa dữ liệu OHLCV
    """
    try:
        config = ConfigManager()
        data_provider = DataProvider(config)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = data_provider.get_historical_data(
            symbol, 
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df is None or len(df) == 0:
            print(f"❌ Không có dữ liệu cho {symbol}")
            return None
        
        # Thêm indicators
        indicators = TechnicalIndicators()
        df = indicators.calculate_all(df)
        df['symbol'] = symbol
        
        print(f"✅ Lấy được {len(df)} ngày dữ liệu cho {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ Lỗi lấy dữ liệu: {e}")
        return None

def demo_single_strategy():
    """Demo backtesting một chiến lược đơn"""
    print("\n" + "="*60)
    print("📊 DEMO: BACKTESTING CHIẾN LƯỢC ĐƠN")
    print("="*60)
    
    # Lấy dữ liệu
    symbol = "VCB"
    data = get_sample_data(symbol, days=365)
    
    if data is None:
        print("❌ Không thể lấy dữ liệu")
        return
    
    # Tạo backtesting engine
    engine = BacktestingEngine(
        initial_capital=100_000_000,  # 100M VND
        commission_rate=0.0015        # 0.15%
    )
    
    # Tạo chiến lược MA Crossover
    strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=20)
    
    print(f"\n🎯 Chiến lược: {strategy.name}")
    print(f"📈 Cổ phiếu: {symbol}")
    print(f"📅 Thời gian: {data.index[0].date()} đến {data.index[-1].date()}")
    
    # Tạo tín hiệu
    signals = strategy.generate_signals(data.copy())
    print(f"📊 Tổng số tín hiệu: {len(signals)}")
    
    # Thực hiện backtesting
    position_size = 1000  # 1000 cổ phiếu mỗi lệnh
    
    for signal in signals:
        # Cập nhật giá hiện tại
        current_row = data.loc[signal.timestamp]
        engine.update_positions({symbol: current_row})
        
        if signal.signal_type == 'BUY':
            order = engine.place_order(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.BUY,
                quantity=position_size,
                price=signal.price
            )
            engine.execute_pending_orders(current_row)
            
        elif signal.signal_type == 'SELL':
            # Chỉ bán nếu có position
            if symbol in engine.current_positions:
                current_quantity = engine.current_positions[symbol].quantity
                sell_quantity = min(position_size, current_quantity)
                
                if sell_quantity > 0:
                    order = engine.place_order(
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        order_type=OrderType.SELL,
                        quantity=sell_quantity,
                        price=signal.price
                    )
                    engine.execute_pending_orders(current_row)
        
        # Ghi equity
        engine.record_equity(signal.timestamp)
    
    # Tính performance
    metrics = engine.calculate_performance_metrics()
    
    print(f"\n📈 KẾT QUỢ BACKTESTING:")
    print(f"💰 Vốn ban đầu: {engine.initial_capital:,.0f} VND")
    print(f"💰 Vốn cuối: {engine.get_portfolio_value():,.0f} VND")
    print(f"📊 Tổng return: {metrics.total_return:.2f}%")
    print(f"📊 Return hàng năm: {metrics.annual_return:.2f}%")
    print(f"📊 Volatility: {metrics.volatility:.2f}%")
    print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"📊 Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"🎯 Tổng số trades: {metrics.total_trades}")
    print(f"🎯 Win Rate: {metrics.win_rate:.2f}%")
    print(f"🎯 Profit Factor: {metrics.profit_factor:.2f}")
    print(f"💸 Tổng phí: {metrics.total_commission:,.0f} VND")
    
    # Xuất báo cáo
    filename = engine.export_results(f"backtest_{symbol}_{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    if filename:
        print(f"📄 Đã xuất báo cáo: {filename}")

def demo_multi_strategy():
    """Demo backtesting chiến lược đa tín hiệu"""
    print("\n" + "="*60)
    print("📊 DEMO: BACKTESTING CHIẾN LƯỢC ĐA TÍN HIỆU")
    print("="*60)
    
    # Lấy dữ liệu
    symbol = "HPG"
    data = get_sample_data(symbol, days=200)
    
    if data is None:
        print("❌ Không thể lấy dữ liệu")
        return
    
    # Tạo backtesting engine
    engine = BacktestingEngine(
        initial_capital=100_000_000,
        commission_rate=0.0015
    )
    
    # Tạo các chiến lược con
    ma_strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=20)
    rsi_strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    macd_strategy = MACDStrategy(fast=12, slow=26, signal=9)
    
    # Tạo chiến lược đa tín hiệu
    multi_strategy = MultiSignalStrategy(
        strategies=[ma_strategy, rsi_strategy, macd_strategy],
        min_signals=2  # Cần ít nhất 2 tín hiệu trùng khớp
    )
    
    print(f"\n🎯 Chiến lược: {multi_strategy.name}")
    print(f"📈 Cổ phiếu: {symbol}")
    print(f"📅 Thời gian: {data.index[0].date()} đến {data.index[-1].date()}")
    print(f"🔧 Yêu cầu tối thiểu: {multi_strategy.min_signals} tín hiệu")
    
    # Tạo tín hiệu
    signals = multi_strategy.generate_signals(data.copy())
    print(f"📊 Tổng số tín hiệu multi: {len(signals)}")
    
    # Thực hiện backtesting
    position_size = 2000
    
    for signal in signals:
        current_row = data.loc[signal.timestamp]
        engine.update_positions({symbol: current_row})
        
        if signal.signal_type == 'BUY':
            engine.place_order(
                timestamp=signal.timestamp,
                symbol=symbol,
                order_type=OrderType.BUY,
                quantity=position_size,
                price=signal.price
            )
            engine.execute_pending_orders(current_row)
            
        elif signal.signal_type == 'SELL':
            if symbol in engine.current_positions:
                current_quantity = engine.current_positions[symbol].quantity
                sell_quantity = min(position_size, current_quantity)
                
                if sell_quantity > 0:
                    engine.place_order(
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        order_type=OrderType.SELL,
                        quantity=sell_quantity,
                        price=signal.price
                    )
                    engine.execute_pending_orders(current_row)
        
        engine.record_equity(signal.timestamp)
    
    # Kết quả
    metrics = engine.calculate_performance_metrics()
    
    print(f"\n📈 KẾT QUỢ MULTI-STRATEGY:")
    print(f"💰 Tổng return: {metrics.total_return:.2f}%")
    print(f"📊 Return hàng năm: {metrics.annual_return:.2f}%")
    print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"📊 Max Drawdown: {metrics.max_drawdown:.2f}%")
    print(f"🎯 Win Rate: {metrics.win_rate:.2f}%")
    print(f"🎯 Tổng trades: {metrics.total_trades}")
    
    # So sánh với Buy & Hold
    buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
    print(f"\n📊 SO SÁNH:")
    print(f"🎯 Multi-Strategy: {metrics.total_return:.2f}%")
    print(f"🏪 Buy & Hold: {buy_hold_return:.2f}%")
    print(f"🚀 Alpha: {metrics.total_return - buy_hold_return:.2f}%")

def demo_strategy_comparison():
    """Demo so sánh nhiều chiến lược"""
    print("\n" + "="*60)
    print("📊 DEMO: SO SÁNH CÁC CHIẾN LƯỢC")
    print("="*60)
    
    # Lấy dữ liệu
    symbol = "VNM"
    data = get_sample_data(symbol, days=300)
    
    if data is None:
        print("❌ Không thể lấy dữ liệu")
        return
    
    # Danh sách chiến lược để test
    strategies = [
        MovingAverageCrossoverStrategy(fast_period=5, slow_period=15),
        MovingAverageCrossoverStrategy(fast_period=10, slow_period=20),
        RSIStrategy(period=14),
        MACDStrategy(),
        BollingerBandsStrategy(period=20)
    ]
    
    results = []
    
    print(f"\n🎯 Cổ phiếu: {symbol}")
    print(f"📅 Thời gian: {data.index[0].date()} đến {data.index[-1].date()}")
    print(f"🔧 Testing {len(strategies)} chiến lược...")
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n📊 [{i}/{len(strategies)}] Testing {strategy.name}...")
        
        # Tạo engine mới
        engine = BacktestingEngine(
            initial_capital=100_000_000,
            commission_rate=0.0015
        )
        
        # Tạo tín hiệu
        signals = strategy.generate_signals(data.copy())
        
        # Thực hiện backtesting
        position_size = 1500
        
        for signal in signals:
            current_row = data.loc[signal.timestamp]
            engine.update_positions({symbol: current_row})
            
            if signal.signal_type == 'BUY':
                engine.place_order(
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.BUY,
                    quantity=position_size,
                    price=signal.price
                )
                engine.execute_pending_orders(current_row)
                
            elif signal.signal_type == 'SELL':
                if symbol in engine.current_positions:
                    current_quantity = engine.current_positions[symbol].quantity
                    sell_quantity = min(position_size, current_quantity)
                    
                    if sell_quantity > 0:
                        engine.place_order(
                            timestamp=signal.timestamp,
                            symbol=symbol,
                            order_type=OrderType.SELL,
                            quantity=sell_quantity,
                            price=signal.price
                        )
                        engine.execute_pending_orders(current_row)
            
            engine.record_equity(signal.timestamp)
        
        # Tính metrics
        metrics = engine.calculate_performance_metrics()
        
        results.append({
            'Strategy': strategy.name,
            'Total Return (%)': f"{metrics.total_return:.2f}",
            'Annual Return (%)': f"{metrics.annual_return:.2f}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
            'Max Drawdown (%)': f"{metrics.max_drawdown:.2f}",
            'Win Rate (%)': f"{metrics.win_rate:.2f}",
            'Total Trades': metrics.total_trades,
            'Profit Factor': f"{metrics.profit_factor:.2f}"
        })
    
    # Hiển thị bảng so sánh
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 BẢNG SO SÁNH CHIẾN LƯỢC:")
    print("="*100)
    print(results_df.to_string(index=False))
    
    # Tìm chiến lược tốt nhất
    best_return_idx = results_df['Total Return (%)'].astype(float).idxmax()
    best_sharpe_idx = results_df['Sharpe Ratio'].astype(float).idxmax()
    
    print(f"\n🏆 CHIẾN LƯỢC TỐT NHẤT:")
    print(f"📈 Return cao nhất: {results_df.loc[best_return_idx, 'Strategy']} ({results_df.loc[best_return_idx, 'Total Return (%)']}%)")
    print(f"⚖️ Sharpe tốt nhất: {results_df.loc[best_sharpe_idx, 'Strategy']} ({results_df.loc[best_sharpe_idx, 'Sharpe Ratio']})")

def demo_parameter_optimization():
    """Demo tối ưu hóa tham số"""
    print("\n" + "="*60)
    print("🔧 DEMO: TỐI ƯU HÓA THAM SỐ")
    print("="*60)
    
    # Lấy dữ liệu
    symbol = "FPT"
    data = get_sample_data(symbol, days=180)
    
    if data is None:
        print("❌ Không thể lấy dữ liệu")
        return
    
    print(f"\n🎯 Tối ưu MA Crossover cho {symbol}")
    print(f"📅 Dữ liệu: {len(data)} ngày")
    
    # Tạo optimizer
    engine = BacktestingEngine(initial_capital=100_000_000)
    optimizer = StrategyOptimizer(engine)
    
    # Chạy optimization (simplified version)
    print("🔍 Đang tìm tham số tốt nhất...")
    
    best_return = -np.inf
    best_params = None
    test_results = []
    
    # Test các tham số MA
    for fast in range(5, 16, 2):  # 5, 7, 9, 11, 13, 15
        for slow in range(20, 31, 5):  # 20, 25, 30
            if fast >= slow:
                continue
            
            # Reset engine
            engine.reset()
            
            # Test strategy
            strategy = MovingAverageCrossoverStrategy(fast, slow)
            signals = strategy.generate_signals(data.copy())
            
            # Execute trades
            for signal in signals:
                current_row = data.loc[signal.timestamp]
                engine.update_positions({symbol: current_row})
                
                if signal.signal_type == 'BUY':
                    engine.place_order(signal.timestamp, symbol, OrderType.BUY, 1000, signal.price)
                    engine.execute_pending_orders(current_row)
                elif signal.signal_type == 'SELL':
                    if symbol in engine.current_positions:
                        qty = min(1000, engine.current_positions[symbol].quantity)
                        if qty > 0:
                            engine.place_order(signal.timestamp, symbol, OrderType.SELL, qty, signal.price)
                            engine.execute_pending_orders(current_row)
                
                engine.record_equity(signal.timestamp)
            
            metrics = engine.calculate_performance_metrics()
            
            test_results.append({
                'Fast MA': fast,
                'Slow MA': slow,
                'Return (%)': round(metrics.total_return, 2),
                'Sharpe': round(metrics.sharpe_ratio, 2),
                'Max DD (%)': round(metrics.max_drawdown, 2),
                'Trades': metrics.total_trades
            })
            
            if metrics.total_return > best_return:
                best_return = metrics.total_return
                best_params = (fast, slow)
    
    # Hiển thị kết quả
    results_df = pd.DataFrame(test_results)
    
    print(f"\n📊 KẾT QUẢ OPTIMIZATION:")
    print("="*80)
    print(results_df.sort_values('Return (%)', ascending=False).head(10).to_string(index=False))
    
    print(f"\n🏆 THAM SỐ TỐI NHẤT:")
    print(f"📈 Fast MA: {best_params[0]}")
    print(f"📈 Slow MA: {best_params[1]}")
    print(f"📊 Return: {best_return:.2f}%")

def main():
    """Hàm chính"""
    setup_logging()
    
    print("🔄 BACKTESTING SYSTEM DEMO")
    print("=========================")
    
    while True:
        print("\n📋 MENU:")
        print("1. Demo chiến lược đơn")
        print("2. Demo chiến lược đa tín hiệu")
        print("3. So sánh các chiến lược")
        print("4. Tối ưu hóa tham số")
        print("5. Thoát")
        
        choice = input("\nNhập lựa chọn (1-5): ").strip()
        
        if choice == '1':
            demo_single_strategy()
        elif choice == '2':
            demo_multi_strategy()
        elif choice == '3':
            demo_strategy_comparison()
        elif choice == '4':
            demo_parameter_optimization()
        elif choice == '5':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        input("\n📝 Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()
