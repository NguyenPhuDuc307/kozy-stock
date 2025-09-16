"""
🔄 BACKTEST EXAMPLE - Ví dụ backtest chiến lược
==============================================

Ví dụ về cách sử dụng hệ thống backtest với các chiến lược khác nhau
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

try:
    from src.main import StockAnalysisSystem
    from src.strategies.backtest import BacktestEngine, StrategyLibrary
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Không thể import hệ thống: {e}")
    SYSTEM_AVAILABLE = False

def setup_logging():
    """
    Thiết lập logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def backtest_golden_cross(symbol: str = "VCB", period: str = "2y"):
    """
    Backtest chiến lược Golden Cross
    
    Args:
        symbol: Mã cổ phiếu
        period: Khoảng thời gian
    """
    print(f"🔄 Backtest Golden Cross cho {symbol} ({period})")
    
    # Khởi tạo hệ thống
    system = StockAnalysisSystem()
    engine = BacktestEngine(initial_capital=100_000_000)  # 100M VND
    
    try:
        # Lấy dữ liệu và tính chỉ báo
        df = system.get_stock_data(symbol, period)
        if df is None or len(df) < 200:
            print(f"❌ Không đủ dữ liệu cho {symbol}")
            return
        
        df_with_indicators = system.calculate_indicators(df)
        
        # Chạy backtest
        result = engine.run_backtest(df_with_indicators, StrategyLibrary.golden_cross_strategy)
        
        if result:
            print(f"\n📊 KẾT QUẢ GOLDEN CROSS - {symbol}")
            print(f"=" * 50)
            engine.print_results(result)
            
            # So sánh với Buy & Hold
            buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            print(f"\n📈 So sánh với Buy & Hold:")
            print(f"   Golden Cross: {result.total_return:.2f}%")
            print(f"   Buy & Hold: {buy_hold_return:.2f}%")
            print(f"   Chênh lệch: {result.total_return - buy_hold_return:+.2f}%")
            
            return result
    
    except Exception as e:
        print(f"❌ Lỗi backtest {symbol}: {e}")
        return None

def backtest_rsi_strategy(symbol: str = "VPB", period: str = "1y"):
    """
    Backtest chiến lược RSI
    
    Args:
        symbol: Mã cổ phiếu
        period: Khoảng thời gian
    """
    print(f"\n🔄 Backtest RSI Strategy cho {symbol} ({period})")
    
    system = StockAnalysisSystem()
    engine = BacktestEngine(initial_capital=100_000_000)
    
    try:
        df = system.get_stock_data(symbol, period)
        if df is None or len(df) < 50:
            print(f"❌ Không đủ dữ liệu cho {symbol}")
            return
        
        df_with_indicators = system.calculate_indicators(df)
        
        # Chạy backtest RSI
        result = engine.run_backtest(df_with_indicators, StrategyLibrary.rsi_strategy)
        
        if result:
            print(f"\n📊 KẾT QUẢ RSI STRATEGY - {symbol}")
            print(f"=" * 50)
            engine.print_results(result)
            
            return result
    
    except Exception as e:
        print(f"❌ Lỗi backtest RSI {symbol}: {e}")
        return None

def backtest_macd_strategy(symbol: str = "CTG", period: str = "1y"):
    """
    Backtest chiến lược MACD
    
    Args:
        symbol: Mã cổ phiếu
        period: Khoảng thời gian
    """
    print(f"\n🔄 Backtest MACD Strategy cho {symbol} ({period})")
    
    system = StockAnalysisSystem()
    engine = BacktestEngine(initial_capital=100_000_000)
    
    try:
        df = system.get_stock_data(symbol, period)
        if df is None or len(df) < 50:
            print(f"❌ Không đủ dữ liệu cho {symbol}")
            return
        
        df_with_indicators = system.calculate_indicators(df)
        
        # Chạy backtest MACD
        result = engine.run_backtest(df_with_indicators, StrategyLibrary.macd_strategy)
        
        if result:
            print(f"\n📊 KẾT QUẢ MACD STRATEGY - {symbol}")
            print(f"=" * 50)
            engine.print_results(result)
            
            return result
    
    except Exception as e:
        print(f"❌ Lỗi backtest MACD {symbol}: {e}")
        return None

def compare_strategies(symbol: str = "VCB", period: str = "2y"):
    """
    So sánh các chiến lược khác nhau
    
    Args:
        symbol: Mã cổ phiếu
        period: Khoảng thời gian
    """
    print(f"\n🔄 So sánh chiến lược cho {symbol} ({period})")
    
    system = StockAnalysisSystem()
    
    try:
        df = system.get_stock_data(symbol, period)
        if df is None or len(df) < 200:
            print(f"❌ Không đủ dữ liệu cho {symbol}")
            return
        
        df_with_indicators = system.calculate_indicators(df)
        
        strategies = [
            ("Golden Cross", StrategyLibrary.golden_cross_strategy),
            ("RSI", StrategyLibrary.rsi_strategy),
            ("MACD", StrategyLibrary.macd_strategy),
            ("Bollinger Bands", StrategyLibrary.bollinger_bands_strategy)
        ]
        
        results = {}
        
        for strategy_name, strategy_func in strategies:
            try:
                engine = BacktestEngine(initial_capital=100_000_000)
                result = engine.run_backtest(df_with_indicators, strategy_func)
                
                if result:
                    results[strategy_name] = {
                        'total_return': result.total_return,
                        'max_drawdown': result.max_drawdown,
                        'sharpe_ratio': result.sharpe_ratio,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades
                    }
                    print(f"✅ {strategy_name}: {result.total_return:.2f}%")
            
            except Exception as e:
                print(f"❌ Lỗi với {strategy_name}: {e}")
        
        # So sánh với Buy & Hold
        buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        results['Buy & Hold'] = {
            'total_return': buy_hold_return,
            'max_drawdown': 0,  # Simplified
            'sharpe_ratio': 0,  # Would need calculation
            'win_rate': 0,
            'total_trades': 1
        }
        
        # Hiển thị bảng so sánh
        print(f"\n📊 BẢNG SO SÁNH CHIẾN LƯỢC - {symbol}")
        print(f"=" * 80)
        print(f"{'Chiến lược':<15} {'Return':<10} {'Max DD':<10} {'Sharpe':<8} {'Win%':<8} {'Trades':<8}")
        print(f"-" * 80)
        
        for strategy, metrics in results.items():
            print(f"{strategy:<15} "
                  f"{metrics['total_return']:>8.2f}% "
                  f"{metrics['max_drawdown']:>8.2f}% "
                  f"{metrics['sharpe_ratio']:>6.2f} "
                  f"{metrics['win_rate']:>6.1f}% "
                  f"{metrics['total_trades']:>6.0f}")
        
        # Tìm chiến lược tốt nhất
        best_return = max(results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        
        print(f"\n🏆 KẾT QUẢ:")
        print(f"   📈 Return cao nhất: {best_return[0]} ({best_return[1]['total_return']:.2f}%)")
        print(f"   📊 Sharpe cao nhất: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
        
        return results
    
    except Exception as e:
        print(f"❌ Lỗi so sánh chiến lược: {e}")
        return None

def portfolio_backtest(symbols: list = ["VCB", "BID", "CTG"], period: str = "1y"):
    """
    Backtest danh mục đầu tư
    
    Args:
        symbols: Danh sách mã cổ phiếu
        period: Khoảng thời gian
    """
    print(f"\n🔄 Backtest danh mục: {', '.join(symbols)} ({period})")
    
    system = StockAnalysisSystem()
    
    portfolio_results = {}
    total_return = 0
    
    for symbol in symbols:
        try:
            df = system.get_stock_data(symbol, period)
            if df is not None and len(df) > 0:
                first_price = df['close'].iloc[0]
                last_price = df['close'].iloc[-1]
                stock_return = ((last_price / first_price) - 1) * 100
                
                portfolio_results[symbol] = stock_return
                total_return += stock_return
                
                print(f"   {symbol}: {stock_return:+.2f}%")
        
        except Exception as e:
            print(f"   ❌ Lỗi với {symbol}: {e}")
    
    if portfolio_results:
        avg_return = total_return / len(portfolio_results)
        print(f"\n📊 Kết quả danh mục:")
        print(f"   📈 Return trung bình: {avg_return:.2f}%")
        print(f"   📊 Tổng return: {total_return:.2f}%")
        
        best_stock = max(portfolio_results.items(), key=lambda x: x[1])
        worst_stock = min(portfolio_results.items(), key=lambda x: x[1])
        
        print(f"   🏆 Tốt nhất: {best_stock[0]} ({best_stock[1]:+.2f}%)")
        print(f"   📉 Tệ nhất: {worst_stock[0]} ({worst_stock[1]:+.2f}%)")

def main():
    """
    Hàm chính
    """
    setup_logging()
    
    print("🔄 HỆ THỐNG BACKTEST CHIẾN LƯỢC")
    print("=" * 50)
    print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not SYSTEM_AVAILABLE:
        print("\n❌ Hệ thống không khả dụng!")
        return
    
    try:
        # 1. Backtest Golden Cross
        backtest_golden_cross("VCB", "2y")
        
        # 2. Backtest RSI
        backtest_rsi_strategy("BID", "1y")
        
        # 3. Backtest MACD
        backtest_macd_strategy("CTG", "1y")
        
        # 4. So sánh chiến lược
        compare_strategies("VCB", "2y")
        
        # 5. Backtest danh mục
        portfolio_backtest(["VCB", "BID", "CTG", "VPB", "TCB"], "1y")
        
        print(f"\n🎉 Hoàn thành backtest!")
        print(f"💡 Lưu ý: Kết quả mang tính chất tham khảo, không phải lời khuyên đầu tư")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        logging.error(f"Lỗi chính: {e}")

if __name__ == "__main__":
    main()
