"""
🧪 BASIC ANALYSIS EXAMPLE - Ví dụ phân tích cơ bản
===============================================

Ví dụ đơn giản về cách sử dụng hệ thống phân tích chứng khoán
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
    from src.utils.config import config
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Không thể import hệ thống: {e}")
    print("💡 Vui lòng chạy: pip install -r requirements.txt")
    SYSTEM_AVAILABLE = False

def setup_logging():
    """
    Thiết lập logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def analyze_stock(symbol: str = "VCB", period: str = "6mo"):
    """
    Phân tích một cổ phiếu
    
    Args:
        symbol: Mã cổ phiếu
        period: Khoảng thời gian
    """
    if not SYSTEM_AVAILABLE:
        print("❌ Hệ thống không khả dụng")
        return
    
    print(f"🔍 Đang phân tích cổ phiếu {symbol}...")
    
    # Khởi tạo hệ thống
    system = StockAnalysisSystem()
    
    try:
        # 1. Lấy thông tin cổ phiếu
        print(f"\n📊 Thông tin {symbol}:")
        info = system.get_stock_info(symbol)
        
        if info:
            print(f"💰 Giá hiện tại: {info.get('current_price', 'N/A'):,} VND")
            print(f"📈 Thay đổi: {info.get('change_percent', 'N/A')}%")
            print(f"📊 Khối lượng: {info.get('volume', 'N/A'):,}")
            print(f"🏢 Công ty: {info.get('company_name', 'N/A')}")
        
        # 2. Lấy dữ liệu lịch sử
        print(f"\n📈 Dữ liệu lịch sử {period}:")
        df = system.get_stock_data(symbol, period)
        
        if df is not None and len(df) > 0:
            print(f"📊 Số điểm dữ liệu: {len(df)}")
            print(f"📅 Từ {df['time'].min()} đến {df['time'].max()}")
            print(f"💰 Giá cao nhất: {df['high'].max():,.0f} VND")
            print(f"💰 Giá thấp nhất: {df['low'].min():,.0f} VND")
            
            # 3. Tính chỉ báo kỹ thuật
            print(f"\n🔧 Tính toán chỉ báo kỹ thuật...")
            df_indicators = system.calculate_indicators(df)
            
            if df_indicators is not None:
                latest = df_indicators.iloc[-1]
                print(f"📊 Chỉ báo gần nhất:")
                
                if 'rsi' in df_indicators.columns:
                    rsi = latest.get('rsi', 0)
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    print(f"   RSI: {rsi:.1f} ({rsi_status})")
                
                if 'macd' in df_indicators.columns:
                    macd = latest.get('macd', 0)
                    macd_signal = latest.get('macd_signal', 0)
                    macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                    print(f"   MACD: {macd:.3f} ({macd_trend})")
                
                if 'bb_percent' in df_indicators.columns:
                    bb_percent = latest.get('bb_percent', 0) * 100
                    bb_position = "Upper" if bb_percent > 80 else "Lower" if bb_percent < 20 else "Middle"
                    print(f"   BB%: {bb_percent:.1f}% ({bb_position})")
            
            # 4. Tạo tín hiệu giao dịch
            print(f"\n🎯 Tín hiệu giao dịch:")
            df_signals = system.generate_signals(df_indicators)
            
            if df_signals is not None:
                latest_signals = df_signals.iloc[-1]
                signal_type = latest_signals.get('signal_type', 'HOLD')
                signal_score = latest_signals.get('signal_score', 0)
                
                print(f"   Tín hiệu: {signal_type}")
                print(f"   Điểm số: {signal_score}")
                
                # Tìm tín hiệu active
                signal_columns = [col for col in df_signals.columns if 
                                any(keyword in col for keyword in ['cross', 'touch', 'divergence'])]
                
                active_signals = []
                for col in signal_columns:
                    if latest_signals.get(col, False):
                        active_signals.append(col)
                
                if active_signals:
                    print(f"   Tín hiệu active: {', '.join(active_signals)}")
            
            # 5. Tạo biểu đồ (chỉ thông báo)
            print(f"\n📊 Tạo biểu đồ...")
            fig = system.create_chart(df_signals, symbol, ['rsi'])
            
            if fig:
                print(f"   ✅ Biểu đồ đã được tạo thành công")
                print(f"   💡 Chạy ứng dụng web để xem biểu đồ tương tác")
            else:
                print(f"   ❌ Không thể tạo biểu đồ")
            
            # 6. Phân tích tổng quan
            print(f"\n📋 Tóm tắt phân tích:")
            
            # Tính toán return
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_return = ((last_price / first_price) - 1) * 100
            
            print(f"   📈 Tổng return: {total_return:.2f}%")
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * (252**0.5) * 100  # Annualized
            print(f"   📊 Volatility (năm): {volatility:.1f}%")
            
            # Xu hướng
            if 'sma_20' in df_indicators.columns and 'sma_50' in df_indicators.columns:
                sma20 = latest.get('sma_20', 0)
                sma50 = latest.get('sma_50', 0)
                trend = "Tăng" if sma20 > sma50 else "Giảm"
                print(f"   📊 Xu hướng: {trend}")
        
        else:
            print("❌ Không thể lấy dữ liệu lịch sử")
    
    except Exception as e:
        print(f"❌ Lỗi khi phân tích: {e}")
        logging.error(f"Lỗi phân tích {symbol}: {e}")

def compare_stocks(symbols: list = ["VCB", "BID", "CTG"], period: str = "3mo"):
    """
    So sánh nhiều cổ phiếu
    
    Args:
        symbols: Danh sách mã cổ phiếu
        period: Khoảng thời gian
    """
    if not SYSTEM_AVAILABLE:
        return
    
    print(f"\n🔄 So sánh {len(symbols)} cổ phiếu: {', '.join(symbols)}")
    
    system = StockAnalysisSystem()
    results = {}
    
    for symbol in symbols:
        try:
            df = system.get_stock_data(symbol, period)
            if df is not None and len(df) > 0:
                first_price = df['close'].iloc[0]
                last_price = df['close'].iloc[-1]
                total_return = ((last_price / first_price) - 1) * 100
                
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * (252**0.5) * 100
                
                results[symbol] = {
                    'return': total_return,
                    'volatility': volatility,
                    'current_price': last_price
                }
                
                print(f"   {symbol}: Return {total_return:+.1f}%, Vol {volatility:.1f}%, Giá {last_price:,.0f}")
        
        except Exception as e:
            print(f"   ❌ Lỗi với {symbol}: {e}")
    
    # Tìm best performer
    if results:
        best_return = max(results.items(), key=lambda x: x[1]['return'])
        lowest_vol = min(results.items(), key=lambda x: x[1]['volatility'])
        
        print(f"\n🏆 Kết quả:")
        print(f"   📈 Return cao nhất: {best_return[0]} ({best_return[1]['return']:+.1f}%)")
        print(f"   📊 Volatility thấp nhất: {lowest_vol[0]} ({lowest_vol[1]['volatility']:.1f}%)")

def main():
    """
    Hàm chính
    """
    setup_logging()
    
    print("🚀 HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print("=" * 50)
    print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not SYSTEM_AVAILABLE:
        print("\n❌ Hệ thống không khả dụng!")
        print("💡 Vui lòng:")
        print("   1. Cài đặt requirements: pip install -r requirements.txt")
        print("   2. Kiểm tra kết nối internet")
        print("   3. Chạy lại script")
        return
    
    try:
        # 1. Phân tích cổ phiếu chính
        analyze_stock("VCB", "6mo")
        
        # 2. So sánh các ngân hàng
        compare_stocks(["VCB", "BID", "CTG", "VPB"], "3mo")
        
        # 3. Phân tích cổ phiếu công nghệ
        print(f"\n{'='*50}")
        analyze_stock("FPT", "3mo")
        
        print(f"\n🎉 Hoàn thành phân tích!")
        print(f"💡 Để xem biểu đồ tương tác, chạy: streamlit run web_app/app.py")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        logging.error(f"Lỗi chính: {e}")

if __name__ == "__main__":
    main()
