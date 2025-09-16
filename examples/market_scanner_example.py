"""
🔍 MARKET SCANNER EXAMPLE - Ví dụ quét thị trường
===============================================

Ví dụ sử dụng Market Scanner để quét và phân tích toàn bộ thị trường chứng khoán
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.analysis.market_scanner import MarketScanner
from src.data.data_provider import DataProvider
import pandas as pd

def main():
    """
    Chạy market scanner để quét và phân tích thị trường
    """
    print("🔍 Bắt đầu quét thị trường chứng khoán Việt Nam...")
    print("=" * 60)
    
    # Khởi tạo Market Scanner
    scanner = MarketScanner()
    
    # Quét thị trường (có thể chỉ định danh sách cổ phiếu cụ thể)
    # Ví dụ: quét một số cổ phiếu lớn
    test_symbols = ['VCB', 'VIC', 'VHM', 'VNM', 'HPG', 'FPT', 'CTG', 'BID', 'TCB', 'ACB']
    
    print(f"📊 Phân tích {len(test_symbols)} cổ phiếu hàng đầu...")
    
    # Quét thị trường
    results_df = scanner.scan_market(symbols=test_symbols, period='3mo')
    
    if results_df.empty:
        print("❌ Không có kết quả phân tích")
        return
    
    # In tóm tắt thị trường
    scanner.print_market_summary(results_df)
    
    # Chi tiết phân tích
    print("\n📋 CHI TIẾT PHÂN TÍCH:")
    print("=" * 80)
    
    # Hiển thị kết quả với định dạng đẹp
    display_columns = [
        'Symbol', 'Price', 'Overall_Signal', 'Overall_Score', 
        'Liquidity_Ratio', 'RSI_Signal', 'MACD_Signal', 'Volume_Signal'
    ]
    
    print(results_df[display_columns].to_string(index=False))
    
    # Lưu báo cáo Excel
    print(f"\n💾 Đang xuất báo cáo Excel...")
    excel_file = scanner.export_market_report(results_df)
    if excel_file:
        print(f"✅ Đã lưu báo cáo: {excel_file}")
    
    # Phân tích chi tiết cho các tín hiệu mua/bán
    print(f"\n🎯 PHÂN TÍCH CHI TIẾT:")
    print("-" * 40)
    
    # Top cổ phiếu mua
    top_buys = scanner.get_top_buy_signals(results_df, 3)
    if not top_buys.empty:
        print(f"\n🟢 TOP 3 CỔ PHIẾU NÊN MUA:")
        for _, stock in top_buys.iterrows():
            print(f"""
📈 {stock['Symbol']} - Điểm: {stock['Overall_Score']:.3f}
   💰 Giá: {stock['Price']:,.0f} VND
   📊 Thanh khoản: {stock['Liquidity_Ratio']:.1f}x
   🎯 Tín hiệu: RSI({stock['RSI_Signal']}) | MACD({stock['MACD_Signal']}) | BB({stock['BB_Signal']})
   📈 Động lực: {stock['Momentum_Signal']} | Khối lượng: {stock['Volume_Signal']}
""")
    
    # Top cổ phiếu bán
    top_sells = scanner.get_top_sell_signals(results_df, 3)
    if not top_sells.empty:
        print(f"\n🔴 TOP 3 CỔ PHIẾU NÊN BÁN:")
        for _, stock in top_sells.iterrows():
            print(f"""
📉 {stock['Symbol']} - Điểm: {stock['Overall_Score']:.3f}
   💰 Giá: {stock['Price']:,.0f} VND
   📊 Thanh khoản: {stock['Liquidity_Ratio']:.1f}x
   🎯 Tín hiệu: RSI({stock['RSI_Signal']}) | MACD({stock['MACD_Signal']}) | BB({stock['BB_Signal']})
   📈 Động lực: {stock['Momentum_Signal']} | Khối lượng: {stock['Volume_Signal']}
""")
    
    # Thống kê theo từng loại tín hiệu
    print(f"\n📊 THỐNG KÊ TÍN HIỆU:")
    print("-" * 30)
    
    for signal_type in ['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'Volume_Signal']:
        signal_counts = results_df[signal_type].value_counts()
        print(f"\n{signal_type.replace('_Signal', '')}:")
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} cổ phiếu")
    
    print(f"\n✅ Hoàn thành phân tích thị trường!")
    print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def analyze_full_market():
    """
    Phân tích toàn bộ thị trường (tất cả cổ phiếu)
    """
    print("🔍 PHÂN TÍCH TOÀN BỘ THỊ TRƯỜNG")
    print("=" * 60)
    
    scanner = MarketScanner()
    
    # Quét toàn bộ danh sách cổ phiếu mặc định
    results_df = scanner.scan_market(period='3mo')
    
    if results_df.empty:
        print("❌ Không có kết quả phân tích")
        return
    
    # Thống kê tổng quan
    scanner.print_market_summary(results_df)
    
    # Lưu báo cáo đầy đủ
    excel_file = scanner.export_market_report(results_df)
    if excel_file:
        print(f"✅ Đã lưu báo cáo đầy đủ: {excel_file}")
    
    return results_df

def analyze_by_sector():
    """
    Phân tích theo ngành
    """
    print("🏭 PHÂN TÍCH THEO NGÀNH")
    print("=" * 40)
    
    sectors = {
        'Ngân hàng': ['VCB', 'CTG', 'BID', 'TCB', 'ACB', 'MBB', 'VPB', 'TPB', 'STB', 'SHB'],
        'Bất động sản': ['VHM', 'VIC', 'VRE', 'NVL', 'KDH', 'DXG', 'PDR', 'HDG'],
        'Công nghệ': ['FPT', 'CMG', 'SAM', 'ELC'],
        'Tiêu dùng': ['VNM', 'SAB', 'MSN', 'MWG', 'PNJ'],
        'Công nghiệp': ['HPG', 'HSG', 'DPM', 'DCM', 'GEX']
    }
    
    scanner = MarketScanner()
    
    sector_results = {}
    
    for sector_name, symbols in sectors.items():
        print(f"\n📊 Phân tích ngành {sector_name}...")
        
        sector_df = scanner.scan_market(symbols=symbols, period='3mo')
        if not sector_df.empty:
            sector_results[sector_name] = sector_df
            
            # Thống kê ngành
            buy_count = len(sector_df[sector_df['Overall_Signal'] == 'BUY'])
            sell_count = len(sector_df[sector_df['Overall_Signal'] == 'SELL'])
            total_count = len(sector_df)
            
            print(f"   🟢 Mua: {buy_count}/{total_count} ({buy_count/total_count*100:.1f}%)")
            print(f"   🔴 Bán: {sell_count}/{total_count} ({sell_count/total_count*100:.1f}%)")
            
            # Top cổ phiếu trong ngành
            if buy_count > 0:
                top_buy = sector_df[sector_df['Overall_Signal'] == 'BUY'].iloc[0]
                print(f"   ⭐ Top mua: {top_buy['Symbol']} (điểm: {top_buy['Overall_Score']:.3f})")
    
    return sector_results

if __name__ == "__main__":
    # Chọn loại phân tích
    print("Chọn loại phân tích:")
    print("1. Phân tích nhanh (10 cổ phiếu)")
    print("2. Phân tích toàn thị trường")
    print("3. Phân tích theo ngành")
    
    choice = input("Nhập lựa chọn (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        analyze_full_market()
    elif choice == "3":
        analyze_by_sector()
    else:
        print("Chạy phân tích mặc định...")
        main()
