"""
🔍 MARKET SCANNER - Quét thị trường và phân tích tín hiệu
======================================================

Module quét toàn bộ thị trường chứng khoán, phân tích tín hiệu giao dịch
và xếp hạng cổ phiếu theo độ mạnh tín hiệu và thanh khoản
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..data.data_provider import DataProvider
from .indicators import TechnicalIndicators
from .signals import TradingSignals


class MarketScanner:
    """
    Quét thị trường và phân tích tín hiệu giao dịch
    """
    
    def __init__(self, data_provider: DataProvider = None):
        """
        Khởi tạo Market Scanner
        
        Args:
            data_provider: Provider để lấy dữ liệu thị trường
        """
        self.logger = logging.getLogger(__name__)
        
        # Import config
        from ..utils.config import ConfigManager
        config = ConfigManager()
        
        self.data_provider = data_provider or DataProvider(config)
        self.indicators = TechnicalIndicators()
        self.signals = TradingSignals()
        
        # Danh sách cổ phiếu VN30 và các cổ phiếu lớn
        self.top_stocks = [
            # VN30
            'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
            'KDH', 'KHG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB',
            'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIC', 'VJC', 'VNM', 'VRE',
            
            # Các cổ phiếu lớn khác
            'DGC', 'VPB', 'PDR', 'NVL', 'VGC', 'PNJ', 'DXG', 'REE', 'GMD', 'HAG',
            'HNG', 'LPB', 'OCB', 'VND', 'EIB', 'TPB', 'DPM', 'DCM', 'HSG', 'TNG'
        ]
        
        # Thông số phân tích
        self.signal_weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bb': 0.15,
            'ma_cross': 0.2,
            'volume': 0.1,
            'momentum': 0.1
        }
    
    def get_market_data(self, symbols: List[str], period: str = '3mo') -> Dict[str, pd.DataFrame]:
        """
        Lấy dữ liệu cho danh sách cổ phiếu
        
        Args:
            symbols: Danh sách mã cổ phiếu
            period: Khoảng thời gian
            
        Returns:
            Dictionary với key là symbol, value là DataFrame
        """
        market_data = {}
        failed_symbols = []
        
        def fetch_stock_data(symbol):
            try:
                # Chuyển đổi period thành start_date, end_date
                from datetime import datetime, timedelta
                
                end_date = datetime.now()
                
                if period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(days=365)
                
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                # Gọi method đúng của DataProvider
                df = self.data_provider.get_historical_data(symbol, start_str, end_str)
                
                if df is not None and len(df) > 50:  # Đủ dữ liệu để phân tích
                    return symbol, df
                else:
                    failed_symbols.append(symbol)
                    return symbol, None
            except Exception as e:
                self.logger.warning(f"Không thể lấy dữ liệu {symbol}: {e}")
                failed_symbols.append(symbol)
                return symbol, None
        
        # Sử dụng ThreadPoolExecutor để tăng tốc
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(fetch_stock_data, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    market_data[symbol] = data
                    self.logger.info(f"✅ Đã lấy dữ liệu {symbol}")
                else:
                    self.logger.warning(f"❌ Không có dữ liệu {symbol}")
                
                # Delay nhỏ để tránh rate limit
                time.sleep(0.1)
        
        if failed_symbols:
            self.logger.warning(f"Không lấy được dữ liệu: {failed_symbols}")
        
        return market_data
    
    def calculate_stock_signals(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Tính toán tín hiệu giao dịch cho một cổ phiếu
        
        Args:
            df: DataFrame chứa dữ liệu giá
            symbol: Mã cổ phiếu
            
        Returns:
            Dictionary chứa các tín hiệu và điểm số
        """
        try:
            # Tính các chỉ báo kỹ thuật
            df_with_indicators = self.indicators.calculate_all(df)
            
            if df_with_indicators is None or len(df_with_indicators) == 0:
                return None
            
            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
            
            signals = {
                'symbol': symbol,
                'current_price': latest.get('close', 0),
                'volume': latest.get('volume', 0),
                'signals': {},
                'scores': {},
                'overall_signal': 'HOLD',
                'overall_score': 0,
                'liquidity_rank': 0
            }
            
            # 1. RSI Signal
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                signals['signals']['rsi'] = 'BUY'
                signals['scores']['rsi'] = 1.0
            elif rsi > 70:
                signals['signals']['rsi'] = 'SELL'
                signals['scores']['rsi'] = -1.0
            else:
                signals['signals']['rsi'] = 'HOLD'
                signals['scores']['rsi'] = 0.0
            
            # 2. MACD Signal
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            prev_macd = prev.get('macd', 0)
            prev_macd_signal = prev.get('macd_signal', 0)
            
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals['signals']['macd'] = 'BUY'
                signals['scores']['macd'] = 1.0
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals['signals']['macd'] = 'SELL'
                signals['scores']['macd'] = -1.0
            else:
                signals['signals']['macd'] = 'HOLD'
                signals['scores']['macd'] = 0.0
            
            # 3. Bollinger Bands Signal
            bb_percent = latest.get('bb_percent', 0.5)
            if bb_percent < 0:
                signals['signals']['bb'] = 'BUY'
                signals['scores']['bb'] = 1.0
            elif bb_percent > 1:
                signals['signals']['bb'] = 'SELL'
                signals['scores']['bb'] = -1.0
            else:
                signals['signals']['bb'] = 'HOLD'
                signals['scores']['bb'] = 0.0
            
            # 4. Moving Average Crossover
            ma20 = latest.get('sma_20', 0)
            ma50 = latest.get('sma_50', 0)
            prev_ma20 = prev.get('sma_20', 0)
            prev_ma50 = prev.get('sma_50', 0)
            
            if ma20 > ma50 and prev_ma20 <= prev_ma50:
                signals['signals']['ma_cross'] = 'BUY'
                signals['scores']['ma_cross'] = 1.0
            elif ma20 < ma50 and prev_ma20 >= prev_ma50:
                signals['signals']['ma_cross'] = 'SELL'
                signals['scores']['ma_cross'] = -1.0
            else:
                signals['signals']['ma_cross'] = 'HOLD'
                signals['scores']['ma_cross'] = 0.0
            
            # 5. Volume Analysis
            avg_volume = df_with_indicators['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = latest.get('volume', 0) / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                signals['signals']['volume'] = 'STRONG'
                signals['scores']['volume'] = 1.0
            elif volume_ratio < 0.5:
                signals['signals']['volume'] = 'WEAK'
                signals['scores']['volume'] = -0.5
            else:
                signals['signals']['volume'] = 'NORMAL'
                signals['scores']['volume'] = 0.0
            
            # 6. Momentum Analysis
            price_change_5d = (latest.get('close', 0) / df_with_indicators['close'].iloc[-6] - 1) * 100 if len(df_with_indicators) > 5 else 0
            
            if price_change_5d > 5:
                signals['signals']['momentum'] = 'STRONG_UP'
                signals['scores']['momentum'] = 1.0
            elif price_change_5d < -5:
                signals['signals']['momentum'] = 'STRONG_DOWN'
                signals['scores']['momentum'] = -1.0
            else:
                signals['signals']['momentum'] = 'NEUTRAL'
                signals['scores']['momentum'] = 0.0
            
            # Tính điểm tổng thể
            overall_score = sum(
                signals['scores'][signal] * self.signal_weights[signal]
                for signal in self.signal_weights.keys()
                if signal in signals['scores']
            )
            
            signals['overall_score'] = overall_score
            
            # Xác định tín hiệu tổng thể
            if overall_score > 0.3:
                signals['overall_signal'] = 'BUY'
            elif overall_score < -0.3:
                signals['overall_signal'] = 'SELL'
            else:
                signals['overall_signal'] = 'HOLD'
            
            # Tính thanh khoản (dựa trên volume trung bình)
            signals['avg_volume'] = avg_volume
            signals['liquidity_rank'] = volume_ratio
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính tín hiệu cho {symbol}: {e}")
            return None
    
    def scan_market(self, symbols: List[str] = None, period: str = '3mo') -> pd.DataFrame:
        """
        Quét toàn bộ thị trường và tạo báo cáo tín hiệu
        
        Args:
            symbols: Danh sách cổ phiếu cần quét (None để dùng danh sách mặc định)
            period: Khoảng thời gian phân tích
            
        Returns:
            DataFrame chứa kết quả phân tích và xếp hạng
        """
        if symbols is None:
            symbols = self.top_stocks
        
        self.logger.info(f"🔍 Bắt đầu quét {len(symbols)} cổ phiếu...")
        
        # Lấy dữ liệu thị trường
        market_data = self.get_market_data(symbols, period)
        
        if not market_data:
            self.logger.error("❌ Không có dữ liệu để phân tích")
            return pd.DataFrame()
        
        # Phân tích tín hiệu cho từng cổ phiếu
        all_signals = []
        
        for symbol, df in market_data.items():
            self.logger.info(f"📊 Phân tích {symbol}...")
            signals = self.calculate_stock_signals(df, symbol)
            
            if signals:
                all_signals.append(signals)
        
        if not all_signals:
            self.logger.error("❌ Không có tín hiệu nào được tạo")
            return pd.DataFrame()
        
        # Tạo DataFrame kết quả
        results_df = self._create_results_dataframe(all_signals)
        
        # Xếp hạng
        results_df = self._rank_stocks(results_df)
        
        self.logger.info(f"✅ Hoàn thành quét thị trường. Tìm thấy {len(results_df)} cổ phiếu.")
        
        return results_df
    
    def _create_results_dataframe(self, signals_list: List[Dict]) -> pd.DataFrame:
        """
        Tạo DataFrame từ danh sách tín hiệu
        """
        rows = []
        
        for signals in signals_list:
            row = {
                'Symbol': signals['symbol'],
                'Price': signals['current_price'],
                'Volume': signals['volume'],
                'Avg_Volume': signals.get('avg_volume', 0),
                'Overall_Signal': signals['overall_signal'],
                'Overall_Score': signals['overall_score'],
                'Liquidity_Ratio': signals['liquidity_rank'],
                
                # Chi tiết tín hiệu
                'RSI_Signal': signals['signals'].get('rsi', 'HOLD'),
                'MACD_Signal': signals['signals'].get('macd', 'HOLD'),
                'BB_Signal': signals['signals'].get('bb', 'HOLD'),
                'MA_Cross_Signal': signals['signals'].get('ma_cross', 'HOLD'),
                'Volume_Signal': signals['signals'].get('volume', 'NORMAL'),
                'Momentum_Signal': signals['signals'].get('momentum', 'NEUTRAL'),
                
                # Điểm số chi tiết
                'RSI_Score': signals['scores'].get('rsi', 0),
                'MACD_Score': signals['scores'].get('macd', 0),
                'BB_Score': signals['scores'].get('bb', 0),
                'MA_Score': signals['scores'].get('ma_cross', 0),
                'Volume_Score': signals['scores'].get('volume', 0),
                'Momentum_Score': signals['scores'].get('momentum', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Format số
        df['Price'] = df['Price'].round(0)
        df['Volume'] = df['Volume'].astype(int)
        df['Avg_Volume'] = df['Avg_Volume'].astype(int)
        df['Overall_Score'] = df['Overall_Score'].round(3)
        df['Liquidity_Ratio'] = df['Liquidity_Ratio'].round(2)
        
        return df
    
    def _rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xếp hạng cổ phiếu theo tín hiệu và thanh khoản
        """
        # Xếp hạng theo điểm tín hiệu
        df['Signal_Rank'] = df['Overall_Score'].rank(ascending=False)
        
        # Xếp hạng theo thanh khoản
        df['Liquidity_Rank'] = df['Liquidity_Ratio'].rank(ascending=False)
        
        # Điểm tổng hợp (70% tín hiệu, 30% thanh khoản)
        df['Combined_Score'] = (
            df['Overall_Score'] * 0.7 + 
            (df['Liquidity_Ratio'] / df['Liquidity_Ratio'].max()) * 0.3
        )
        
        df['Final_Rank'] = df['Combined_Score'].rank(ascending=False)
        
        # Sắp xếp theo xếp hạng cuối cùng
        df = df.sort_values('Final_Rank')
        
        return df
    
    def get_top_buy_signals(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Lấy top cổ phiếu có tín hiệu mua mạnh nhất
        """
        buy_signals = df[df['Overall_Signal'] == 'BUY'].copy()
        return buy_signals.head(top_n)
    
    def get_top_sell_signals(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Lấy top cổ phiếu có tín hiệu bán mạnh nhất
        """
        sell_signals = df[df['Overall_Signal'] == 'SELL'].copy()
        return sell_signals.head(top_n)
    
    def get_high_liquidity_stocks(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Lấy cổ phiếu có thanh khoản cao nhất
        """
        return df.nlargest(top_n, 'Liquidity_Ratio')
    
    def export_market_report(self, df: pd.DataFrame, filepath: str = None) -> str:
        """
        Xuất báo cáo thị trường ra file Excel
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"market_analysis_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Tổng quan thị trường
                df.to_excel(writer, sheet_name='Market_Overview', index=False)
                
                # Top tín hiệu mua
                buy_signals = self.get_top_buy_signals(df)
                buy_signals.to_excel(writer, sheet_name='Top_Buy_Signals', index=False)
                
                # Top tín hiệu bán
                sell_signals = self.get_top_sell_signals(df)
                sell_signals.to_excel(writer, sheet_name='Top_Sell_Signals', index=False)
                
                # Thanh khoản cao
                high_liquidity = self.get_high_liquidity_stocks(df)
                high_liquidity.to_excel(writer, sheet_name='High_Liquidity', index=False)
            
            self.logger.info(f"✅ Đã xuất báo cáo: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi xuất báo cáo: {e}")
            return None
    
    def print_market_summary(self, df: pd.DataFrame):
        """
        In tóm tắt thị trường
        """
        if df.empty:
            print("❌ Không có dữ liệu để hiển thị")
            return
        
        total_stocks = len(df)
        buy_signals = len(df[df['Overall_Signal'] == 'BUY'])
        sell_signals = len(df[df['Overall_Signal'] == 'SELL'])
        hold_signals = len(df[df['Overall_Signal'] == 'HOLD'])
        
        print(f"\n{'='*60}")
        print(f"📊 TÓM TẮT THỊ TRƯỜNG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"📈 Tổng số cổ phiếu phân tích: {total_stocks}")
        print(f"🟢 Tín hiệu MUA: {buy_signals} ({buy_signals/total_stocks*100:.1f}%)")
        print(f"🔴 Tín hiệu BÁN: {sell_signals} ({sell_signals/total_stocks*100:.1f}%)")
        print(f"🟡 Tín hiệu GIỮ: {hold_signals} ({hold_signals/total_stocks*100:.1f}%)")
        
        print(f"\n🏆 TOP 5 TÍN HIỆU MUA MẠNH:")
        print("-" * 40)
        top_buy = self.get_top_buy_signals(df, 5)
        for _, row in top_buy.iterrows():
            print(f"{row['Symbol']}: {row['Overall_Score']:.3f} | Giá: {row['Price']:,.0f} | Vol: {row['Liquidity_Ratio']:.1f}x")
        
        print(f"\n🔻 TOP 5 TÍN HIỆU BÁN MẠNH:")
        print("-" * 40)
        top_sell = self.get_top_sell_signals(df, 5)
        for _, row in top_sell.iterrows():
            print(f"{row['Symbol']}: {row['Overall_Score']:.3f} | Giá: {row['Price']:,.0f} | Vol: {row['Liquidity_Ratio']:.1f}x")
        
        print(f"\n💧 TOP 5 THANH KHOẢN CAO:")
        print("-" * 40)
        top_liquid = self.get_high_liquidity_stocks(df, 5)
        for _, row in top_liquid.iterrows():
            print(f"{row['Symbol']}: {row['Liquidity_Ratio']:.1f}x | Tín hiệu: {row['Overall_Signal']} | Điểm: {row['Overall_Score']:.3f}")
        
        print(f"{'='*60}\n")
