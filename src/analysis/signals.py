"""
🎯 SIGNAL GENERATION - Tạo tín hiệu giao dịch
===========================================

Module này tạo ra các tín hiệu giao dịch dựa trên phân tích kỹ thuật
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

class SignalType(Enum):
    """
    Loại tín hiệu giao dịch
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class SignalStrength(Enum):
    """
    Độ mạnh của tín hiệu
    """
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

class SignalResult:
    """
    Kết quả tín hiệu giao dịch
    """
    def __init__(self, signal_type: str, confidence: float, reasons: List[str] = None):
        self.signal_type = signal_type
        self.confidence = confidence
        self.reasons = reasons or []

class TradingSignals:
    """
    Lớp tạo tín hiệu giao dịch
    """
    
    def __init__(self):
        """
        Khởi tạo TradingSignals
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 TradingSignals đã được khởi tạo")
    
    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo tất cả tín hiệu giao dịch
        
        Args:
            df: DataFrame với các chỉ báo kỹ thuật
            
        Returns:
            DataFrame với các tín hiệu được thêm vào
        """
        df_result = df.copy()
        
        try:
            # Moving Average Signals
            df_result = self.moving_average_signals(df_result)
            
            # RSI Signals
            df_result = self.rsi_signals(df_result)
            
            # MACD Signals
            df_result = self.macd_signals(df_result)
            
            # Bollinger Bands Signals
            df_result = self.bollinger_signals(df_result)
            
            # Stochastic Signals
            df_result = self.stochastic_signals(df_result)
            
            # Volume Signals
            df_result = self.volume_signals(df_result)
            
            # Pattern Signals
            df_result = self.pattern_signals(df_result)
            
            # Combined Signals
            df_result = self.combined_signals(df_result)
            
            self.logger.info("✅ Đã tạo tất cả tín hiệu giao dịch")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tạo tín hiệu: {e}")
        
        return df_result
    
    def moving_average_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ Moving Averages
        """
        # Golden Cross & Death Cross
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['ma_golden_cross'] = (
                (df['sma_50'] > df['sma_200']) & 
                (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
            )
            df['ma_death_cross'] = (
                (df['sma_50'] < df['sma_200']) & 
                (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
            )
        
        # Price vs Moving Average
        if 'sma_20' in df.columns:
            df['price_above_ma20'] = df['close'] > df['sma_20']
            df['price_below_ma20'] = df['close'] < df['sma_20']
        
        # EMA Crossover
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_bullish_cross'] = (
                (df['ema_12'] > df['ema_26']) & 
                (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
            )
            df['ema_bearish_cross'] = (
                (df['ema_12'] < df['ema_26']) & 
                (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
            )
        
        return df
    
    def rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ RSI
        """
        if 'rsi' not in df.columns:
            return df
        
        # Oversold/Overbought
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        
        # RSI Divergence (simplified)
        df['rsi_bullish_divergence'] = (
            (df['close'] < df['close'].shift(5)) & 
            (df['rsi'] > df['rsi'].shift(5)) &
            (df['rsi'] < 50)
        )
        df['rsi_bearish_divergence'] = (
            (df['close'] > df['close'].shift(5)) & 
            (df['rsi'] < df['rsi'].shift(5)) &
            (df['rsi'] > 50)
        )
        
        # RSI Crossing levels
        df['rsi_buy_signal'] = (
            (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30)
        )
        df['rsi_sell_signal'] = (
            (df['rsi'] < 70) & (df['rsi'].shift(1) >= 70)
        )
        
        return df
    
    def macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ MACD
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return df
        
        # MACD Crossover
        df['macd_bullish_cross'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df['macd_bearish_cross'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # MACD Zero Line Cross
        df['macd_above_zero'] = df['macd'] > 0
        df['macd_below_zero'] = df['macd'] < 0
        
        # MACD Histogram signals
        if 'macd_histogram' in df.columns:
            df['macd_hist_increasing'] = (
                df['macd_histogram'] > df['macd_histogram'].shift(1)
            )
            df['macd_hist_decreasing'] = (
                df['macd_histogram'] < df['macd_histogram'].shift(1)
            )
        
        return df
    
    def bollinger_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ Bollinger Bands
        """
        if not all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_percent']):
            return df
        
        # Bollinger Band Squeeze
        if 'bb_bandwidth' in df.columns:
            bb_squeeze_threshold = df['bb_bandwidth'].rolling(20).quantile(0.2)
            df['bb_squeeze'] = df['bb_bandwidth'] < bb_squeeze_threshold
        
        # Price touching bands
        df['bb_upper_touch'] = df['close'] >= df['bb_upper']
        df['bb_lower_touch'] = df['close'] <= df['bb_lower']
        
        # %B signals
        df['bb_oversold'] = df['bb_percent'] < 0
        df['bb_overbought'] = df['bb_percent'] > 1
        
        # Band walk
        df['bb_upper_walk'] = df['bb_percent'] > 0.8
        df['bb_lower_walk'] = df['bb_percent'] < 0.2
        
        return df
    
    def stochastic_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ Stochastic
        """
        if not all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            return df
        
        # Stochastic Crossover
        df['stoch_bullish_cross'] = (
            (df['stoch_k'] > df['stoch_d']) & 
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        )
        df['stoch_bearish_cross'] = (
            (df['stoch_k'] < df['stoch_d']) & 
            (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        )
        
        # Oversold/Overbought levels
        df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        
        return df
    
    def volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ khối lượng
        """
        if 'volume' not in df.columns:
            return df
        
        # Volume spike
        if 'volume_sma_20' in df.columns:
            df['volume_spike'] = df['volume'] > (df['volume_sma_20'] * 2)
        
        # Volume trend
        df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
        df['volume_decreasing'] = df['volume'] < df['volume'].shift(1)
        
        # Price-Volume confirmation
        df['bullish_volume'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        )
        df['bearish_volume'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['volume'] > df['volume'].shift(1))
        )
        
        return df
    
    def pattern_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ các pattern
        """
        # Doji pattern
        body = abs(df['close'] - df['open'])
        range_size = df['high'] - df['low']
        df['doji'] = body < (range_size * 0.1)
        
        # Hammer/Shooting Star
        lower_shadow = df['low'] - df[['open', 'close']].min(axis=1)
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        df['hammer'] = (
            (lower_shadow > body * 2) & 
            (upper_shadow < body * 0.5) &
            (df['close'] < df['close'].shift(1))  # After downtrend
        )
        df['shooting_star'] = (
            (upper_shadow > body * 2) & 
            (lower_shadow < body * 0.5) &
            (df['close'] > df['close'].shift(1))  # After uptrend
        )
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['open'] < df['close']) &  # Current candle is bullish
            (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is bearish
            (df['open'] < df['close'].shift(1)) &  # Current open < previous close
            (df['close'] > df['open'].shift(1))  # Current close > previous open
        )
        df['bearish_engulfing'] = (
            (df['open'] > df['close']) &  # Current candle is bearish
            (df['open'].shift(1) < df['close'].shift(1)) &  # Previous candle is bullish
            (df['open'] > df['close'].shift(1)) &  # Current open > previous close
            (df['close'] < df['open'].shift(1))  # Current close < previous open
        )
        
        return df
    
    def combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tổng hợp tín hiệu từ nhiều chỉ báo
        """
        # Initialize signal columns
        df['signal_score'] = 0
        df['signal_type'] = SignalType.HOLD.value
        df['signal_strength'] = SignalStrength.WEAK.value
        
        # Count bullish signals
        bullish_signals = [
            'ma_golden_cross', 'price_above_ma20', 'ema_bullish_cross',
            'rsi_buy_signal', 'rsi_bullish_divergence',
            'macd_bullish_cross', 'macd_hist_increasing',
            'bb_lower_touch', 'stoch_bullish_cross',
            'bullish_volume', 'hammer', 'bullish_engulfing'
        ]
        
        # Count bearish signals
        bearish_signals = [
            'ma_death_cross', 'price_below_ma20', 'ema_bearish_cross',
            'rsi_sell_signal', 'rsi_bearish_divergence',
            'macd_bearish_cross', 'macd_hist_decreasing',
            'bb_upper_touch', 'stoch_bearish_cross',
            'bearish_volume', 'shooting_star', 'bearish_engulfing'
        ]
        
        # Calculate signal scores
        for signal in bullish_signals:
            if signal in df.columns:
                df['signal_score'] += df[signal].astype(int)
        
        for signal in bearish_signals:
            if signal in df.columns:
                df['signal_score'] -= df[signal].astype(int)
        
        # Determine signal type and strength
        def determine_signal(score):
            if score >= 4:
                return SignalType.STRONG_BUY.value, SignalStrength.VERY_STRONG.value
            elif score >= 2:
                return SignalType.BUY.value, SignalStrength.STRONG.value
            elif score >= 1:
                return SignalType.BUY.value, SignalStrength.MODERATE.value
            elif score <= -4:
                return SignalType.STRONG_SELL.value, SignalStrength.VERY_STRONG.value
            elif score <= -2:
                return SignalType.SELL.value, SignalStrength.STRONG.value
            elif score <= -1:
                return SignalType.SELL.value, SignalStrength.MODERATE.value
            else:
                return SignalType.HOLD.value, SignalStrength.WEAK.value
        
        signals = df['signal_score'].apply(determine_signal)
        df['signal_type'] = [s[0] for s in signals]
        df['signal_strength'] = [s[1] for s in signals]
        
        return df
    
    def get_latest_signals(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Lấy tín hiệu gần nhất
        
        Args:
            df: DataFrame với tín hiệu
            n: Số tín hiệu gần nhất
            
        Returns:
            DataFrame với tín hiệu gần nhất
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # Lấy các cột tín hiệu
        signal_columns = [col for col in df.columns if 
                         any(keyword in col for keyword in 
                             ['signal', 'cross', 'touch', 'divergence', 'engulfing', 'hammer', 'doji'])]
        
        latest_df = df[['time', 'close'] + signal_columns].tail(n).copy()
        
        return latest_df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Tóm tắt tín hiệu hiện tại
        
        Args:
            df: DataFrame với tín hiệu
            
        Returns:
            Dict chứa tóm tắt tín hiệu
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        
        summary = {
            'current_price': latest['close'],
            'signal_type': latest.get('signal_type', SignalType.HOLD.value),
            'signal_strength': latest.get('signal_strength', SignalStrength.WEAK.value),
            'signal_score': latest.get('signal_score', 0),
            'active_signals': []
        }
        
        # Tìm các tín hiệu đang active
        signal_columns = [col for col in df.columns if 
                         any(keyword in col for keyword in 
                             ['signal', 'cross', 'touch', 'divergence', 'engulfing', 'hammer', 'doji'])]
        
        for col in signal_columns:
            if latest.get(col, False):
                summary['active_signals'].append(col)
        
        return summary
    
    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        """
        Tạo tín hiệu giao dịch tổng hợp
        
        Args:
            df: DataFrame với các chỉ báo kỹ thuật
            
        Returns:
            SignalResult: Kết quả tín hiệu giao dịch
        """
        if df is None or len(df) == 0:
            return SignalResult('HOLD', 0.0, ['Không có dữ liệu'])
        
        try:
            # Tạo tất cả tín hiệu
            df_with_signals = self.generate_all_signals(df)
            latest = df_with_signals.iloc[-1]
            
            # Tính điểm số tổng hợp
            signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            reasons = []
            
            # RSI signals
            if 'rsi' in latest:
                rsi = latest['rsi']
                if rsi < 30:
                    signals_count['BUY'] += 1
                    reasons.append(f"RSI quá bán ({rsi:.1f})")
                elif rsi > 70:
                    signals_count['SELL'] += 1
                    reasons.append(f"RSI quá mua ({rsi:.1f})")
                else:
                    signals_count['HOLD'] += 1
            
            # MACD signals
            if 'macd' in latest and 'macd_signal' in latest:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                if macd > macd_signal:
                    signals_count['BUY'] += 1
                    reasons.append("MACD bullish crossover")
                else:
                    signals_count['SELL'] += 1
                    reasons.append("MACD bearish crossover")
            
            # Moving Average signals
            if 'sma_20' in latest and 'sma_50' in latest:
                sma_20 = latest['sma_20']
                sma_50 = latest['sma_50']
                close = latest['close']
                
                if sma_20 > sma_50 and close > sma_20:
                    signals_count['BUY'] += 1
                    reasons.append("Giá trên MA tăng")
                elif sma_20 < sma_50 and close < sma_20:
                    signals_count['SELL'] += 1
                    reasons.append("Giá dưới MA giảm")
                else:
                    signals_count['HOLD'] += 1
            
            # Bollinger Bands signals
            if all(col in latest for col in ['bb_upper', 'bb_lower', 'close']):
                close = latest['close']
                bb_upper = latest['bb_upper']
                bb_lower = latest['bb_lower']
                
                if close <= bb_lower:
                    signals_count['BUY'] += 1
                    reasons.append("Giá chạm Bollinger Band dưới")
                elif close >= bb_upper:
                    signals_count['SELL'] += 1
                    reasons.append("Giá chạm Bollinger Band trên")
                else:
                    signals_count['HOLD'] += 1
            
            # Xác định tín hiệu chủ đạo
            total_signals = sum(signals_count.values())
            if total_signals == 0:
                return SignalResult('HOLD', 0.0, ['Không có tín hiệu rõ ràng'])
            
            # Tín hiệu mạnh nhất
            dominant_signal = max(signals_count.keys(), key=lambda k: signals_count[k])
            confidence = signals_count[dominant_signal] / total_signals
            
            # Điều chỉnh confidence dựa trên số lượng tín hiệu
            if signals_count[dominant_signal] >= 3:
                confidence = min(confidence + 0.2, 1.0)  # Boost confidence
            
            return SignalResult(dominant_signal, confidence, reasons)
        
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo tín hiệu: {e}")
            return SignalResult('HOLD', 0.0, [f'Lỗi: {str(e)}'])

# Test module
if __name__ == "__main__":
    """
    Test TradingSignals
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data with indicators
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 50 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volumes = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Add some sample indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = 50 + np.random.randn(100) * 20
    df['macd'] = np.random.randn(100) * 0.5
    df['macd_signal'] = df['macd'].rolling(9).mean()
    
    print("🧪 Testing TradingSignals...")
    
    # Test signals
    signals = TradingSignals()
    df_with_signals = signals.generate_all_signals(df)
    
    print(f"✅ Số cột ban đầu: {len(df.columns)}")
    print(f"✅ Số cột sau khi tạo tín hiệu: {len(df_with_signals.columns)}")
    
    # Test signal summary
    summary = signals.get_signal_summary(df_with_signals)
    print(f"\n📊 Tóm tắt tín hiệu:")
    print(f"Giá hiện tại: {summary.get('current_price', 0):.2f}")
    print(f"Loại tín hiệu: {summary.get('signal_type', 'N/A')}")
    print(f"Độ mạnh: {summary.get('signal_strength', 'N/A')}")
    print(f"Điểm số: {summary.get('signal_score', 0)}")
    print(f"Tín hiệu active: {len(summary.get('active_signals', []))}")
    
    print("✅ Test completed!")
