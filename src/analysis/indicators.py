"""
📊 TECHNICAL INDICATORS - Chỉ báo kỹ thuật
==========================================

Module này chứa tất cả các chỉ báo kỹ thuật được sử dụng trong hệ thống
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

class TechnicalIndicators:
    """
    Lớp tính toán các chỉ báo kỹ thuật
    """
    
    def __init__(self):
        """
        Khởi tạo TechnicalIndicators
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("📈 TechnicalIndicators đã được khởi tạo")
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán tất cả chỉ báo kỹ thuật
        
        Args:
            df: DataFrame chứa dữ liệu OHLCV
            
        Returns:
            DataFrame với các chỉ báo được thêm vào
        """
        df_result = df.copy()
        
        try:
            # Moving Averages
            df_result = self.calculate_moving_averages(df_result)
            
            # Momentum Indicators
            df_result = self.calculate_rsi(df_result)
            df_result = self.calculate_stochastic(df_result)
            df_result = self.calculate_williams_r(df_result)
            df_result = self.calculate_roc(df_result)
            
            # Trend Indicators
            df_result = self.calculate_macd(df_result)
            df_result = self.calculate_adx(df_result)
            df_result = self.calculate_parabolic_sar(df_result)
            
            # Volatility Indicators
            df_result = self.calculate_bollinger_bands(df_result)
            df_result = self.calculate_atr(df_result)
            df_result = self.calculate_keltner_channels(df_result)
            
            # Volume Indicators
            df_result = self.calculate_volume_indicators(df_result)
            
            self.logger.info(f"✅ Đã tính toán {len(self.get_available_indicators())} chỉ báo")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tính chỉ báo: {e}")
        
        return df_result
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán Moving Averages
        """
        # Simple Moving Averages
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        ema_periods = [5, 10, 12, 20, 26, 50]
        for period in ema_periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Weighted Moving Average
        if len(df) >= 20:
            weights = np.arange(1, 21)
            df['wma_20'] = df['close'].rolling(window=20).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Tính toán RSI (Relative Strength Index)
        """
        if len(df) < period:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                           d_period: int = 3) -> pd.DataFrame:
        """
        Tính toán Stochastic Oscillator
        """
        if len(df) < k_period:
            return df
        
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Tính toán Williams %R
        """
        if len(df) < period:
            return df
        
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * ((high_max - df['close']) / (high_max - low_min))
        
        return df
    
    def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """
        Tính toán Rate of Change
        """
        if len(df) < period:
            return df
        
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Tính toán MACD
        """
        if len(df) < slow:
            return df
        
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Tính toán ADX (Average Directional Index)
        """
        if len(df) < period + 1:
            return df
        
        # True Range
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Directional Movement
        plus_dm = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                          np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        minus_dm = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                           np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        # Smoothed values
        tr_smooth = pd.Series(tr).rolling(window=period).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def calculate_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02,
                               af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        Tính toán Parabolic SAR
        """
        if len(df) < 2:
            return df
        
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af = np.zeros(len(df))
        ep = np.zeros(len(df))
        
        # Initialize
        sar[0] = df['low'].iloc[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = af_start
        ep[0] = df['high'].iloc[0]
        
        for i in range(1, len(df)):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            if prev_trend == 1:  # Uptrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                if current_low <= sar[i]:
                    # Trend reversal
                    trend[i] = -1
                    sar[i] = prev_ep
                    af[i] = af_start
                    ep[i] = current_low
                else:
                    trend[i] = 1
                    if current_high > prev_ep:
                        ep[i] = current_high
                        af[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # Downtrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                if current_high >= sar[i]:
                    # Trend reversal
                    trend[i] = 1
                    sar[i] = prev_ep
                    af[i] = af_start
                    ep[i] = current_high
                else:
                    trend[i] = -1
                    if current_low < prev_ep:
                        ep[i] = current_low
                        af[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
        
        df['parabolic_sar'] = sar
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20,
                                 std_dev: float = 2) -> pd.DataFrame:
        """
        Tính toán Bollinger Bands
        """
        if len(df) < period:
            return df
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # %B (Position within bands)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Tính toán Average True Range
        """
        if len(df) < 2:
            return df
        
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        df['atr'] = pd.Series(tr).rolling(window=period).mean()
        
        return df
    
    def calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20,
                                  multiplier: float = 2) -> pd.DataFrame:
        """
        Tính toán Keltner Channels
        """
        if len(df) < period:
            return df
        
        # Calculate ATR first
        df = self.calculate_atr(df, period)
        
        df['kc_middle'] = df['close'].ewm(span=period).mean()
        df['kc_upper'] = df['kc_middle'] + (multiplier * df['atr'])
        df['kc_lower'] = df['kc_middle'] - (multiplier * df['atr'])
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các chỉ báo khối lượng
        """
        # Volume Moving Averages
        periods = [10, 20, 50]
        for period in periods:
            if len(df) >= period:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # On Balance Volume (OBV)
        if len(df) > 1:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            df['obv'] = obv
        
        # Volume Rate of Change
        if len(df) >= 12:
            df['volume_roc'] = ((df['volume'] - df['volume'].shift(12)) / df['volume'].shift(12)) * 100
        
        # Accumulation/Distribution Line
        if len(df) > 0:
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['ad_line'] = money_flow_volume.cumsum()
        
        return df
    
    def get_available_indicators(self) -> List[str]:
        """
        Lấy danh sách các chỉ báo có sẵn
        
        Returns:
            List các tên chỉ báo
        """
        return [
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_12', 'ema_20', 'ema_26', 'ema_50',
            'wma_20',
            
            # Momentum
            'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'roc',
            
            # Trend
            'macd', 'macd_signal', 'macd_histogram',
            'adx', 'plus_di', 'minus_di', 'parabolic_sar',
            
            # Volatility
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent', 'bb_bandwidth',
            'atr', 'kc_upper', 'kc_middle', 'kc_lower',
            
            # Volume
            'volume_sma_10', 'volume_sma_20', 'volume_sma_50',
            'obv', 'volume_roc', 'ad_line'
        ]
    
    def get_indicator_description(self, indicator: str) -> str:
        """
        Lấy mô tả của chỉ báo
        
        Args:
            indicator: Tên chỉ báo
            
        Returns:
            Mô tả chỉ báo
        """
        descriptions = {
            'rsi': 'Relative Strength Index - Chỉ báo momentum (0-100)',
            'macd': 'Moving Average Convergence Divergence - Chỉ báo xu hướng',
            'bb_upper': 'Bollinger Bands Upper - Kháng cự động',
            'bb_lower': 'Bollinger Bands Lower - Hỗ trợ động',
            'atr': 'Average True Range - Đo độ biến động',
            'adx': 'Average Directional Index - Đo sức mạnh xu hướng',
            'obv': 'On Balance Volume - Khối lượng tích lũy',
            'stoch_k': 'Stochastic %K - Chỉ báo momentum',
            'parabolic_sar': 'Parabolic SAR - Điểm dừng và đảo chiều'
        }
        
        return descriptions.get(indicator, f'Chỉ báo {indicator}')

# Test module
if __name__ == "__main__":
    """
    Test TechnicalIndicators
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
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
    
    print("🧪 Testing TechnicalIndicators...")
    
    # Test indicators
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.calculate_all(df)
    
    print(f"✅ Số cột ban đầu: {len(df.columns)}")
    print(f"✅ Số cột sau khi tính chỉ báo: {len(df_with_indicators.columns)}")
    print(f"📊 Chỉ báo có sẵn: {len(indicators.get_available_indicators())}")
    
    # Test specific indicators
    latest = df_with_indicators.iloc[-1]
    print(f"\n📈 Giá trị chỉ báo gần nhất:")
    print(f"RSI: {latest.get('rsi', 'N/A'):.2f}")
    print(f"MACD: {latest.get('macd', 'N/A'):.2f}")
    print(f"BB %: {latest.get('bb_percent', 'N/A'):.2f}")
    print(f"ATR: {latest.get('atr', 'N/A'):.2f}")
    
    print("✅ Test completed!")
