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
            
            # Ichimoku Signals
            df_result = self.ichimoku_signals(df_result)
            
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
            sma_50_curr = df['sma_50'].fillna(0)
            sma_200_curr = df['sma_200'].fillna(0)
            sma_50_prev = df['sma_50'].shift(1).fillna(0)
            sma_200_prev = df['sma_200'].shift(1).fillna(0)
            
            df['ma_golden_cross'] = (
                (sma_50_curr > sma_200_curr) & 
                (sma_50_prev <= sma_200_prev)
            )
            df['ma_death_cross'] = (
                (sma_50_curr < sma_200_curr) & 
                (sma_50_prev >= sma_200_prev)
            )
        
        # Price vs Moving Average
        if 'sma_20' in df.columns:
            df['price_above_ma20'] = df['close'].fillna(0) > df['sma_20'].fillna(0)
            df['price_below_ma20'] = df['close'].fillna(0) < df['sma_20'].fillna(0)
        
        # EMA Crossover
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            ema_12_curr = df['ema_12'].fillna(0)
            ema_26_curr = df['ema_26'].fillna(0)
            ema_12_prev = df['ema_12'].shift(1).fillna(0)
            ema_26_prev = df['ema_26'].shift(1).fillna(0)
            
            df['ema_bullish_cross'] = (
                (ema_12_curr > ema_26_curr) & 
                (ema_12_prev <= ema_26_prev)
            )
            df['ema_bearish_cross'] = (
                (ema_12_curr < ema_26_curr) & 
                (ema_12_prev >= ema_26_prev)
            )
        
        return df
    
    def rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ RSI
        """
        if 'rsi' not in df.columns:
            return df
        
        rsi_curr = df['rsi'].fillna(50)
        rsi_prev = df['rsi'].shift(1).fillna(50)
        close_curr = df['close'].fillna(0)
        close_prev = df['close'].shift(5).fillna(0)
        rsi_prev_5 = df['rsi'].shift(5).fillna(50)
        
        # Oversold/Overbought
        df['rsi_oversold'] = rsi_curr < 30
        df['rsi_overbought'] = rsi_curr > 70
        
        # RSI Divergence (simplified)
        df['rsi_bullish_divergence'] = (
            (close_curr < close_prev) & 
            (rsi_curr > rsi_prev_5) &
            (rsi_curr < 50)
        )
        df['rsi_bearish_divergence'] = (
            (close_curr > close_prev) & 
            (rsi_curr < rsi_prev_5) &
            (rsi_curr > 50)
        )
        
        # RSI Crossing levels
        df['rsi_buy_signal'] = (
            (rsi_curr > 30) & (rsi_prev <= 30)
        )
        df['rsi_sell_signal'] = (
            (rsi_curr < 70) & (rsi_prev >= 70)
        )
        
        return df
    
    def macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ MACD
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return df
        
        macd_curr = df['macd'].fillna(0)
        macd_signal_curr = df['macd_signal'].fillna(0)
        macd_prev = df['macd'].shift(1).fillna(0)
        macd_signal_prev = df['macd_signal'].shift(1).fillna(0)
        
        # MACD Crossover
        df['macd_bullish_cross'] = (
            (macd_curr > macd_signal_curr) & 
            (macd_prev <= macd_signal_prev)
        )
        df['macd_bearish_cross'] = (
            (macd_curr < macd_signal_curr) & 
            (macd_prev >= macd_signal_prev)
        )
        
        # MACD Zero Line Cross
        df['macd_above_zero'] = macd_curr > 0
        df['macd_below_zero'] = macd_curr < 0
        
        # MACD Histogram signals
        if 'macd_histogram' in df.columns:
            macd_hist_curr = df['macd_histogram'].fillna(0)
            macd_hist_prev = df['macd_histogram'].shift(1).fillna(0)
            
            df['macd_hist_increasing'] = macd_hist_curr > macd_hist_prev
            df['macd_hist_decreasing'] = macd_hist_curr < macd_hist_prev
        
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
        
        stoch_k_curr = df['stoch_k'].fillna(50)
        stoch_d_curr = df['stoch_d'].fillna(50)
        stoch_k_prev = df['stoch_k'].shift(1).fillna(50)
        stoch_d_prev = df['stoch_d'].shift(1).fillna(50)
        
        # Stochastic Crossover
        df['stoch_bullish_cross'] = (
            (stoch_k_curr > stoch_d_curr) & 
            (stoch_k_prev <= stoch_d_prev)
        )
        df['stoch_bearish_cross'] = (
            (stoch_k_curr < stoch_d_curr) & 
            (stoch_k_prev >= stoch_d_prev)
        )
        
        # Oversold/Overbought levels
        df['stoch_oversold'] = (stoch_k_curr < 20) & (stoch_d_curr < 20)
        df['stoch_overbought'] = (stoch_k_curr > 80) & (stoch_d_curr > 80)
        
        return df
    
    def ichimoku_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ Ichimoku Cloud
        """
        required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        if not all(col in df.columns for col in required_cols):
            return df
        
        tenkan_curr = df['tenkan_sen'].fillna(0)
        kijun_curr = df['kijun_sen'].fillna(0)
        tenkan_prev = df['tenkan_sen'].shift(1).fillna(0)
        kijun_prev = df['kijun_sen'].shift(1).fillna(0)
        close_curr = df['close'].fillna(0)
        close_prev = df['close'].shift(1).fillna(0)
        
        # Tenkan-sen vs Kijun-sen crossover
        df['tenkan_kijun_bullish'] = (
            (tenkan_curr > kijun_curr) & 
            (tenkan_prev <= kijun_prev)
        )
        df['tenkan_kijun_bearish'] = (
            (tenkan_curr < kijun_curr) & 
            (tenkan_prev >= kijun_prev)
        )
        
        # Price vs Cloud (Kumo)
        df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1).fillna(0)
        df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1).fillna(0)
        cloud_top_curr = df['cloud_top']
        cloud_bottom_curr = df['cloud_bottom']
        cloud_top_prev = df['cloud_top'].shift(1).fillna(0)
        cloud_bottom_prev = df['cloud_bottom'].shift(1).fillna(0)
        
        df['price_above_cloud'] = close_curr > cloud_top_curr
        df['price_below_cloud'] = close_curr < cloud_bottom_curr
        df['price_in_cloud'] = (close_curr >= cloud_bottom_curr) & (close_curr <= cloud_top_curr)
        
        # Cloud breakout signals
        df['cloud_breakout_bullish'] = (
            (close_curr > cloud_top_curr) & 
            (close_prev <= cloud_top_prev)
        )
        df['cloud_breakout_bearish'] = (
            (close_curr < cloud_bottom_curr) & 
            (close_prev >= cloud_bottom_prev)
        )
        
        # Cloud color (direction)
        df['cloud_bullish'] = df['senkou_span_a'].fillna(0) > df['senkou_span_b'].fillna(0)
        df['cloud_bearish'] = df['senkou_span_a'].fillna(0) < df['senkou_span_b'].fillna(0)
        
        # Chikou Span signals (lagging span)
        chikou_span = df['chikou_span'].fillna(0)
        close_26_ago = df['close'].shift(26).fillna(0)
        df['chikou_above_price'] = chikou_span > close_26_ago
        df['chikou_below_price'] = chikou_span < close_26_ago
        
        # Strong Ichimoku signals (multiple confirmations)
        df['ichimoku_strong_bullish'] = (
            df['tenkan_kijun_bullish'].fillna(False) & 
            df['price_above_cloud'].fillna(False) & 
            df['cloud_bullish'].fillna(False) &
            df['chikou_above_price'].fillna(False)
        )
        df['ichimoku_strong_bearish'] = (
            df['tenkan_kijun_bearish'].fillna(False) & 
            df['price_below_cloud'].fillna(False) & 
            df['cloud_bearish'].fillna(False) &
            df['chikou_below_price'].fillna(False)
        )
        
        # Trend continuation signals
        df['ichimoku_uptrend'] = (
            df['price_above_cloud'].fillna(False) & 
            (tenkan_curr > kijun_curr) &
            df['cloud_bullish'].fillna(False)
        )
        df['ichimoku_downtrend'] = (
            df['price_below_cloud'].fillna(False) & 
            (tenkan_curr < kijun_curr) &
            df['cloud_bearish'].fillna(False)
        )
        
        return df
    
    def volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ khối lượng
        """
        if 'volume' not in df.columns:
            return df
        
        volume_curr = df['volume'].fillna(0)
        volume_prev = df['volume'].shift(1).fillna(0)
        close_curr = df['close'].fillna(0)
        close_prev = df['close'].shift(1).fillna(0)
        
        # Volume spike
        if 'volume_sma_20' in df.columns:
            volume_sma_20 = df['volume_sma_20'].fillna(0)
            df['volume_spike'] = volume_curr > (volume_sma_20 * 2)
        
        # Volume trend
        df['volume_increasing'] = volume_curr > volume_prev
        df['volume_decreasing'] = volume_curr < volume_prev
        
        # Price-Volume confirmation
        df['bullish_volume'] = (
            (close_curr > close_prev) & 
            (volume_curr > volume_prev)
        )
        df['bearish_volume'] = (
            (close_curr < close_prev) & 
            (volume_curr > volume_prev)
        )
        
        return df
    
    def pattern_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tín hiệu từ các pattern
        """
        # Fill NaN values for OHLC data
        open_curr = df['open'].fillna(0)
        high_curr = df['high'].fillna(0)
        low_curr = df['low'].fillna(0)
        close_curr = df['close'].fillna(0)
        open_prev = df['open'].shift(1).fillna(0)
        close_prev = df['close'].shift(1).fillna(0)
        
        # Doji pattern
        body = abs(close_curr - open_curr)
        range_size = high_curr - low_curr
        df['doji'] = body < (range_size * 0.1)
        
        # Hammer/Shooting Star
        min_oc = pd.concat([open_curr, close_curr], axis=1).min(axis=1)
        max_oc = pd.concat([open_curr, close_curr], axis=1).max(axis=1)
        lower_shadow = low_curr - min_oc
        upper_shadow = high_curr - max_oc
        
        df['hammer'] = (
            (lower_shadow > body * 2) & 
            (upper_shadow < body * 0.5) &
            (close_curr < close_prev)  # After downtrend
        )
        df['shooting_star'] = (
            (upper_shadow > body * 2) & 
            (lower_shadow < body * 0.5) &
            (close_curr > close_prev)  # After uptrend
        )
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (open_curr < close_curr) &  # Current candle is bullish
            (open_prev > close_prev) &  # Previous candle is bearish
            (open_curr < close_prev) &  # Current open < previous close
            (close_curr > open_prev)  # Current close > previous open
        )
        df['bearish_engulfing'] = (
            (open_curr > close_curr) &  # Current candle is bearish
            (open_prev < close_prev) &  # Previous candle is bullish
            (open_curr > close_prev) &  # Current open > previous close
            (close_curr < open_prev)  # Current close < previous open
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
            'tenkan_kijun_bullish', 'cloud_breakout_bullish', 'ichimoku_strong_bullish', 'ichimoku_uptrend',
            'bullish_volume', 'hammer', 'bullish_engulfing'
        ]
        
        # Count bearish signals
        bearish_signals = [
            'ma_death_cross', 'price_below_ma20', 'ema_bearish_cross',
            'rsi_sell_signal', 'rsi_bearish_divergence',
            'macd_bearish_cross', 'macd_hist_decreasing',
            'bb_upper_touch', 'stoch_bearish_cross',
            'tenkan_kijun_bearish', 'cloud_breakout_bearish', 'ichimoku_strong_bearish', 'ichimoku_downtrend',
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
            
            # Ichimoku signals
            if all(col in latest for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
                tenkan = latest['tenkan_sen']
                kijun = latest['kijun_sen']
                close = latest['close']
                cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
                cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
                
                # Tenkan vs Kijun
                if tenkan > kijun:
                    signals_count['BUY'] += 1
                    reasons.append("Tenkan-sen trên Kijun-sen")
                else:
                    signals_count['SELL'] += 1
                    reasons.append("Tenkan-sen dưới Kijun-sen")
                
                # Price vs Cloud
                if close > cloud_top:
                    signals_count['BUY'] += 1
                    reasons.append("Giá trên Ichimoku Cloud")
                elif close < cloud_bottom:
                    signals_count['SELL'] += 1
                    reasons.append("Giá dưới Ichimoku Cloud")
                else:
                    signals_count['HOLD'] += 1
                    reasons.append("Giá trong Ichimoku Cloud")
                
                # Cloud color
                if latest['senkou_span_a'] > latest['senkou_span_b']:
                    signals_count['BUY'] += 1
                    reasons.append("Ichimoku Cloud màu xanh")
                else:
                    signals_count['SELL'] += 1
                    reasons.append("Ichimoku Cloud màu đỏ")
            
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
