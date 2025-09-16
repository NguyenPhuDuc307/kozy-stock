"""
Unified Configuration System for Stock Analysis

Hệ thống cấu hình thống nhất cho phân tích cổ phiếu
- Quản lý khoảng thời gian phân tích
- Thống nhất logic tín hiệu và chỉ số kỹ thuật
- Cấu hình chung cho tất cả các trang
"""

from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


class TimeFrame(Enum):
    """Enum cho các khoảng thời gian phân tích"""
    SHORT_TERM = "short_term"      # Ngắn hạn
    MEDIUM_TERM = "medium_term"    # Trung hạn  
    LONG_TERM = "long_term"        # Dài hạn


@dataclass
class TimeFrameConfig:
    """Cấu hình khoảng thời gian"""
    name: str                      # Tên hiển thị
    display_days: int              # Số ngày hiển thị
    analysis_days: int             # Số ngày dữ liệu để phân tích (bao gồm buffer cho indicators)
    short_ma: int                  # Moving Average ngắn hạn
    long_ma: int                   # Moving Average dài hạn
    rsi_period: int               # Chu kỳ RSI
    macd_fast: int                # MACD fast
    macd_slow: int                # MACD slow
    macd_signal: int              # MACD signal
    description: str              # Mô tả


@dataclass 
class SignalConfig:
    """Cấu hình tín hiệu giao dịch"""
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_signal_threshold: float = 0.0
    bb_overbought: float = 0.8    # Bollinger Bands %B > 80%
    bb_oversold: float = 0.2      # Bollinger Bands %B < 20%
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    williams_overbought: float = -20.0
    williams_oversold: float = -80.0
    volume_spike_threshold: float = 1.5  # 150% của volume trung bình
    min_confidence_threshold: float = 0.3  # Ngưỡng tin cậy tối thiểu


class UnifiedConfig:
    """Lớp cấu hình thống nhất cho toàn bộ hệ thống"""
    
    # Định nghĩa các khoảng thời gian
    TIME_FRAMES = {
        TimeFrame.SHORT_TERM: TimeFrameConfig(
            name="Ngắn hạn (1-2 tuần)",
            display_days=14,
            analysis_days=60,  # Cần 60 ngày để tính các indicator
            short_ma=5,
            long_ma=10,
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            description="Phù hợp cho swing trading, day trading"
        ),
        TimeFrame.MEDIUM_TERM: TimeFrameConfig(
            name="Trung hạn (1-3 tháng)",
            display_days=90,
            analysis_days=200,  # Cần 200 ngày để tính các indicator dài hạn
            short_ma=20,
            long_ma=50,
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            description="Phù hợp cho đầu tư trung hạn"
        ),
        TimeFrame.LONG_TERM: TimeFrameConfig(
            name="Dài hạn (6 tháng - 1 năm)",
            display_days=365,
            analysis_days=500,  # Cần 500 ngày để tính SMA 200, etc.
            short_ma=50,
            long_ma=200,
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            description="Phù hợp cho đầu tư dài hạn, phân tích xu hướng"
        )
    }
    
    # Cấu hình tín hiệu chung
    SIGNAL_CONFIG = SignalConfig()
    
    @classmethod
    def get_timeframe_config(cls, timeframe: TimeFrame) -> TimeFrameConfig:
        """Lấy cấu hình theo khoảng thời gian"""
        return cls.TIME_FRAMES[timeframe]
    
    @classmethod
    def get_timeframe_options(cls) -> Dict[str, TimeFrame]:
        """Lấy các lựa chọn khoảng thời gian cho UI"""
        return {config.name: timeframe for timeframe, config in cls.TIME_FRAMES.items()}
    
    @classmethod
    def get_date_range(cls, timeframe: TimeFrame) -> Tuple[datetime, datetime]:
        """Tính toán khoảng thời gian từ ngày hiện tại"""
        config = cls.get_timeframe_config(timeframe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.analysis_days)
        return start_date, end_date
    
    @classmethod
    def get_display_date_range(cls, timeframe: TimeFrame) -> Tuple[datetime, datetime]:
        """Tính toán khoảng thời gian hiển thị"""
        config = cls.get_timeframe_config(timeframe)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.display_days)
        return start_date, end_date
    
    @classmethod
    def create_sidebar_timeframe_selector(cls, key: str = "timeframe_selector") -> TimeFrame:
        """Tạo selector cho khoảng thời gian trong sidebar"""
        import streamlit as st
        
        st.sidebar.markdown("### ⏱️ Khoảng thời gian phân tích")
        
        options = cls.get_timeframe_options()
        selected_name = st.sidebar.selectbox(
            "Chọn khoảng thời gian:",
            list(options.keys()),
            index=1,  # Mặc định là trung hạn
            key=key,
            help="Chọn khoảng thời gian phân tích phù hợp với chiến lược đầu tư"
        )
        
        selected_timeframe = options[selected_name]
        config = cls.get_timeframe_config(selected_timeframe)
        
        # Hiển thị thông tin cấu hình
        with st.sidebar.expander("ℹ️ Thông tin cấu hình", expanded=False):
            st.write(f"**Mô tả**: {config.description}")
            st.write(f"**Hiển thị**: {config.display_days} ngày")
            st.write(f"**Dữ liệu phân tích**: {config.analysis_days} ngày")
            st.write(f"**MA ngắn hạn**: {config.short_ma} ngày")
            st.write(f"**MA dài hạn**: {config.long_ma} ngày")
            st.write(f"**RSI**: {config.rsi_period} ngày")
            st.write(f"**MACD**: {config.macd_fast}/{config.macd_slow}/{config.macd_signal}")
        
        return selected_timeframe
    
    @classmethod
    def get_unified_indicators_config(cls, timeframe: TimeFrame) -> Dict:
        """Lấy cấu hình indicators theo timeframe"""
        config = cls.get_timeframe_config(timeframe)
        
        return {
            'sma_periods': [config.short_ma, config.long_ma],
            'ema_periods': [config.short_ma, config.long_ma],
            'rsi_period': config.rsi_period,
            'macd_params': {
                'fast': config.macd_fast,
                'slow': config.macd_slow,
                'signal': config.macd_signal
            },
            'bb_period': config.long_ma,  # Bollinger Bands dùng MA dài hạn
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'williams_period': 14,
            'adx_period': 14,
            'atr_period': 14
        }
    
    @classmethod
    def get_signal_thresholds(cls) -> Dict:
        """Lấy ngưỡng tín hiệu"""
        return {
            'rsi_overbought': cls.SIGNAL_CONFIG.rsi_overbought,
            'rsi_oversold': cls.SIGNAL_CONFIG.rsi_oversold,
            'macd_signal_threshold': cls.SIGNAL_CONFIG.macd_signal_threshold,
            'bb_overbought': cls.SIGNAL_CONFIG.bb_overbought,
            'bb_oversold': cls.SIGNAL_CONFIG.bb_oversold,
            'stoch_overbought': cls.SIGNAL_CONFIG.stoch_overbought,
            'stoch_oversold': cls.SIGNAL_CONFIG.stoch_oversold,
            'williams_overbought': cls.SIGNAL_CONFIG.williams_overbought,
            'williams_oversold': cls.SIGNAL_CONFIG.williams_oversold,
            'volume_spike_threshold': cls.SIGNAL_CONFIG.volume_spike_threshold,
            'min_confidence_threshold': cls.SIGNAL_CONFIG.min_confidence_threshold
        }
    
    @classmethod
    def create_advanced_settings_expander(cls, key: str = "advanced_settings") -> Dict:
        """Tạo expander cho cài đặt nâng cao"""
        import streamlit as st
        
        with st.sidebar.expander("⚙️ Cài đặt nâng cao", expanded=False):
            st.markdown("#### 🎯 Ngưỡng tín hiệu")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rsi_overbought = st.slider(
                    "RSI Quá mua", 
                    60, 90, 
                    int(cls.SIGNAL_CONFIG.rsi_overbought),
                    key=f"{key}_rsi_ob"
                )
                
                bb_overbought = st.slider(
                    "BB %B Quá mua",
                    0.6, 1.0,
                    cls.SIGNAL_CONFIG.bb_overbought,
                    0.05,
                    key=f"{key}_bb_ob"
                )
                
                stoch_overbought = st.slider(
                    "Stoch Quá mua",
                    70, 90,
                    int(cls.SIGNAL_CONFIG.stoch_overbought),
                    key=f"{key}_stoch_ob"
                )
            
            with col2:
                rsi_oversold = st.slider(
                    "RSI Quá bán",
                    10, 40,
                    int(cls.SIGNAL_CONFIG.rsi_oversold),
                    key=f"{key}_rsi_os"
                )
                
                bb_oversold = st.slider(
                    "BB %B Quá bán",
                    0.0, 0.4,
                    cls.SIGNAL_CONFIG.bb_oversold,
                    0.05,
                    key=f"{key}_bb_os"
                )
                
                stoch_oversold = st.slider(
                    "Stoch Quá bán",
                    10, 30,
                    int(cls.SIGNAL_CONFIG.stoch_oversold),
                    key=f"{key}_stoch_os"
                )
            
            min_confidence = st.slider(
                "Ngưỡng tin cậy tối thiểu",
                0.0, 0.8,
                cls.SIGNAL_CONFIG.min_confidence_threshold,
                0.05,
                key=f"{key}_min_conf",
                help="Chỉ hiển thị tín hiệu có độ tin cậy >= ngưỡng này"
            )
            
            volume_spike = st.slider(
                "Ngưỡng tăng đột biến khối lượng",
                1.0, 3.0,
                cls.SIGNAL_CONFIG.volume_spike_threshold,
                0.1,
                key=f"{key}_vol_spike",
                help="Khối lượng >= X lần khối lượng trung bình"
            )
            
            return {
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'bb_overbought': bb_overbought,
                'bb_oversold': bb_oversold,
                'stoch_overbought': stoch_overbought,
                'stoch_oversold': stoch_oversold,
                'min_confidence_threshold': min_confidence,
                'volume_spike_threshold': volume_spike
            }


class UnifiedSignalAnalyzer:
    """Lớp phân tích tín hiệu thống nhất"""
    
    def __init__(self, timeframe: TimeFrame, custom_thresholds: Optional[Dict] = None):
        self.timeframe = timeframe
        self.config = UnifiedConfig.get_timeframe_config(timeframe)
        self.thresholds = UnifiedConfig.get_signal_thresholds()
        
        # Override với custom thresholds nếu có
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def analyze_comprehensive_signal(self, df_with_indicators) -> Dict:
        """
        Thuật toán confidence nâng cao - thống nhất cho tất cả các trang
        Sử dụng phân tích đa yếu tố để tính confidence chính xác
        """
        if df_with_indicators is None or df_with_indicators.empty:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Không có dữ liệu'],
                'details': {}
            }
        
        # Nếu có đủ dữ liệu (>= 30 bars), dùng phân tích đầy đủ
        if len(df_with_indicators) >= 30:
            return self._analyze_full_technical_signal(df_with_indicators)
        # Nếu ít dữ liệu, dùng thuật toán advanced fallback
        else:
            return self._analyze_advanced_fallback_signal(df_with_indicators)
    
    def _analyze_full_technical_signal(self, df_with_indicators) -> Dict:
        """
        Phân tích kỹ thuật đầy đủ với thuật toán confidence advanced
        """
        prices = df_with_indicators['close'].values
        volumes = df_with_indicators['volume'].values if 'volume' in df_with_indicators.columns else None
        current_price = prices[-1]
        details = {}
        
        # Tính multiple MA periods
        ma5 = prices[-5:].mean()
        ma10 = prices[-10:].mean()
        ma20 = prices[-20:].mean()
        
        details['ma5'] = ma5
        details['ma10'] = ma10
        details['ma20'] = ma20
        
        # Tính volatility
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(30, len(prices)))]
        volatility = sum([abs(change) for change in price_changes]) / len(price_changes)
        details['volatility'] = volatility
        
        # Tính trend strength với slope regression
        x = list(range(20))
        y = prices[-20:]
        n = len(x)
        trend_slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        trend_slope_percent = trend_slope / current_price
        details['trend_slope'] = trend_slope_percent
        
        # Volume confirmation
        volume_factor = 1.0
        if volumes is not None:
            recent_avg_volume = volumes[-10:].mean()
            total_avg_volume = volumes[-30:].mean()
            if recent_avg_volume > total_avg_volume * 1.2:
                volume_factor = 1.3
            elif recent_avg_volume < total_avg_volume * 0.8:
                volume_factor = 0.7
        details['volume_factor'] = volume_factor
        
        # MA alignment score (độ chính xác cao)
        ma_alignment = 0
        if current_price > ma5 > ma10 > ma20:
            ma_alignment = 1.0  # Perfect uptrend
        elif current_price > ma5 > ma10:
            ma_alignment = 0.7
        elif current_price > ma5:
            ma_alignment = 0.4
        elif current_price < ma5 < ma10 < ma20:
            ma_alignment = -1.0  # Perfect downtrend
        elif current_price < ma5 < ma10:
            ma_alignment = -0.7
        elif current_price < ma5:
            ma_alignment = -0.4
        else:
            ma_alignment = 0
        details['ma_alignment'] = ma_alignment
        
        # Base signal strength
        price_vs_ma5 = (current_price - ma5) / ma5
        base_strength = min(abs(price_vs_ma5) * 12, 0.9)  # More sensitive
        details['base_strength'] = base_strength
        details['price_vs_ma5'] = price_vs_ma5
        
        # Combine factors
        confidence_factors = [
            base_strength,  # Price vs MA5
            abs(ma_alignment) * 0.35,  # MA alignment
            min(abs(trend_slope_percent) * 100, 0.25),  # Trend strength
            (volume_factor - 1) * 0.15  # Volume confirmation
        ]
        
        technical_score = sum(confidence_factors)
        
        # Volatility penalty (high volatility = less reliable)
        volatility_penalty = min(volatility * 8, 0.4)  # Adaptive penalty
        technical_score = max(0, technical_score - volatility_penalty)
        
        details['confidence_factors'] = confidence_factors
        details['volatility_penalty'] = volatility_penalty
        
        # Determine signal and confidence
        reasons = []
        if price_vs_ma5 > 0.015:  # 1.5% above MA5
            signal = "BUY"
            confidence = min(technical_score, 1.0)  # Cap at 100%
            reasons.append(f'Giá cao hơn MA5 {price_vs_ma5:.1%}')
            if ma_alignment > 0.6:
                reasons.append('Xu hướng tăng mạnh')
            if volume_factor > 1.2:
                reasons.append('Volume xác nhận tích cực')
            if trend_slope_percent > 0.001:
                reasons.append('Momentum tăng')
        elif price_vs_ma5 < -0.015:  # 1.5% below MA5
            signal = "SELL"
            confidence = min(technical_score, 1.0)  # Cap at 100%
            reasons.append(f'Giá thấp hơn MA5 {abs(price_vs_ma5):.1%}')
            if ma_alignment < -0.6:
                reasons.append('Xu hướng giảm mạnh')
            if volume_factor > 1.2:
                reasons.append('Volume xác nhận tiêu cực')
            if trend_slope_percent < -0.001:
                reasons.append('Momentum giảm')
        else:
            signal = "HOLD"
            # Cho HOLD cũng có confidence dựa trên technical_score nhưng giảm đi
            confidence = min(technical_score * 0.5, 1.0)  # HOLD có confidence thấp hơn, cap at 100%
            reasons.append('Giá dao động quanh MA5')
            
            # Thêm lý do cụ thể cho HOLD
            if abs(ma_alignment) > 0.3:
                if ma_alignment > 0:
                    reasons.append('Có dấu hiệu tăng nhẹ')
                else:
                    reasons.append('Có dấu hiệu giảm nhẹ')
            
            if abs(trend_slope_percent) > 0.0005:
                if trend_slope_percent > 0:
                    reasons.append('Momentum tăng yếu')
                else:
                    reasons.append('Momentum giảm yếu')
        
        if volatility > 0.06:  # 6% daily volatility
            reasons.append(f'Cảnh báo: Biến động cao ({volatility:.1%})')
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'details': details
        }
    
    def _analyze_advanced_fallback_signal(self, df_with_indicators) -> Dict:
        """
        Thuật toán advanced cho trường hợp ít dữ liệu (< 30 bars)
        """
        if len(df_with_indicators) < 5:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Không đủ dữ liệu cho phân tích'],
                'details': {}
            }
        
        prices = df_with_indicators['close'].values
        volumes = df_with_indicators['volume'].values if 'volume' in df_with_indicators.columns else None
        current_price = prices[-1]
        details = {}
        
        # Tính multiple MA periods
        ma5 = prices[-5:].mean()
        ma10 = prices[-10:].mean() if len(prices) >= 10 else ma5
        ma20 = prices[-20:].mean() if len(prices) >= 20 else ma10
        
        details['ma5'] = ma5
        details['ma10'] = ma10
        details['ma20'] = ma20
        
        # Tính volatility với dữ liệu có hạn
        if len(prices) >= 10:
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = sum([abs(change) for change in price_changes]) / len(price_changes)
        else:
            volatility = 0.02  # Default 2%
        details['volatility'] = volatility
        
        # Tính trend strength
        if len(prices) >= 5:
            trend_slope = (prices[-1] - prices[-5]) / prices[-5]
        else:
            trend_slope = 0
        details['trend_slope'] = trend_slope
        
        # Volume confirmation (nếu có)
        volume_factor = 1.0
        if volumes is not None and len(volumes) >= 5:
            recent_avg_volume = volumes[-5:].mean()
            total_avg_volume = volumes.mean()
            if recent_avg_volume > total_avg_volume * 1.2:
                volume_factor = 1.2
            elif recent_avg_volume < total_avg_volume * 0.8:
                volume_factor = 0.8
        details['volume_factor'] = volume_factor
        
        # MA alignment score
        ma_alignment = 0
        if current_price > ma5 > ma10 > ma20:
            ma_alignment = 1.0
        elif current_price > ma5 > ma10:
            ma_alignment = 0.7
        elif current_price > ma5:
            ma_alignment = 0.3
        elif current_price < ma5 < ma10 < ma20:
            ma_alignment = -1.0
        elif current_price < ma5 < ma10:
            ma_alignment = -0.7
        elif current_price < ma5:
            ma_alignment = -0.3
        else:
            ma_alignment = 0
        details['ma_alignment'] = ma_alignment
        
        # Base signal strength
        price_vs_ma5 = (current_price - ma5) / ma5
        base_strength = min(abs(price_vs_ma5) * 10, 0.8)
        details['base_strength'] = base_strength
        details['price_vs_ma5'] = price_vs_ma5
        
        # Combine factors
        confidence_factors = [
            base_strength,
            abs(ma_alignment) * 0.3,
            min(abs(trend_slope) * 15, 0.2),
            (volume_factor - 1) * 0.1
        ]
        
        technical_score = sum(confidence_factors)
        
        # Volatility penalty
        volatility_penalty = min(volatility * 5, 0.3)
        technical_score = max(0, technical_score - volatility_penalty)
        
        details['confidence_factors'] = confidence_factors
        details['volatility_penalty'] = volatility_penalty
        
        # Determine signal
        reasons = []
        if price_vs_ma5 > 0.02:  # 2% above MA5
            signal = "BUY"
            confidence = min(technical_score, 1.0)  # Cap at 100%
            reasons.append(f'Giá cao hơn MA5 {price_vs_ma5:.1%}')
            if ma_alignment > 0.5:
                reasons.append('Xu hướng tăng')
            if volume_factor > 1.1:
                reasons.append('Volume xác nhận')
        elif price_vs_ma5 < -0.02:  # 2% below MA5
            signal = "SELL"
            confidence = min(technical_score, 1.0)  # Cap at 100%
            reasons.append(f'Giá thấp hơn MA5 {abs(price_vs_ma5):.1%}')
            if ma_alignment < -0.5:
                reasons.append('Xu hướng giảm')
            if volume_factor > 1.1:
                reasons.append('Volume xác nhận')
        else:
            signal = "HOLD"
            # HOLD cũng có confidence dựa trên technical_score
            confidence = min(technical_score * 0.6, 1.0)  # Giảm confidence cho HOLD, cap at 100%
            reasons.append('Giá dao động quanh MA5')
            
            # Thêm thông tin chi tiết cho HOLD
            if abs(ma_alignment) > 0.2:
                if ma_alignment > 0:
                    reasons.append('Có xu hướng tăng nhẹ')
                else:
                    reasons.append('Có xu hướng giảm nhẹ')
            
            if abs(trend_slope) > 0.01:
                if trend_slope > 0:
                    reasons.append('Momentum tăng yếu')
                else:
                    reasons.append('Momentum giảm yếu')
        
        if volatility > 0.05:  # 5% daily volatility
            reasons.append(f'Cảnh báo: Biến động cao ({volatility:.1%})')
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'details': details
        }


# Import pandas ở đây để tránh lỗi
import pandas as pd