"""
📊 INTERACTIVE CHARTS - Biểu đồ tương tác
=======================================

Module này tạo các biểu đồ tương tác cho phân tích kỹ thuật
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class InteractiveCharts:
    """
    Lớp tạo biểu đồ tương tác
    """
    
    def __init__(self):
        """
        Khởi tạo InteractiveCharts
        """
        self.logger = logging.getLogger(__name__)
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("⚠️ Plotly không có sẵn. Vui lòng cài đặt: pip install plotly")
        else:
            self.logger.info("📊 InteractiveCharts đã được khởi tạo với Plotly")
    
    def create_technical_chart(self, df: pd.DataFrame, 
                               symbol: str,
                               indicators: List[str] = None,
                               height: int = 1000) -> Any:
        """
        Tạo biểu đồ nến với chỉ báo kỹ thuật
        
        Args:
            df: DataFrame chứa dữ liệu OHLCV
            symbol: Mã cổ phiếu
            indicators: Danh sách chỉ báo cần hiển thị
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            self.logger.error("❌ Plotly không có sẵn")
            return None
        
        if len(df) == 0:
            self.logger.error("❌ Không có dữ liệu để vẽ biểu đồ")
            return None
        
        # Tạo subplots với spacing tốt hơn
        has_indicators = any(ind in df.columns for ind in ['rsi', 'macd', 'stoch_k', 'williams_r'])
        
        if has_indicators:
            rows = 5  # Main chart, Volume, RSI, Williams/Stoch, MACD
            row_heights = [0.35, 0.15, 0.15, 0.15, 0.2]
            subplot_titles = (f'{symbol} - Biểu đồ nến', 'Khối lượng', 'RSI', 'Williams %R / Stochastic', 'MACD')
        else:
            rows = 2  # Main chart, Volume only
            row_heights = [0.7, 0.3]
            subplot_titles = (f'{symbol} - Biểu đồ nến', 'Khối lượng')
        
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,  # Tăng spacing từ 0.05 lên 0.08
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['time'] if 'time' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Giá',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # Add moving averages
        ma_indicators = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']
        colors = ['blue', 'orange', 'red', 'purple', 'brown']
        
        for i, ma in enumerate(ma_indicators):
            if ma in df.columns and not df[ma].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['time'] if 'time' in df.columns else df.index,
                    y=df[ma],
                    name=ma.upper(),
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.8
                ), row=1, col=1)
        
        # Add Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            # Upper band
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(173,204,255,0.5)', width=1),
                fill=None
            ), row=1, col=1)
            
            # Lower band
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(173,204,255,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(173,204,255,0.1)'
            ), row=1, col=1)
            
            # Middle band
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['bb_middle'],
                name='BB Middle',
                line=dict(color='rgba(173,204,255,0.8)', width=1, dash='dash')
            ), row=1, col=1)
        
        # Volume chart
        colors_volume = ['red' if close < open else 'green' 
                        for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['time'] if 'time' in df.columns else df.index,
            y=df['volume'],
            name='Khối lượng',
            marker_color=colors_volume,
            opacity=0.7
        ), row=2, col=1)
        
        # Add volume moving average
        if 'volume_sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['volume_sma_20'],
                name='Volume MA20',
                line=dict(color='orange', width=2)
            ), row=2, col=1)
        
        # Add indicators to separate subplots nếu có đủ subplot
        if has_indicators:
                        # Row 3: RSI
            if 'rsi' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2),
                    yaxis='y3'
                ), row=3, col=1)
                
                # RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
            
            # Row 4: Williams %R và Stochastic
            if 'williams_r' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['williams_r'],
                    mode='lines',
                    name='Williams %R',
                    line=dict(color='orange', width=2),
                    yaxis='y4'
                ), row=4, col=1)
            
            # Stochastic oscillator
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['stoch_k'],
                    mode='lines',
                    name='Stoch %K',
                    line=dict(color='cyan', width=2),
                    yaxis='y4'
                ), row=4, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['stoch_d'],
                    mode='lines',
                    name='Stoch %D',
                    line=dict(color='magenta', width=2),
                    yaxis='y4'
                ), row=4, col=1)
                
                # Stochastic reference lines
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
            
            # Row 5: MACD
            if 'macd' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2),
                    yaxis='y5'
                ), row=5, col=1)
                
                if 'macd_signal' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red', width=2),
                        yaxis='y5'
                    ), row=5, col=1)
                
                if 'macd_histogram' in df.columns:
                    fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['macd_histogram'],
                        name='MACD Histogram',
                        marker_color='green',
                        yaxis='y5'
                    ), row=5, col=1)        # Add user-specified indicators if provided
        if indicators:
            available_rows = [3, 4] if has_indicators else [3]
            for i, indicator in enumerate(indicators):
                if indicator in df.columns and not df[indicator].isna().all():
                    row_idx = available_rows[i % len(available_rows)]
                    fig.add_trace(go.Scatter(
                        x=df['time'] if 'time' in df.columns else df.index,
                        y=df[indicator],
                        name=indicator.upper(),
                        line=dict(width=2)
                    ), row=row_idx, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Phân tích kỹ thuật',
            xaxis_title='Thời gian',
            height=height,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Remove x-axis labels for all except bottom subplot
        for i in range(1, rows):
            fig.update_xaxes(showticklabels=False, row=i, col=1)
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
        fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
        
        if has_indicators:
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="Williams %R / Stoch", row=4, col=1)
            fig.update_yaxes(title_text="MACD", row=5, col=1)
        else:
            fig.update_yaxes(title_text="Chỉ báo", row=3, col=1)
        
        return fig
    
    def create_indicator_chart(self, df: pd.DataFrame, indicator: str,
                             symbol: str, height: int = 400) -> Any:
        """
        Tạo biểu đồ cho chỉ báo cụ thể
        
        Args:
            df: DataFrame chứa dữ liệu
            indicator: Tên chỉ báo
            symbol: Mã cổ phiếu
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        if indicator not in df.columns:
            self.logger.error(f"❌ Chỉ báo {indicator} không tồn tại")
            return None
        
        fig = go.Figure()
        
        # Main indicator line
        fig.add_trace(go.Scatter(
            x=df['time'] if 'time' in df.columns else df.index,
            y=df[indicator],
            name=indicator.upper(),
            line=dict(width=3)
        ))
        
        # Add specific levels for different indicators
        if indicator == 'rsi':
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (30)")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                         annotation_text="Midline (50)")
        
        elif indicator == 'stoch_k' and 'stoch_d' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['stoch_d'],
                name='Stoch %D',
                line=dict(width=2, dash='dash')
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="red")
            fig.add_hline(y=20, line_dash="dash", line_color="green")
        
        elif indicator == 'macd' and 'macd_signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['macd_signal'],
                name='MACD Signal',
                line=dict(width=2, dash='dash')
            ))
            
            if 'macd_histogram' in df.columns:
                colors = ['red' if val < 0 else 'green' for val in df['macd_histogram']]
                fig.add_trace(go.Bar(
                    x=df['time'] if 'time' in df.columns else df.index,
                    y=df['macd_histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ))
        
        fig.update_layout(
            title=f'{symbol} - {indicator.upper()}',
            xaxis_title='Thời gian',
            yaxis_title=indicator.upper(),
            height=height,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_comparison_chart(self, data_dict: Dict[str, pd.DataFrame],
                              height: int = 600) -> Any:
        """
        Tạo biểu đồ so sánh nhiều cổ phiếu
        
        Args:
            data_dict: Dict chứa data của các cổ phiếu {symbol: df}
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (symbol, df) in enumerate(data_dict.items()):
            if len(df) == 0:
                continue
            
            # Normalize prices to percentage change
            normalized_prices = (df['close'] / df['close'].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=normalized_prices,
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='So sánh hiệu suất cổ phiếu (%)',
            xaxis_title='Thời gian',
            yaxis_title='Thay đổi (%)',
            height=height,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_volume_analysis_chart(self, df: pd.DataFrame, symbol: str,
                                   height: int = 500) -> Any:
        """
        Tạo biểu đồ phân tích khối lượng
        
        Args:
            df: DataFrame chứa dữ liệu
            symbol: Mã cổ phiếu
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} - Giá', 'Phân tích khối lượng'),
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=df['time'] if 'time' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Giá'
        ), row=1, col=1)
        
        # Volume with color coding
        colors_volume = ['red' if close < open else 'green' 
                        for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(go.Bar(
            x=df['time'] if 'time' in df.columns else df.index,
            y=df['volume'],
            name='Khối lượng',
            marker_color=colors_volume,
            opacity=0.7
        ), row=2, col=1)
        
        # Volume moving average
        if 'volume_sma_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=df['volume_sma_20'],
                name='Volume MA20',
                line=dict(color='orange', width=2)
            ), row=2, col=1)
        
        # OBV if available
        if 'obv' in df.columns:
            # Normalize OBV for display
            obv_normalized = (df['obv'] - df['obv'].min()) / (df['obv'].max() - df['obv'].min()) * df['volume'].max()
            
            fig.add_trace(go.Scatter(
                x=df['time'] if 'time' in df.columns else df.index,
                y=obv_normalized,
                name='OBV (normalized)',
                line=dict(color='purple', width=2),
                yaxis='y3'
            ), row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Phân tích khối lượng',
            height=height,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Giá", row=1, col=1)
        fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
        
        return fig
    
    def create_signal_chart(self, df: pd.DataFrame, symbol: str,
                          height: int = 700) -> Any:
        """
        Tạo biểu đồ với tín hiệu giao dịch
        
        Args:
            df: DataFrame chứa dữ liệu và tín hiệu
            symbol: Mã cổ phiếu
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['time'] if 'time' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Giá'
        ))
        
        # Buy signals
        buy_signals = df[df.get('signal_type', '') == 'BUY']
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals['time'] if 'time' in buy_signals.columns else buy_signals.index,
                y=buy_signals['low'] * 0.98,  # Slightly below the low
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    color='green',
                    size=15
                ),
                name='Tín hiệu MUA',
                text='MUA',
                textposition='bottom center'
            ))
        
        # Sell signals
        sell_signals = df[df.get('signal_type', '') == 'SELL']
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals['time'] if 'time' in sell_signals.columns else sell_signals.index,
                y=sell_signals['high'] * 1.02,  # Slightly above the high
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    color='red',
                    size=15
                ),
                name='Tín hiệu BÁN',
                text='BÁN',
                textposition='top center'
            ))
        
        # Strong buy signals
        strong_buy_signals = df[df.get('signal_type', '') == 'STRONG_BUY']
        if len(strong_buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=strong_buy_signals['time'] if 'time' in strong_buy_signals.columns else strong_buy_signals.index,
                y=strong_buy_signals['low'] * 0.96,
                mode='markers',
                marker=dict(
                    symbol='star',
                    color='darkgreen',
                    size=20
                ),
                name='Tín hiệu MUA MẠNH',
                text='MUA MẠNH',
                textposition='bottom center'
            ))
        
        # Strong sell signals
        strong_sell_signals = df[df.get('signal_type', '') == 'STRONG_SELL']
        if len(strong_sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=strong_sell_signals['time'] if 'time' in strong_sell_signals.columns else strong_sell_signals.index,
                y=strong_sell_signals['high'] * 1.04,
                mode='markers',
                marker=dict(
                    symbol='star',
                    color='darkred',
                    size=20
                ),
                name='Tín hiệu BÁN MẠNH',
                text='BÁN MẠNH',
                textposition='top center'
            ))
        
        fig.update_layout(
            title=f'{symbol} - Tín hiệu giao dịch',
            xaxis_title='Thời gian',
            yaxis_title='Giá',
            height=height,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def save_chart_html(self, fig: Any, filename: str) -> bool:
        """
        Lưu biểu đồ thành file HTML
        
        Args:
            fig: Plotly figure object
            filename: Tên file để lưu
            
        Returns:
            True nếu lưu thành công
        """
        if not PLOTLY_AVAILABLE or fig is None:
            return False
        
        try:
            fig.write_html(filename)
            self.logger.info(f"✅ Đã lưu biểu đồ: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi lưu biểu đồ: {e}")
            return False
    
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str, 
                                indicators: List[str] = None, height: int = 800) -> Any:
        """
        Tạo biểu đồ nến (alias cho create_technical_chart)
        
        Args:
            df: DataFrame chứa dữ liệu OHLCV
            symbol: Mã cổ phiếu
            indicators: Danh sách chỉ báo cần hiển thị
            height: Chiều cao biểu đồ
            
        Returns:
            Plotly figure object
        """
        return self.create_technical_chart(df, symbol, indicators, height)
    
    def get_chart_config(self) -> Dict:
        """
        Lấy cấu hình biểu đồ mặc định
        
        Returns:
            Dict cấu hình
        """
        return {
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'stock_chart',
                'height': 800,
                'width': 1200,
                'scale': 1
            }
        }

# Test module
if __name__ == "__main__":
    """
    Test InteractiveCharts
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    if PLOTLY_AVAILABLE:
        import numpy as np
        
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
        
        # Add indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = 50 + np.random.randn(100) * 20
        df['signal_type'] = 'HOLD'
        df.loc[df.index[::20], 'signal_type'] = 'BUY'
        df.loc[df.index[10::20], 'signal_type'] = 'SELL'
        
        print("🧪 Testing InteractiveCharts...")
        
        # Test charts
        charts = InteractiveCharts()
        
        # Test candlestick chart
        fig1 = charts.create_candlestick_chart(df, 'TEST', ['rsi'])
        if fig1:
            print("✅ Candlestick chart created")
        
        # Test indicator chart
        fig2 = charts.create_indicator_chart(df, 'rsi', 'TEST')
        if fig2:
            print("✅ Indicator chart created")
        
        # Test signal chart
        fig3 = charts.create_signal_chart(df, 'TEST')
        if fig3:
            print("✅ Signal chart created")
        
        print("✅ Test completed!")
    else:
        print("❌ Plotly không có sẵn. Vui lòng cài đặt: pip install plotly")
