"""
📈 STOCK ANALYSIS PAGE - Trang phân tích cổ phiếu
=================================================

Trang phân tích chi tiết cổ phiếu với các chỉ báo kỹ thuật
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_stock_analysis_page():
    """
    Render trang phân tích cổ phiếu
    """
    st.markdown("# 📈 Phân tích cổ phiếu")
    
    try:
        # Import here to avoid circular imports
        from src.data.data_provider import DataProvider
        from src.analysis.indicators import TechnicalIndicators
        from src.analysis.signals import TradingSignals
        from src.utils.config import ConfigManager
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Simple config class for DataProvider
        class SimpleConfig:
            CACHE_ENABLED = True
            CACHE_DURATION = 300
        
        # Initialize components
        config = ConfigManager()
        data_provider = DataProvider(SimpleConfig())
        indicators = TechnicalIndicators()
        signals = TradingSignals()
        
        # Sidebar controls
        st.sidebar.markdown("## ⚙️ Cài đặt")
        
        # Symbol selection
        symbols = config.get_supported_symbols()
        if not symbols:
            symbols = ['VCB', 'FPT', 'VHM', 'HPG', 'VNM', 'MSN', 'TCB', 'CTG', 'BID']
            
        selected_symbol = st.sidebar.selectbox(
            "📈 Chọn mã cổ phiếu:",
            symbols,
            index=0
        )
        
        # Time period
        period_options = {
            "1 tháng": 30,
            "3 tháng": 90, 
            "6 tháng": 180,
            "1 năm": 365,
            "2 năm": 730
        }
        
        selected_period_name = st.sidebar.selectbox(
            "📅 Thời gian:",
            list(period_options.keys()),
            index=2
        )
        days = period_options[selected_period_name]
        
        # Analysis button
        if st.sidebar.button("📊 Phân tích", type="primary"):
            with st.spinner("Đang phân tích..."):
                try:
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    
                    # Get data
                    df = data_provider.get_historical_data(selected_symbol, start_str, end_str)
                    
                    if df is None or df.empty:
                        st.error(f"❌ Không thể lấy dữ liệu cho {selected_symbol}")
                        return
                    
                    # Calculate technical indicators
                    df_with_indicators = indicators.calculate_all(df)
                    
                    if df_with_indicators is None or df_with_indicators.empty:
                        st.error("❌ Lỗi khi tính toán chỉ báo kỹ thuật")
                        return
                    
                    # Get latest values
                    latest = df_with_indicators.iloc[-1]
                    prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
                    
                    # Price change
                    price_change = latest['close'] - prev['close']
                    price_change_pct = (price_change / prev['close']) * 100
                    
                    # Display basic info
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "💰 Giá đóng cửa", 
                            f"{latest['close']:,.0f} VND",
                            delta=f"{price_change:+,.0f} ({price_change_pct:+.2f}%)"
                        )
                    
                    with col2:
                        st.metric("📊 Khối lượng", f"{latest['volume']:,.0f}")
                    
                    with col3:
                        if 'rsi' in latest:
                            st.metric("📊 RSI", f"{latest['rsi']:.1f}")
                    
                    with col4:
                        if 'macd' in latest and 'macd_signal' in latest:
                            macd_diff = latest['macd'] - latest['macd_signal']
                            st.metric("📊 MACD", f"{latest['macd']:.3f}", delta=f"{macd_diff:+.3f}")
                    
                    # Create advanced chart
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=(
                            f'{selected_symbol} - Biểu đồ nến và chỉ báo kỹ thuật',
                            'Khối lượng giao dịch',
                            'RSI (14)',
                            'MACD'
                        ),
                        row_heights=[0.5, 0.2, 0.15, 0.15]
                    )
                    
                    # 1. Candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=df_with_indicators.index,
                            open=df_with_indicators['open'],
                            high=df_with_indicators['high'],
                            low=df_with_indicators['low'],
                            close=df_with_indicators['close'],
                            name="Giá",
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ),
                        row=1, col=1
                    )
                    
                    # 2. Moving Averages
                    if 'sma_20' in df_with_indicators.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['sma_20'],
                                name='SMA 20',
                                line=dict(color='blue', width=1)
                            ),
                            row=1, col=1
                        )
                    
                    if 'sma_50' in df_with_indicators.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['sma_50'],
                                name='SMA 50',
                                line=dict(color='orange', width=1)
                            ),
                            row=1, col=1
                        )
                    
                    # 3. Bollinger Bands
                    if all(col in df_with_indicators.columns for col in ['bb_upper', 'bb_lower']):
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['bb_upper'],
                                line=dict(color='rgba(173,204,255,0.5)', width=1),
                                name='BB Upper',
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['bb_lower'],
                                line=dict(color='rgba(173,204,255,0.5)', width=1),
                                fill='tonexty',
                                fillcolor='rgba(173,204,255,0.2)',
                                name='Bollinger Bands',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    
                    # 4. Volume
                    colors = ['#26a69a' if close >= open else '#ef5350' 
                              for close, open in zip(df_with_indicators['close'], df_with_indicators['open'])]
                    
                    fig.add_trace(
                        go.Bar(
                            x=df_with_indicators.index,
                            y=df_with_indicators['volume'],
                            marker_color=colors,
                            name="Volume",
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                    
                    # 5. RSI
                    if 'rsi' in df_with_indicators.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['rsi'],
                                name='RSI',
                                line=dict(color='purple', width=2)
                            ),
                            row=3, col=1
                        )
                        
                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.7)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.7)
                        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1, opacity=0.5)
                    
                    # 6. MACD
                    if all(col in df_with_indicators.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd'],
                                name='MACD',
                                line=dict(color='blue', width=2)
                            ),
                            row=4, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd_signal'],
                                name='Signal',
                                line=dict(color='red', width=2)
                            ),
                            row=4, col=1
                        )
                        
                        # MACD Histogram
                        colors_macd = ['green' if val >= 0 else 'red' for val in df_with_indicators['macd_histogram']]
                        fig.add_trace(
                            go.Bar(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd_histogram'],
                                marker_color=colors_macd,
                                name="MACD Histogram",
                                opacity=0.6
                            ),
                            row=4, col=1
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Phân tích kỹ thuật {selected_symbol} - {selected_period_name}",
                        xaxis_rangeslider_visible=False,
                        height=1000,
                        showlegend=True,
                        template="plotly_white"
                    )
                    
                    # Update y-axis
                    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
                    fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
                    if 'rsi' in df_with_indicators.columns:
                        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
                    fig.update_yaxes(title_text="MACD", row=4, col=1)
                    fig.update_xaxes(title_text="Thời gian", row=4, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical Analysis Summary
                    st.subheader("📊 Bảng phân tích chỉ báo kỹ thuật")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Chỉ báo xu hướng")
                        trend_data = []
                        
                        # Moving Averages
                        if 'sma_20' in latest and 'sma_50' in latest:
                            ma_signal = "Tăng" if latest['sma_20'] > latest['sma_50'] else "Giảm"
                            trend_data.append(["SMA 20/50", f"{latest['sma_20']:.0f}/{latest['sma_50']:.0f}", ma_signal])
                        
                        # MACD
                        if 'macd' in latest and 'macd_signal' in latest:
                            macd_signal = "Mua" if latest['macd'] > latest['macd_signal'] else "Bán"
                            trend_data.append(["MACD", f"{latest['macd']:.3f}", macd_signal])
                        
                        if trend_data:
                            trend_df = pd.DataFrame(trend_data, columns=['Chỉ báo', 'Giá trị', 'Tín hiệu'])
                            st.dataframe(trend_df, hide_index=True)
                    
                    with col2:
                        st.markdown("### 📊 Chỉ báo momentum")
                        momentum_data = []
                        
                        # RSI
                        if 'rsi' in latest:
                            if latest['rsi'] > 70:
                                rsi_signal = "Quá mua"
                            elif latest['rsi'] < 30:
                                rsi_signal = "Quá bán"
                            else:
                                rsi_signal = "Trung tính"
                            momentum_data.append(["RSI (14)", f"{latest['rsi']:.1f}", rsi_signal])
                        
                        # Stochastic
                        if 'stoch_k' in latest:
                            if latest['stoch_k'] > 80:
                                stoch_signal = "Quá mua"
                            elif latest['stoch_k'] < 20:
                                stoch_signal = "Quá bán"
                            else:
                                stoch_signal = "Trung tính"
                            momentum_data.append(["Stochastic", f"{latest['stoch_k']:.1f}", stoch_signal])
                        
                        if momentum_data:
                            momentum_df = pd.DataFrame(momentum_data, columns=['Chỉ báo', 'Giá trị', 'Tín hiệu'])
                            st.dataframe(momentum_df, hide_index=True)
                    
                    # Generate trading signal
                    st.subheader("🎯 Tín hiệu giao dịch tổng hợp")
                    signal_result = signals.generate_signal(df_with_indicators)
                    
                    if signal_result:
                        signal_type = signal_result.signal_type
                        confidence = signal_result.confidence
                        reasons = signal_result.reasons
                        
                        if signal_type == 'BUY':
                            st.success(f"🟢 **TÍN HIỆU MUA** - Độ tin cậy: {confidence:.0%}")
                        elif signal_type == 'SELL':
                            st.error(f"🔴 **TÍN HIỆU BÁN** - Độ tin cậy: {confidence:.0%}")
                        else:
                            st.warning(f"🟡 **GIỮ** - Độ tin cậy: {confidence:.0%}")
                        
                        if reasons:
                            st.markdown("**Lý do:**")
                            for reason in reasons:
                                st.markdown(f"• {reason}")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Main page function for st.Page
render_stock_analysis_page()
