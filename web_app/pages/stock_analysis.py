"""
📈 STOCK ANALYSIS PAGE - Trang phân tích cổ phiếu
=================================================

Trang phân tích chi tiết cổ phiếu với các chỉ báo kỹ thuật
"""

import streamlit as st
import pandas as pd
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
            index=1
        )
        days = period_options[selected_period_name]
        
        # Analysis button
        analyze_clicked = st.sidebar.button("📊 Phân tích", type="primary")
        
        # Hiển thị thông tin cơ bản mặc định
        if not analyze_clicked:
            st.info("💡 **Hướng dẫn:** Chọn mã cổ phiếu và thời gian, sau đó nhấn nút '📊 Phân tích' để xem biểu đồ và chỉ báo kỹ thuật chi tiết.")
            
            # Hiển thị preview với dữ liệu cơ bản
            with st.spinner("Đang tải thông tin cơ bản..."):
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)  # Chỉ lấy 7 ngày cho preview
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    
                    df_preview = data_provider.get_historical_data(selected_symbol, start_str, end_str)
                    
                    if df_preview is not None and not df_preview.empty:
                        latest_preview = df_preview.iloc[-1]
                        prev_preview = df_preview.iloc[-2] if len(df_preview) > 1 else latest_preview
                        
                        price_change = latest_preview['close'] - prev_preview['close']
                        price_change_pct = (price_change / prev_preview['close']) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "💰 Giá đóng cửa", 
                                f"{latest_preview['close']:,.0f} VND",
                                delta=f"{price_change:+,.0f} ({price_change_pct:+.2f}%)"
                            )
                        
                        with col2:
                            st.metric("📊 Khối lượng", f"{latest_preview['volume']:,.0f}")
                        
                        with col3:
                            high_52w = df_preview['high'].max()
                            st.metric("📈 Cao nhất (7 ngày)", f"{high_52w:,.0f} VND")
                        
                        with col4:
                            low_52w = df_preview['low'].min()
                            st.metric("📉 Thấp nhất (7 ngày)", f"{low_52w:,.0f} VND")
                        
                        st.markdown("---")
                        st.markdown("### 📈 Biểu đồ giá 7 ngày gần nhất")
                        
                        # Simple price chart
                        import plotly.graph_objects as go
                        fig_simple = go.Figure()
                        
                        fig_simple.add_trace(
                            go.Candlestick(
                                x=df_preview.index,
                                open=df_preview['open'],
                                high=df_preview['high'],
                                low=df_preview['low'],
                                close=df_preview['close'],
                                name=selected_symbol
                            )
                        )
                        
                        fig_simple.update_layout(
                            title=f"Biểu đồ nến {selected_symbol} - 7 ngày gần nhất",
                            xaxis_rangeslider_visible=False,
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig_simple, use_container_width=True)
                        
                except Exception as e:
                    st.warning("⚠️ Không thể tải thông tin preview")
        
        if analyze_clicked:
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
                            showlegend=True,
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
                                showlegend=True,
                                line=dict(color='blue', width=2)
                            ),
                            row=1, col=1
                        )
                    
                    if 'sma_50' in df_with_indicators.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['sma_50'],
                                name='SMA 50',
                                showlegend=True,
                                line=dict(color='orange', width=2)
                            ),
                            row=1, col=1
                        )
                    
                    # 3. Bollinger Bands
                    if all(col in df_with_indicators.columns for col in ['bb_upper', 'bb_lower']):
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['bb_upper'],
                                line=dict(color='rgba(173,204,255,0.8)', width=1),
                                name='BB Upper',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['bb_lower'],
                                line=dict(color='rgba(173,204,255,0.8)', width=1),
                                fill='tonexty',
                                fillcolor='rgba(173,204,255,0.2)',
                                name='BB Lower',
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
                            showlegend=True,
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
                                showlegend=True,
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
                                showlegend=True,
                                line=dict(color='blue', width=2)
                            ),
                            row=4, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd_signal'],
                                name='Signal',
                                showlegend=True,
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
                                showlegend=True,
                                opacity=0.6
                            ),
                            row=4, col=1
                        )
                    
                    # Update layout with legend at bottom
                    fig.update_layout(
                        title=f"Phân tích kỹ thuật {selected_symbol} - {selected_period_name}",
                        xaxis_rangeslider_visible=False,
                        height=1100,  # Increase height for bottom legend
                        showlegend=True,
                        legend=dict(
                            orientation="h",  # Horizontal legend
                            yanchor="top",
                            y=-0.05,  # Position below the chart
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.5)",
                            borderwidth=1,
                            font=dict(size=12)
                        ),
                        template="plotly_white",
                        margin=dict(r=50, l=50, t=80, b=100)  # Bottom margin for legend
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
                        
                        # Moving Averages - ưu tiên chỉ báo phù hợp với thời gian
                        if 'sma_5' in latest and 'sma_10' in latest and pd.notna(latest['sma_5']) and pd.notna(latest['sma_10']):
                            ma_signal = "Tăng" if latest['sma_5'] > latest['sma_10'] else "Giảm"
                            trend_data.append(["SMA 5/10", f"{latest['sma_5']:.0f}/{latest['sma_10']:.0f}", ma_signal])
                        
                        if 'sma_10' in latest and 'sma_20' in latest and pd.notna(latest['sma_10']) and pd.notna(latest['sma_20']):
                            ma_signal = "Tăng" if latest['sma_10'] > latest['sma_20'] else "Giảm"
                            trend_data.append(["SMA 10/20", f"{latest['sma_10']:.0f}/{latest['sma_20']:.0f}", ma_signal])
                        
                        # EMA cho thời gian ngắn
                        if 'ema_5' in latest and 'ema_10' in latest and pd.notna(latest['ema_5']) and pd.notna(latest['ema_10']):
                            ema_signal = "Tăng" if latest['ema_5'] > latest['ema_10'] else "Giảm"
                            trend_data.append(["EMA 5/10", f"{latest['ema_5']:.0f}/{latest['ema_10']:.0f}", ema_signal])
                        
                        if 'ema_12' in latest and 'ema_20' in latest and pd.notna(latest['ema_12']) and pd.notna(latest['ema_20']):
                            ema_signal = "Tăng" if latest['ema_12'] > latest['ema_20'] else "Giảm"
                            trend_data.append(["EMA 12/20", f"{latest['ema_12']:.0f}/{latest['ema_20']:.0f}", ema_signal])
                        
                        # MACD (nếu có)
                        if 'macd' in latest and 'macd_signal' in latest and pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
                            macd_signal = "Mua" if latest['macd'] > latest['macd_signal'] else "Bán"
                            trend_data.append(["MACD", f"{latest['macd']:.3f}", macd_signal])
                        
                        # Bollinger Bands
                        if 'bb_percent' in latest and pd.notna(latest['bb_percent']):
                            if latest['bb_percent'] > 80:
                                bb_signal = "Quá mua"
                            elif latest['bb_percent'] < 20:
                                bb_signal = "Quá bán"
                            else:
                                bb_signal = "Trung tính"
                            trend_data.append(["Bollinger %B", f"{latest['bb_percent']:.1f}%", bb_signal])
                        
                        if trend_data:
                            trend_df = pd.DataFrame(trend_data, columns=['Chỉ báo', 'Giá trị', 'Tín hiệu'])
                            st.dataframe(trend_df, hide_index=True)
                        else:
                            st.warning("⚠️ Không có chỉ báo xu hướng khả dụng")
                    
                    with col2:
                        st.markdown("### 📊 Chỉ báo momentum")
                        momentum_data = []
                        
                        # RSI
                        if 'rsi' in latest and pd.notna(latest['rsi']):
                            if latest['rsi'] > 70:
                                rsi_signal = "Quá mua"
                            elif latest['rsi'] < 30:
                                rsi_signal = "Quá bán"
                            else:
                                rsi_signal = "Trung tính"
                            momentum_data.append(["RSI (14)", f"{latest['rsi']:.1f}", rsi_signal])
                        
                        # Stochastic
                        if 'stoch_k' in latest and pd.notna(latest['stoch_k']):
                            if latest['stoch_k'] > 80:
                                stoch_signal = "Quá mua"
                            elif latest['stoch_k'] < 20:
                                stoch_signal = "Quá bán"
                            else:
                                stoch_signal = "Trung tính"
                            momentum_data.append(["Stochastic", f"{latest['stoch_k']:.1f}", stoch_signal])
                        
                        # Williams %R
                        if 'williams_r' in latest and pd.notna(latest['williams_r']):
                            if latest['williams_r'] > -20:
                                wr_signal = "Quá mua"
                            elif latest['williams_r'] < -80:
                                wr_signal = "Quá bán"
                            else:
                                wr_signal = "Trung tính"
                            momentum_data.append(["Williams %R", f"{latest['williams_r']:.1f}", wr_signal])
                        
                        # ROC (Rate of Change)
                        if 'roc' in latest and pd.notna(latest['roc']):
                            roc_signal = "Tăng" if latest['roc'] > 0 else "Giảm"
                            momentum_data.append(["ROC", f"{latest['roc']:.2f}%", roc_signal])
                        
                        # ADX (Average Directional Index)
                        if 'adx' in latest and pd.notna(latest['adx']):
                            if latest['adx'] > 25:
                                adx_signal = "xu hướng mạnh"
                            elif latest['adx'] > 20:
                                adx_signal = "xu hướng trung bình"
                            else:
                                adx_signal = "không có xu hướng"
                            momentum_data.append(["ADX", f"{latest['adx']:.1f}", adx_signal])
                        
                        # ATR (Average True Range)
                        if 'atr' in latest and pd.notna(latest['atr']):
                            momentum_data.append(["ATR", f"{latest['atr']:.1f}", "Độ biến động"])
                        
                        if momentum_data:
                            momentum_df = pd.DataFrame(momentum_data, columns=['Chỉ báo', 'Giá trị', 'Tín hiệu'])
                            st.dataframe(momentum_df, hide_index=True)
                        else:
                            st.warning("⚠️ Không có chỉ báo momentum khả dụng")
                    
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
