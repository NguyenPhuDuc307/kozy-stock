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
                    # Calculate date range với thêm dữ liệu trước đó để tính chỉ báo
                    end_date = datetime.now()
                    # Thêm 200 ngày để có đủ dữ liệu cho các chỉ báo như SMA 200
                    extended_days = days + 200
                    start_date = end_date - timedelta(days=extended_days)
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    
                    # Get extended data for indicator calculation
                    df_extended = data_provider.get_historical_data(selected_symbol, start_str, end_str)
                    
                    if df_extended is None or df_extended.empty:
                        st.error(f"❌ Không thể lấy dữ liệu cho {selected_symbol}")
                        return
                    
                    # Calculate technical indicators on extended data
                    df_with_indicators_extended = indicators.calculate_all(df_extended)
                    
                    if df_with_indicators_extended is None or df_with_indicators_extended.empty:
                        st.error("❌ Lỗi khi tính toán chỉ báo kỹ thuật")
                        return
                    
                    # Cắt về period người dùng chọn (lấy các ngày cuối)
                    df_with_indicators = df_with_indicators_extended.tail(days)
                    df = df_with_indicators[['open', 'high', 'low', 'close', 'volume']].copy()
                    
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
                    
                    # Create separate charts for better visualization
                    
                    # 1. Price Chart with Volume (2 subplots)
                    st.subheader("📈 Biểu đồ giá và khối lượng")
                    
                    from plotly.subplots import make_subplots
                    
                    fig_price = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Giá cổ phiếu', 'Khối lượng giao dịch'),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Candlestick chart
                    fig_price.add_trace(
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
                    
                    # Moving Averages
                    if 'sma_20' in df_with_indicators.columns:
                        fig_price.add_trace(
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
                        fig_price.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['sma_50'],
                                name='SMA 50',
                                showlegend=True,
                                line=dict(color='orange', width=2)
                            ),
                            row=1, col=1
                        )
                    
                    # Bollinger Bands
                    if all(col in df_with_indicators.columns for col in ['bb_upper', 'bb_lower']):
                        fig_price.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['bb_upper'],
                                line=dict(color='rgba(173,204,255,0.8)', width=1),
                                name='BB Upper',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                        
                        fig_price.add_trace(
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
                    
                    # Volume
                    colors = ['#26a69a' if close >= open else '#ef5350' 
                              for close, open in zip(df_with_indicators['close'], df_with_indicators['open'])]
                    
                    fig_price.add_trace(
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
                    
                    # Update layout for price chart
                    fig_price.update_layout(
                        title=f"Giá cổ phiếu {selected_symbol} - {selected_period_name}",
                        xaxis_rangeslider_visible=False,
                        height=600,
                        showlegend=True,
                        legend=dict(
                            orientation="v",  # Vertical legend
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02,  # Position to the right
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.5)",
                            borderwidth=1
                        ),
                        template="plotly_white",
                        margin=dict(r=150)  # Right margin for legend
                    )
                    
                    fig_price.update_yaxes(title_text="Giá (VND)", row=1, col=1)
                    fig_price.update_yaxes(title_text="Khối lượng", row=2, col=1)
                    
                    # Simple date formatting - let Plotly handle it automatically
                    fig_price.update_xaxes(title_text="Thời gian", row=2, col=1)
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # 2. RSI Chart
                    if 'rsi' in df_with_indicators.columns:
                        st.subheader("📊 Chỉ số RSI")
                        
                        fig_rsi = go.Figure()
                        
                        fig_rsi.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['rsi'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple', width=2),
                                showlegend=True
                            )
                        )
                        
                        # Add RSI reference lines
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                         annotation_text="Overbought (70)")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                         annotation_text="Oversold (30)")
                        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                         annotation_text="Neutral (50)")
                        
                        fig_rsi.update_layout(
                            title=f"RSI - {selected_symbol}",
                            height=300,
                            yaxis=dict(title="RSI", range=[0, 100]),
                            xaxis=dict(title="Thời gian"),
                            template="plotly_white",
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            ),
                            margin=dict(r=150)
                        )
                        
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # 3. MACD Chart
                    if all(col in df_with_indicators.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                        st.subheader("📈 Chỉ số MACD")
                        
                        fig_macd = go.Figure()
                        
                        # MACD line
                        fig_macd.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue', width=2),
                                showlegend=True
                            )
                        )
                        
                        # Signal line
                        fig_macd.add_trace(
                            go.Scatter(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd_signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='red', width=2),
                                showlegend=True
                            )
                        )
                        
                        # Histogram
                        fig_macd.add_trace(
                            go.Bar(
                                x=df_with_indicators.index,
                                y=df_with_indicators['macd_histogram'],
                                name='Histogram',
                                marker_color='gray',
                                opacity=0.6,
                                showlegend=True
                            )
                        )
                        
                        fig_macd.update_layout(
                            title=f"MACD - {selected_symbol}",
                            height=300,
                            yaxis=dict(title="MACD"),
                            xaxis=dict(title="Thời gian"),
                            template="plotly_white",
                            showlegend=True,
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            ),
                            margin=dict(r=150)
                        )
                        
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Technical Analysis Summary
                    st.subheader("📊 Bảng phân tích chỉ báo kỹ thuật")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Chỉ báo xu hướng")
                        trend_data = []
                        
                        # Moving Averages - ưu tiên chỉ báo phù hợp với thời gian
                        if 'sma_5' in latest and 'sma_10' in latest and pd.notna(latest['sma_5']) and pd.notna(latest['sma_10']):
                            ma_signal = "Tăng" if latest['sma_5'] > latest['sma_10'] else "Giảm"
                            trend_data.append(["SMA 5/10", f"{latest['sma_5']:,.0f}/{latest['sma_10']:,.0f}", ma_signal])
                        
                        if 'sma_10' in latest and 'sma_20' in latest and pd.notna(latest['sma_10']) and pd.notna(latest['sma_20']):
                            ma_signal = "Tăng" if latest['sma_10'] > latest['sma_20'] else "Giảm"
                            trend_data.append(["SMA 10/20", f"{latest['sma_10']:,.0f}/{latest['sma_20']:,.0f}", ma_signal])
                        
                        # EMA cho thời gian ngắn
                        if 'ema_5' in latest and 'ema_10' in latest and pd.notna(latest['ema_5']) and pd.notna(latest['ema_10']):
                            ema_signal = "Tăng" if latest['ema_5'] > latest['ema_10'] else "Giảm"
                            trend_data.append(["EMA 5/10", f"{latest['ema_5']:,.0f}/{latest['ema_10']:,.0f}", ema_signal])
                        
                        if 'ema_12' in latest and 'ema_20' in latest and pd.notna(latest['ema_12']) and pd.notna(latest['ema_20']):
                            ema_signal = "Tăng" if latest['ema_12'] > latest['ema_20'] else "Giảm"
                            trend_data.append(["EMA 12/20", f"{latest['ema_12']:,.0f}/{latest['ema_20']:,.0f}", ema_signal])
                        
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
