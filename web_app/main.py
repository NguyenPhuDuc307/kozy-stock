"""
📱 MAIN STREAMLIT APPLICATION - Ứng dụng web phân tích chứng khoán
================================================================

Ứng dụng web chính với navigation để phân tích cổ phiếu và quét thì trường
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# Simple config class for DataProvider
class SimpleConfig:
    CACHE_ENABLED = True
    CACHE_DURATION = 300

# Helper function to get DataProvider with config
def get_data_provider():
    """Get DataProvider instance with config"""
    try:
        from src.data.data_provider import DataProvider
        return DataProvider(SimpleConfig())
    except Exception as e:
        st.error(f"❌ Không thể khởi tạo DataProvider: {e}")
        return None

# Configure Streamlit page
st.set_page_config(
    page_title="Phân tích chứng khoán Việt Nam",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nav-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 1rem;
        text-align: center;
        background-color: #f0f2f6;
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-decoration: none;
        display: block;
    }
    .nav-button:hover {
        background-color: #1f77b4;
        color: white;
    }
    .nav-button.active {
        background-color: #1f77b4;
        color: white;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .feature-card h3 {
        color: white;
        margin-bottom: 1rem;
    }
    .metric-highlight {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Hàm chính của ứng dụng
    """
    # Header
    st.markdown('<h1 class="main-header">📊 Hệ thống phân tích chứng khoán Việt Nam</h1>', 
                unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("## 🧭 Điều hướng")
    
    # Navigation options
    pages = {
        "🏠 Trang chủ": "home",
        "📈 Phân tích cổ phiếu": "stock_analysis", 
        "🔍 Quét thị trường": "market_scanner",
        "📊 So sánh cổ phiếu": "stock_comparison",
        "📋 Backtest chiến lược": "backtest",
        "ℹ️ Hướng dẫn": "help"
    }
    
    selected_page = st.sidebar.radio(
        "Chọn trang:",
        list(pages.keys()),
        index=0
    )
    
    page_key = pages[selected_page]
    
    # Route to different pages
    if page_key == "home":
        render_home_page()
    elif page_key == "stock_analysis":
        render_stock_analysis_page()
    elif page_key == "market_scanner":
        render_market_scanner_page()
    elif page_key == "stock_comparison":
        render_comparison_page()
    elif page_key == "backtest":
        render_backtest_page()
    elif page_key == "help":
        render_help_page()

def render_home_page():
    """
    Render trang chủ
    """
    st.markdown("## 🏠 Chào mừng đến với hệ thống phân tích chứng khoán!")
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Phân tích cổ phiếu</h3>
            <p>Phân tích chi tiết cổ phiếu với các chỉ báo kỹ thuật như RSI, MACD, Bollinger Bands và nhiều hơn nữa.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 So sánh cổ phiếu</h3>
            <p>So sánh hiệu suất của nhiều cổ phiếu cùng lúc để đưa ra quyết định đầu tư tốt nhất.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Quét thị trường</h3>
            <p>Quét toàn bộ thị trường để tìm ra những cổ phiếu có tín hiệu giao dịch tốt nhất.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📋 Backtest chiến lược</h3>
            <p>Kiểm tra hiệu quả của các chiến lược giao dịch trên dữ liệu lịch sử.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("## 📊 Thống kê nhanh")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Cổ phiếu hỗ trợ", "100+", "VN30 + Top stocks")
    
    with col2:
        st.metric("📊 Chỉ báo kỹ thuật", "40+", "RSI, MACD, BB...")
    
    with col3:
        st.metric("🔍 Tín hiệu giao dịch", "6 loại", "Mua/Bán/Giữ")
    
    with col4:
        st.metric("⚡ Tốc độ quét", "10-50 CP/lần", "Phân tích real-time")
    
    # Quick actions
    st.markdown("## 🚀 Hành động nhanh")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Phân tích VCB", type="primary", use_container_width=True):
            st.session_state.quick_analysis = "VCB"
            st.rerun()
    
    with col2:
        if st.button("🔍 Quét VN30", type="primary", use_container_width=True):
            st.session_state.quick_scan = "VN30"
            st.rerun()
    
    with col3:
        if st.button("📊 So sánh Banks", type="primary", use_container_width=True):
            st.session_state.quick_compare = "Banks"
            st.rerun()
    
    # Recent updates
    st.markdown("## 📰 Cập nhật gần đây")
    
    st.info("""
    🆕 **Tính năng mới:**
    - ✅ Market Scanner: Quét toàn thị trường với tín hiệu AI
    - ✅ Phân tích theo ngành: Banking, Real Estate, Technology...
    - ✅ Xếp hạng thanh khoản: Tìm cổ phiếu có volume giao dịch cao
    - ✅ Export Excel: Xuất báo cáo phân tích chi tiết
    """)
    
    st.success("""
    💡 **Mẹo sử dụng:**
    - Sử dụng Market Scanner để tìm cơ hội đầu tư
    - Kết hợp nhiều chỉ báo để ra quyết định chính xác hơn
    - Luôn kiểm tra thanh khoản trước khi giao dịch
    - Sử dụng Backtest để test chiến lược
    """)

def render_stock_analysis_page():
    """
    Render trang phân tích cổ phiếu
    """
    st.markdown("# 📈 Phân tích cổ phiếu")
    
    try:
        # Import here to avoid circular imports
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(project_root)
        
        from src.data.data_provider import DataProvider
        from src.analysis.indicators import TechnicalIndicators
        from src.analysis.signals import TradingSignals
        from src.utils.config import ConfigManager
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Initialize components
        config = ConfigManager()
        data_provider = DataProvider(config)
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
                            rsi_color = "normal"
                            if latest['rsi'] > 70:
                                rsi_color = "inverse"
                            elif latest['rsi'] < 30:
                                rsi_color = "off"
                            st.metric("� RSI", f"{latest['rsi']:.1f}")
                    
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
                    st.subheader("� Bảng phân tích chỉ báo kỹ thuật")
                    
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
                    
                    # Volatility Analysis
                    st.subheader("📊 Phân tích volatility")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if all(col in latest for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                            bb_width = ((latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']) * 100
                            bb_position = ((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])) * 100
                            
                            st.metric("BB Width", f"{bb_width:.1f}%")
                            st.metric("BB Position", f"{bb_position:.1f}%")
                    
                    with col2:
                        if 'atr' in latest:
                            atr_pct = (latest['atr'] / latest['close']) * 100
                            st.metric("ATR", f"{latest['atr']:.0f}")
                            st.metric("ATR %", f"{atr_pct:.1f}%")
                    
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


def render_market_scanner_page():
            
            with col1:
                st.markdown("""
                ### 📈 Chỉ báo xu hướng
                - Moving Averages (SMA, EMA)
                - MACD với histogram
                - ADX và Parabolic SAR
                """)
            
            with col2:
                st.markdown("""
                ### 📊 Chỉ báo momentum
                - RSI (14)
                - Stochastic Oscillator
                - Williams %R
                - Rate of Change
                """)
            
            with col3:
                st.markdown("""
                ### � Chỉ báo volatility
                - Bollinger Bands
                - Average True Range (ATR)
                - Keltner Channels
                """)
    
    # except Exception as e:
    #     st.error(f"❌ Lỗi khởi tạo: {str(e)}")
    #     import traceback
    #     st.code(traceback.format_exc())

def render_market_scanner_page():
    """
    Render trang Market Scanner
    """
    st.markdown("# 🔍 Quét thị trường")
    
    try:
        # Import here to avoid circular imports
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(project_root)
        
        from src.analysis.market_scanner import MarketScanner
        
        # Sidebar controls
        st.sidebar.markdown("## 🔍 Tùy chọn quét")
        
        scan_types = {
            "Quét nhanh (Top 10)": "quick",
            "Quét VN30": "vn30",
            "Quét ngân hàng": "banks",
            "Quét bất động sản": "real_estate"
        }
        
        selected_scan = st.sidebar.selectbox(
            "📊 Loại quét:",
            list(scan_types.keys())
        )
        
        scan_type = scan_types[selected_scan]
        
        # Filters
        min_score = st.sidebar.slider(
            "Điểm tín hiệu tối thiểu:",
            min_value=-1.0,
            max_value=1.0,
            value=-0.5,
            step=0.1
        )
        
        # Scan button
        if st.sidebar.button("🔍 Quét thị trường", type="primary"):
            with st.spinner("Đang quét thị trường..."):
                try:
                    scanner = MarketScanner()
                    
                    # Run scan based on type
                    if scan_type == "quick":
                        symbols = ["VCB", "CTG", "BID", "ACB", "VHM", "VIC", "VNM", "HPG", "MSN", "PLX"]
                    elif scan_type == "vn30":
                        symbols = ["VCB", "CTG", "BID", "ACB", "VHM", "VIC", "VNM", "HPG", "MSN", "PLX",
                                 "TCB", "MBB", "TPB", "VPB", "STB", "SSI", "VND", "FPT", "GAS", "POW"]
                    elif scan_type == "banks":
                        symbols = ["VCB", "CTG", "BID", "ACB", "TCB", "MBB", "TPB", "VPB", "STB", "SHB"]
                    elif scan_type == "real_estate":
                        symbols = ["VHM", "VIC", "NVL", "PDR", "DXG", "KDH", "DIG", "CEO", "HDG", "NLG"]
                    
                    results = scanner.scan_market(symbols)
                    
                    if results is not None and not results.empty:
                        # Filter results by score
                        filtered_results = results[results['Overall_Score'] >= min_score]
                        
                        if not filtered_results.empty:
                            # Display results
                            st.subheader(f"📊 Kết quả quét ({len(filtered_results)} cổ phiếu)")
                            
                            # Overview metrics
                            buy_signals = len(filtered_results[filtered_results['Overall_Signal'] == 'BUY'])
                            sell_signals = len(filtered_results[filtered_results['Overall_Signal'] == 'SELL'])
                            hold_signals = len(filtered_results[filtered_results['Overall_Signal'] == 'HOLD'])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Tổng cổ phiếu", len(filtered_results))
                            with col2:
                                st.metric("Tín hiệu MUA", buy_signals)
                            with col3:
                                st.metric("Tín hiệu BÁN", sell_signals)
                            with col4:
                                st.metric("Tín hiệu GIỮ", hold_signals)
                            
                            # Results table
                            st.subheader("📋 Chi tiết kết quả")
                            
                            # Format display
                            if not filtered_results.empty:
                                display_df = filtered_results[['Symbol', 'Overall_Signal', 'Overall_Score', 'Liquidity_Ratio']].copy()
                                display_df['Overall_Score'] = display_df['Overall_Score'].round(2)
                                display_df['Liquidity_Ratio'] = display_df['Liquidity_Ratio'].round(2)
                                
                                # Sort by score
                                display_df = display_df.sort_values('Overall_Score', ascending=False)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Top picks
                                st.subheader("🏆 Top picks")
                                
                                if buy_signals > 0:
                                    buy_stocks = filtered_results[filtered_results['Overall_Signal'] == 'BUY'].nlargest(3, 'Overall_Score')
                                    st.success(f"**Tín hiệu MUA:** {', '.join(buy_stocks['Symbol'].tolist())}")
                                
                                if sell_signals > 0:
                                    sell_stocks = filtered_results[filtered_results['Overall_Signal'] == 'SELL'].nsmallest(3, 'Overall_Score')
                                    st.error(f"**Tín hiệu BÁN:** {', '.join(sell_stocks['Symbol'].tolist())}")
                        else:
                            st.warning("❌ Không có cổ phiếu nào thỏa mãn bộ lọc")
                    else:
                        st.error("❌ Không thể quét thị trường")
                        
                except Exception as e:
                    st.error(f"❌ Lỗi quét thị trường: {str(e)}")
        
    except ImportError as e:
        st.error("❌ Không thể load Market Scanner")
        st.info("💡 Vui lòng kiểm tra cài đặt Market Scanner")
        st.code(f"Import error: {e}")
        
        # Fallback simple interface
        st.markdown("## 🔧 Giao diện đơn giản")
        scan_type = st.selectbox("Chọn loại quét:", ["Top 10", "VN30", "Banks"])
        if st.button("Quét"):
            st.info(f"Đang quét {scan_type}...")

def render_comparison_page():
    """
    Render trang so sánh cổ phiếu
    """
    st.markdown("# 📊 So sánh cổ phiếu")
    
    # Basic comparison interface
    st.markdown("## Chọn cổ phiếu để so sánh")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock1 = st.selectbox("Cổ phiếu 1:", ["VCB", "CTG", "BID", "ACB", "VIC", "FPT", "MSN", "VNM", "PLX", "TCB"], index=0, key="stock1")
    
    with col2:
        stock2 = st.selectbox("Cổ phiếu 2:", ["VCB", "CTG", "BID", "ACB", "VIC", "FPT", "MSN", "VNM", "PLX", "TCB"], index=1, key="stock2")
    
    # Thời gian so sánh
    st.markdown("## ⏰ Khoảng thời gian")
    time_period = st.selectbox(
        "Chọn khoảng thời gian:",
        ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm"],
        index=2
    )
    
    # Map time period to days
    period_map = {
        "1 tháng": 30,
        "3 tháng": 90, 
        "6 tháng": 180,
        "1 năm": 365,
        "2 năm": 730
    }
    days = period_map[time_period]
    
    if st.button("📊 So sánh", type="primary"):
        if stock1 == stock2:
            st.error("❌ Vui lòng chọn 2 cổ phiếu khác nhau!")
            return
            
        with st.spinner("🔄 Đang tải dữ liệu và phân tích..."):
            try:
                # Import cần thiết
                import sys
                import os
                from datetime import datetime, timedelta
                import pandas as pd
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.append(project_root)
                
                from src.data.data_provider import DataProvider
                from src.utils.config import ConfigManager
                
                # Get data for both stocks
                config = ConfigManager()
                data_provider = DataProvider(config)
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Fetch data for stock 1
                data1 = data_provider.get_historical_data(
                    symbol=stock1,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    resolution='1D'
                )
                
                # Fetch data for stock 2
                data2 = data_provider.get_historical_data(
                    symbol=stock2,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    resolution='1D'
                )
                
                if data1 is None or data2 is None or data1.empty or data2.empty:
                    st.error("❌ Không thể lấy dữ liệu cho một hoặc cả hai cổ phiếu!")
                    return
                
                # Normalize prices to percentage change
                data1_norm = (data1['close'] / data1['close'].iloc[0] - 1) * 100
                data2_norm = (data2['close'] / data2['close'].iloc[0] - 1) * 100
                
                # Tính toán metrics so sánh
                returns1 = data1['close'].pct_change().dropna()
                returns2 = data2['close'].pct_change().dropna()
                
                # Performance metrics
                total_return1 = (data1['close'].iloc[-1] / data1['close'].iloc[0] - 1) * 100
                total_return2 = (data2['close'].iloc[-1] / data2['close'].iloc[0] - 1) * 100
                
                volatility1 = returns1.std() * (252**0.5) * 100  # Annualized
                volatility2 = returns2.std() * (252**0.5) * 100
                
                # Sharpe ratio (giả sử risk-free rate = 5%)
                sharpe1 = (returns1.mean() * 252 - 0.05) / (returns1.std() * (252**0.5)) if returns1.std() > 0 else 0
                sharpe2 = (returns2.mean() * 252 - 0.05) / (returns2.std() * (252**0.5)) if returns2.std() > 0 else 0
                
                # Max drawdown
                cumulative1 = (1 + returns1).cumprod()
                running_max1 = cumulative1.cummax()
                drawdown1 = (cumulative1 - running_max1) / running_max1
                max_drawdown1 = drawdown1.min() * 100
                
                cumulative2 = (1 + returns2).cumprod()
                running_max2 = cumulative2.cummax()
                drawdown2 = (cumulative2 - running_max2) / running_max2
                max_drawdown2 = drawdown2.min() * 100
                
                # Correlation
                correlation = returns1.corr(returns2)
                
                # Display results
                st.success("✅ Phân tích hoàn thành!")
                
                # Performance comparison chart
                st.markdown("## 📈 Biểu đồ so sánh hiệu suất")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data1.index,
                    y=data1_norm,
                    mode='lines',
                    name=f'{stock1}',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=data2.index,
                    y=data2_norm,
                    mode='lines',
                    name=f'{stock2}',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f'So sánh hiệu suất: {stock1} vs {stock2} ({time_period})',
                    xaxis_title='Thời gian',
                    yaxis_title='Thay đổi giá (%)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics table
                st.markdown("## 📊 Bảng so sánh chỉ số")
                
                metrics_df = pd.DataFrame({
                    'Chỉ số': [
                        'Tổng lợi nhuận (%)',
                        'Volatility hàng năm (%)', 
                        'Sharpe Ratio',
                        'Max Drawdown (%)',
                        'Giá hiện tại (VND)',
                        'Giá cao nhất (VND)',
                        'Giá thấp nhất (VND)'
                    ],
                    stock1: [
                        f"{total_return1:.2f}%",
                        f"{volatility1:.2f}%",
                        f"{sharpe1:.2f}",
                        f"{max_drawdown1:.2f}%",
                        f"{data1['close'].iloc[-1]:,.0f}",
                        f"{data1['high'].max():,.0f}",
                        f"{data1['low'].min():,.0f}"
                    ],
                    stock2: [
                        f"{total_return2:.2f}%",
                        f"{volatility2:.2f}%",
                        f"{sharpe2:.2f}",
                        f"{max_drawdown2:.2f}%",
                        f"{data2['close'].iloc[-1]:,.0f}",
                        f"{data2['high'].max():,.0f}",
                        f"{data2['low'].min():,.0f}"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Correlation analysis
                st.markdown("## 🔗 Phân tích tương quan")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Hệ số tương quan", f"{correlation:.3f}")
                
                with col2:
                    if correlation > 0.7:
                        correlation_desc = "Tương quan cao"
                        color = "🔴"
                    elif correlation > 0.3:
                        correlation_desc = "Tương quan trung bình"
                        color = "🟡"
                    else:
                        correlation_desc = "Tương quan thấp"
                        color = "🟢"
                    st.metric("Mức độ", f"{color} {correlation_desc}")
                
                with col3:
                    diversification = "Tốt" if correlation < 0.5 else "Kém"
                    st.metric("Hiệu quả đa dạng hóa", diversification)
                
                # Winner analysis
                st.markdown("## 🏆 Kết luận")
                
                if total_return1 > total_return2:
                    winner = stock1
                    winner_return = total_return1
                    loser_return = total_return2
                else:
                    winner = stock2
                    winner_return = total_return2
                    loser_return = total_return1
                
                st.success(f"🏆 **{winner}** có hiệu suất tốt hơn với lợi nhuận {winner_return:.2f}% so với {loser_return:.2f}%")
                
                # Risk-adjusted analysis
                if sharpe1 > sharpe2:
                    risk_winner = stock1
                    risk_winner_sharpe = sharpe1
                else:
                    risk_winner = stock2
                    risk_winner_sharpe = sharpe2
                
                st.info(f"💎 **{risk_winner}** có Sharpe Ratio tốt hơn ({risk_winner_sharpe:.2f}) - hiệu suất điều chỉnh theo rủi ro")
                
                # Volume comparison chart
                st.markdown("## 📊 So sánh khối lượng giao dịch")
                
                fig_volume = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f'Khối lượng {stock1}', f'Khối lượng {stock2}'),
                    vertical_spacing=0.1
                )
                
                fig_volume.add_trace(
                    go.Bar(x=data1.index, y=data1['volume'], name=f'{stock1} Volume', marker_color='blue'),
                    row=1, col=1
                )
                
                fig_volume.add_trace(
                    go.Bar(x=data2.index, y=data2['volume'], name=f'{stock2} Volume', marker_color='red'),
                    row=2, col=1
                )
                
                fig_volume.update_layout(height=400, showlegend=False)
                fig_volume.update_xaxes(title_text="Thời gian", row=2, col=1)
                fig_volume.update_yaxes(title_text="Khối lượng", row=1, col=1)
                fig_volume.update_yaxes(title_text="Khối lượng", row=2, col=1)
                
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Trading recommendations
                st.markdown("## 💡 Khuyến nghị")
                
                if correlation < 0.3:
                    st.success("✅ **Phù hợp để đa dạng hóa** - Hai cổ phiếu có tương quan thấp, giúp giảm rủi ro danh mục")
                elif correlation > 0.7:
                    st.warning("⚠️ **Không phù hợp để đa dạng hóa** - Hai cổ phiếu có tương quan cao, rủi ro tương tự nhau")
                else:
                    st.info("ℹ️ **Đa dạng hóa trung bình** - Có thể kết hợp nhưng cần cân nhắc tỷ trọng")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {str(e)}")
                # Fallback simple comparison
                st.info("🔄 Hiển thị giao diện đơn giản...")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 📈 {stock1}")
                    st.info("Dữ liệu sẽ được hiển thị ở đây")
                
                with col2:
                    st.markdown(f"### 📈 {stock2}")
                    st.info("Dữ liệu sẽ được hiển thị ở đây")

def render_backtest_page():
    """
    Render trang backtest
    """
    st.markdown("# 📋 Backtest chiến lược")
    
    try:
        # Import here to avoid circular imports
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(project_root)
        
        from src.backtesting.backtesting_engine import BacktestingEngine, Portfolio
        from src.backtesting.strategies import (
            MovingAverageCrossoverStrategy, 
            MeanReversionStrategy, 
            MomentumStrategy, 
            BollingerBandsStrategy
        )
        
        # Sidebar controls
        st.sidebar.markdown("## ⚙️ Cấu hình Backtest")
        
        # Strategy selection
        strategy_type = st.sidebar.selectbox(
            "🎯 Chọn chiến lược:",
            ["Moving Average Crossover", "Mean Reversion", "Momentum", "Bollinger Bands"]
        )
        
        # Stock selection
        stock_symbol = st.sidebar.selectbox(
            "🎯 Chọn cổ phiếu:",
            ["VCB", "CTG", "BID", "ACB", "VHM", "VIC", "VNM", "HPG", "MSN", "PLX"]
        )
        
        # Time period
        from datetime import datetime, timedelta
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            months_back = st.selectbox("Thời gian:", [3, 6, 12, 24], index=2)
        
        # Portfolio settings
        initial_capital = st.sidebar.number_input(
            "Vốn ban đầu (VND):",
            min_value=10_000_000,
            max_value=1_000_000_000,
            value=100_000_000,
            step=10_000_000
        )
        
        # Strategy parameters
        st.sidebar.markdown("### ⚙️ Tham số chiến lược")

        if strategy_type == "Moving Average Crossover":
            fast_period = st.sidebar.slider("MA nhanh:", 5, 20, 10)
            slow_period = st.sidebar.slider("MA chậm:", 20, 50, 30)
            strategy_params = {'fast_period': fast_period, 'slow_period': slow_period}
        elif strategy_type == "Mean Reversion":
            lookback = st.sidebar.slider("Chu kỳ lookback:", 10, 30, 20)
            threshold = st.sidebar.slider("Ngưỡng:", 1.5, 3.0, 2.0, 0.1)
            strategy_params = {'lookback_period': lookback, 'entry_threshold': threshold}
        elif strategy_type == "Momentum":
            momentum_period = st.sidebar.slider("Chu kỳ momentum:", 10, 30, 20)
            threshold = st.sidebar.slider("Ngưỡng:", 0.02, 0.1, 0.05, 0.01)
            strategy_params = {'momentum_period': momentum_period, 'entry_threshold': threshold}
        else:  # Bollinger Bands
            period = st.sidebar.slider("Chu kỳ BB:", 15, 25, 20)
            std_mult = st.sidebar.slider("Hệ số std:", 1.5, 2.5, 2.0, 0.1)
            strategy_params = {'period': period, 'std_multiplier': std_mult}
        
        # Run backtest button
        if st.sidebar.button("🚀 Chạy Backtest", type="primary"):
            with st.spinner("Đang chạy backtest..."):
                try:
                    # Get data with config
                    data_provider = get_data_provider()
                    if data_provider is None:
                        st.error("❌ Không thể khởi tạo DataProvider!")
                        return
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=months_back*30)
                    
                    data = data_provider.get_historical_data(
                        symbol=stock_symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        resolution='1D'
                    )
                    
                    if data is None or data.empty:
                        st.error("❌ Không thể lấy dữ liệu!")
                        return
                    
                    # Create strategy
                    if strategy_type == "Moving Average Crossover":
                        strategy = MovingAverageCrossoverStrategy(**strategy_params)
                    elif strategy_type == "Mean Reversion":
                        strategy = MeanReversionStrategy(**strategy_params)
                    elif strategy_type == "Momentum":
                        strategy = MomentumStrategy(**strategy_params)
                    else:
                        strategy = BollingerBandsStrategy(**strategy_params)
                    
                    # Create portfolio
                    portfolio = Portfolio(
                        initial_capital=initial_capital,
                        position_size_method='fixed_percentage',
                        position_size_value=0.2
                    )
                    
                    # Run backtest
                    engine = BacktestingEngine(initial_capital=initial_capital)
                    results = engine.backtest(strategy, data)
                    
                    # Display results
                    st.success("✅ Backtest hoàn thành!")
                    
                    # Performance metrics
                    st.subheader("📊 Kết quả tổng quan")
                    
                    metrics = results['performance_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_return = metrics.get('total_return', 0.0)
                        st.metric("Tổng lợi nhuận", f"{total_return:.1f}%")
                    
                    with col2:
                        annual_return = metrics.get('annual_return', 0.0)
                        st.metric("Lợi nhuận năm", f"{annual_return:.1f}%")
                    
                    with col3:
                        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    with col4:
                        max_drawdown = metrics.get('max_drawdown', 0.0)
                        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                    
                    # Trade details
                    if 'trades' in results and results['trades']:
                        st.subheader("📋 Chi tiết giao dịch")
                        
                        import pandas as pd
                        trades_df = pd.DataFrame(results['trades'])
                        
                        st.write(f"**Tổng số lệnh:** {len(trades_df)}")
                        
                        if len(trades_df) > 0:
                            winning_trades = len(trades_df[trades_df['pnl'] > 0])
                            win_rate = winning_trades / len(trades_df) * 100
                            st.write(f"**Tỷ lệ thắng:** {win_rate:.1f}%")
                            
                            # Show recent trades
                            st.dataframe(trades_df.tail(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi backtest: {str(e)}")
        
    except ImportError as e:
        st.error("❌ Không thể load Backtesting engine")
        st.info("💡 Vui lòng kiểm tra cài đặt Backtesting module")
        st.code(f"Import error: {e}")
        
        # Fallback simple interface
        st.markdown("## 🔧 Giao diện đơn giản")
        strategy = st.selectbox("Chọn chiến lược:", ["MA Crossover", "Mean Reversion"])
        symbol = st.selectbox("Chọn cổ phiếu:", ["VCB", "CTG", "BID"])
        if st.button("Chạy Backtest"):
            st.info(f"Đang backtest {strategy} cho {symbol}...")

def render_help_page():
    """
    Render trang hướng dẫn
    """
    st.markdown("# ℹ️ Hướng dẫn sử dụng")
    
    # Help content
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Tổng quan", "📈 Phân tích", "🔍 Market Scanner", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ## 🏠 Tổng quan hệ thống
        
        Hệ thống phân tích chứng khoán Việt Nam cung cấp các công cụ:
        
        ### 📈 Phân tích cổ phiếu
        - Biểu đồ nến tương tác với Plotly
        - Hơn 40 chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands...)
        - Tín hiệu mua/bán tự động
        - Thông tin công ty chi tiết
        
        ### 🔍 Market Scanner
        - Quét toàn thị trường real-time
        - Xếp hạng cổ phiếu theo tín hiệu
        - Phân tích thanh khoản
        - Export báo cáo Excel
        
        ### 📊 Tính năng khác
        - So sánh nhiều cổ phiếu
        - Backtest chiến lược
        - Phân tích theo ngành
        """)
    
    with tab2:
        st.markdown("""
        ## 📈 Hướng dẫn phân tích cổ phiếu
        
        ### Bước 1: Chọn cổ phiếu
        - Sử dụng sidebar để chọn mã cổ phiếu
        - Chọn khoảng thời gian phân tích (1m - 2y)
        
        ### Bước 2: Cấu hình chỉ báo
        - Bật/tắt các chỉ báo muốn hiển thị
        - RSI: Đo momentum (30-70)
        - MACD: Tín hiệu xu hướng
        - Bollinger Bands: Đo volatility
        
        ### Bước 3: Đọc tín hiệu
        - 🟢 MUA: Khi nhiều chỉ báo tích cực
        - 🔴 BÁN: Khi nhiều chỉ báo tiêu cực  
        - 🟡 GIỮ: Tín hiệu chưa rõ ràng
        
        ### Bước 4: Kiểm tra thanh khoản
        - Volume giao dịch cao = Dễ mua/bán
        - Thanh khoản thấp = Rủi ro thanh khoản
        """)
    
    with tab3:
        st.markdown("""
        ## 🔍 Hướng dẫn Market Scanner
        
        ### Chọn nhóm cổ phiếu
        - **VN30**: 30 cổ phiếu lớn nhất
        - **Top Banks**: Ngân hàng hàng đầu
        - **Real Estate**: Bất động sản
        - **Technology**: Công nghệ
        - **Custom**: Tự nhập mã
        
        ### Hiểu kết quả quét
        - **Overall Score**: Điểm tổng hợp (-1 đến 1)
        - **Liquidity Ratio**: Tỷ lệ thanh khoản so với TB
        - **Signal Strength**: Độ mạnh tín hiệu
        
        ### Sử dụng bộ lọc
        - Lọc theo tín hiệu: MUA/BÁN/GIỮ
        - Lọc theo thanh khoản tối thiểu
        - Sắp xếp theo điểm số hoặc thanh khoản
        
        ### Export dữ liệu
        - Tải báo cáo Excel đầy đủ
        - Copy CSV cho xử lý thêm
        """)
    
    with tab4:
        st.markdown("""
        ## ❓ Câu hỏi thường gặp
        
        ### Q: Dữ liệu cập nhật khi nào?
        A: Dữ liệu được lấy real-time từ vnstock API khi bạn thực hiện phân tích.
        
        ### Q: Tín hiệu có chính xác 100%?
        A: Không có tín hiệu nào chính xác 100%. Luôn kết hợp nhiều yếu tố và quản lý rủi ro.
        
        ### Q: Market Scanner quét bao nhiêu cổ phiếu?
        A: Có thể quét từ 10-50 cổ phiếu tùy nhóm được chọn. VN30 có 30 cổ phiếu.
        
        ### Q: Làm sao export được dữ liệu?
        A: Sử dụng nút "Tải Excel" trong Market Scanner hoặc copy CSV.
        
        ### Q: Hệ thống có hỗ trợ cổ phiếu nào?
        A: Hỗ trợ tất cả cổ phiếu trên HOSE, HNX có trong vnstock.
        
        ### Q: Cần internet để sử dụng?
        A: Có, cần internet để lấy dữ liệu real-time từ thị trường.
        """)
    
    # Contact info
    st.markdown("""
    ---
    ### 📞 Liên hệ hỗ trợ
    
    - 📧 Email: support@stockanalysis.vn (demo)
    - 💬 Telegram: @vnstock_support (demo)
    - 📱 Hotline: 1900-xxx-xxx (demo)
    
    **⚠️ Lưu ý quan trọng:**
    - Hệ thống chỉ mang tính chất tham khảo
    - Không phải lời khuyên đầu tư
    - Luôn tự nghiên cứu trước khi đầu tư
    - Quản lý rủi ro là ưu tiên hàng đầu
    """)

if __name__ == "__main__":
    main()
