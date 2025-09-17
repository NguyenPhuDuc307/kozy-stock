"""
🔍 MARKET SCANNER PAGE - Trang quét thị trường
==============================================

Trang quét thị trường để tìm cơ hội đầu tư
"""

import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_market_scanner_page():
    """
    Render trang Market Scanner
    """
    st.markdown("# 🔍 Quét thị trường")
    
    try:
        # Import required modules
        from datetime import datetime, timedelta
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Import local modules
        from src.analysis.market_scanner import MarketScanner
        from src.utils.portfolio_manager import PortfolioManager
        from src.data.data_provider import DataProvider
        from src.analysis.indicators import TechnicalIndicators
        from src.analysis.signals import TradingSignals
        from src.utils.config import ConfigManager
        from src.utils.unified_config import UnifiedConfig, TimeFrame, UnifiedSignalAnalyzer
        
        # Simple config class for DataProvider
        class SimpleConfig:
            CACHE_ENABLED = True
            CACHE_DURATION = 300
        
        # Initialize components
        config = ConfigManager()
        data_provider = DataProvider(SimpleConfig())
        indicators = TechnicalIndicators()
        signals = TradingSignals()
        portfolio_manager = PortfolioManager()
        scanner = MarketScanner()
        
        # Unified timeframe selector
        selected_timeframe = UnifiedConfig.create_sidebar_timeframe_selector("market_scanner_timeframe")
        timeframe_config = UnifiedConfig.get_timeframe_config(selected_timeframe)
        
        # Advanced settings
        custom_thresholds = UnifiedConfig.create_advanced_settings_expander("market_scanner_advanced")
        
        # Sidebar controls
        st.sidebar.markdown("## 🔍 Tùy chọn quét")
        
        # Get available portfolios
        portfolios = portfolio_manager.get_portfolios()
        
        if portfolios:
            scan_types = {}
            # Add portfolio-based scan types
            for portfolio_name in portfolios.keys():
                scan_types[f"Quét {portfolio_name}"] = portfolio_name
        else:
            # Fallback if no portfolios
            scan_types = {
                "Quét nhanh (Top 10)": "quick",
                "Quét VN30": "vn30", 
                "Quét ngân hàng": "banks",
                "Quét bất động sản": "real_estate"
            }
            st.sidebar.warning("⚠️ Chưa có danh mục nào. Sử dụng quét mặc định.")
            st.sidebar.info("💡 Hãy vào 'Quản lý danh mục' để tạo danh mục!")
        
        selected_scan = st.sidebar.selectbox(
            "📊 Loại quét:",
            list(scan_types.keys())
        )
        
        scan_type = scan_types[selected_scan]
        
        # Technical indicators selection
        st.sidebar.markdown("### 📊 Chỉ báo kỹ thuật")
        
        # Trend indicators
        trend_col1, trend_col2 = st.sidebar.columns(2)
        with trend_col1:
            use_sma = st.checkbox("SMA", value=True)
            use_ema = st.checkbox("EMA", value=True)
        with trend_col2:
            use_bb = st.checkbox("Bollinger", value=True)
            use_ichimoku = st.checkbox("Ichimoku", value=False)
            
        # Momentum indicators    
        mom_col1, mom_col2 = st.sidebar.columns(2)
        with mom_col1:
            use_rsi = st.checkbox("RSI", value=True)
            use_macd = st.checkbox("MACD", value=True)
        with mom_col2:
            use_stoch = st.checkbox("Stochastic", value=True)
            use_wr = st.checkbox("Williams %R", value=True)

        # Filters
        st.sidebar.markdown("### 🎯 Bộ lọc kết quả")
        
        filter_col1, filter_col2, filter_col3 = st.sidebar.columns(3)
        with filter_col1:
            min_score = st.number_input(
                "Điểm tối thiểu",
                min_value=-1.0,
                max_value=1.0,
                value=-0.5,
                step=0.1
            )
        with filter_col2:
            min_volume = st.number_input(
                "Khối lượng tối thiểu",
                min_value=0,
                value=100000,
                step=50000
            )
        with filter_col3:
            min_price = st.number_input(
                "Giá tối thiểu",
                min_value=0,
                value=10,
                step=1000
            )
        
        # Strong signals settings
        st.sidebar.markdown("### ⚡ Tùy chọn tín hiệu mạnh")
        
        strong_col1, strong_col2 = st.sidebar.columns(2)
        with strong_col1:
            conf_threshold = st.slider(
                "Độ tin cậy (%)",
                0, 100, 50, 5,
                help="Ngưỡng độ tin cậy tối thiểu để xem là tín hiệu mạnh"
            ) / 100
        with strong_col2:
            score_threshold = st.slider(
                "Điểm tổng (%)",
                0, 100, 30, 5,
                help="Ngưỡng điểm tổng tối thiểu để xem là tín hiệu mạnh"
            ) / 100
        
        # Scan button
        scan_clicked = st.sidebar.button("🔍 Quét thị trường", type="primary")
        
        # Landing page - hiển thị khi chưa quét
        if not scan_clicked:
            st.markdown("""
            ### 📁 Danh mục có sẵn:
            """)
            
            # Hiển thị danh sách portfolios
            portfolios = portfolio_manager.get_portfolios()
            if portfolios:
                cols = st.columns(3)
                for i, (portfolio_name, stocks) in enumerate(portfolios.items()):
                    with cols[i % 3]:
                        st.info(f"**{portfolio_name}**\n{len(stocks)} cổ phiếu")
            else:
                st.warning("⚠️ Chưa có danh mục nào")
            
            return
        
        # Scan execution
        if scan_clicked:
            with st.spinner("Đang quét thị trường..."):
                try:
                    scanner = MarketScanner()
                    
                    # Get symbols based on selected portfolio
                    if scan_type in portfolios:
                        symbols = portfolios[scan_type]
                    else:
                        # Fallback for old scan types
                        if scan_type == "quick":
                            symbols = ["VCB", "CTG", "BID", "ACB", "VHM", "VIC", "VNM", "HPG", "MSN", "PLX"]
                        elif scan_type == "vn30":
                            symbols = ["VCB", "CTG", "BID", "ACB", "VHM", "VIC", "VNM", "HPG", "MSN", "PLX",
                                     "TCB", "MBB", "TPB", "VPB", "STB", "SSI", "VND", "FPT", "GAS", "POW"]
                        elif scan_type == "banks":
                            symbols = ["VCB", "CTG", "BID", "ACB", "TCB", "MBB", "TPB", "VPB", "STB", "SHB"]
                        elif scan_type == "real_estate":
                            symbols = ["VHM", "VIC", "NVL", "PDR", "DXG", "KDH", "DIG", "CEO", "HDG", "NLG"]
                        else:
                            symbols = ["VCB", "FPT", "VHM"]  # Default fallback
                    
                    # Calculate date range using unified config
                    start_date, end_date = UnifiedConfig.get_date_range(selected_timeframe)
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    
                    # Initialize unified signal analyzer
                    signal_analyzer = UnifiedSignalAnalyzer(selected_timeframe, custom_thresholds)
                    
                    # Collect results for all symbols
                    scan_results = []
                    
                    for symbol in symbols:
                        try:
                            # Get historical data
                            df_extended = data_provider.get_historical_data(symbol, start_str, end_str)
                            
                            if df_extended is not None and not df_extended.empty:
                                # Calculate technical indicators
                                df_with_indicators = indicators.calculate_all(df_extended).tail(timeframe_config.display_days)
                                latest = df_with_indicators.iloc[-1]
                                
                                # Calculate basic metrics
                                prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
                                price_change = latest['close'] - prev['close']
                                price_change_pct = (price_change / prev['close']) * 100
                                
                                # Generate unified trading signal
                                signal_analysis = signal_analyzer.analyze_comprehensive_signal(df_with_indicators)
                                
                                # Create scan result  
                                original_price = float(latest['close'])
                                adjusted_price = original_price * 1000 if original_price < 100 else original_price
                                
                                scan_result = {
                                    'Symbol': symbol,
                                    'Price': adjusted_price,
                                    'Volume': float(latest['volume']),
                                    'Change': float(price_change) * (1000 if original_price < 100 else 1),
                                    'Change_Pct': float(price_change_pct)
                                }
                                
                                # Add technical indicators (always calculate if available)
                                if 'rsi' in latest:
                                    scan_result['RSI'] = float(latest['rsi'])
                                
                                if all(col in latest for col in ['macd', 'macd_signal']):
                                    scan_result['MACD'] = float(latest['macd'])
                                    scan_result['MACD_Signal'] = float(latest['macd_signal'])
                                
                                if 'stoch_k' in latest:
                                    scan_result['Stochastic'] = float(latest['stoch_k'])
                                
                                if 'williams_r' in latest:
                                    scan_result['Williams_R'] = float(latest['williams_r'])
                                
                                # Add trend indicators (always calculate if available)
                                for period in [20, 50]:
                                    col = f'sma_{period}'
                                    if col in latest:
                                        scan_result[f'SMA_{period}'] = float(latest[col])
                                
                                for period in [12, 26]:
                                    col = f'ema_{period}'
                                    if col in latest:
                                        scan_result[f'EMA_{period}'] = float(latest[col])
                                
                                if all(col in latest for col in ['bb_upper', 'bb_lower', 'bb_percent']):
                                    scan_result['BB_Upper'] = float(latest['bb_upper'])
                                    scan_result['BB_Lower'] = float(latest['bb_lower'])
                                    scan_result['BB_Percent'] = float(latest['bb_percent'])
                                
                                # Add unified signal result with proper type conversion
                                if signal_analysis:
                                    scan_result['Signal'] = str(signal_analysis['signal'])
                                    scan_result['Confidence'] = float(signal_analysis['confidence'])
                                    scan_result['Signal_Reasons'] = signal_analysis['reasons']
                                else:
                                    scan_result['Signal'] = 'HOLD'
                                    scan_result['Confidence'] = 0.0
                                    scan_result['Signal_Reasons'] = []
                                
                                scan_results.append(scan_result)
                                
                        except Exception as e:
                            st.warning(f"⚠️ Lỗi khi quét {symbol}: {str(e)}")
                            continue
                    
                    # Convert to DataFrame
                    results = pd.DataFrame(scan_results)
                    
                    # Run market scanner analysis
                    scanner_results = scanner.scan_market(symbols)
                    if scanner_results is not None and not scanner_results.empty:
                        # Merge scanner results with technical analysis
                        results = results.merge(
                            scanner_results[['Symbol', 'Overall_Score', 'Liquidity_Ratio']], 
                            on='Symbol', 
                            how='left'
                        )
                        # Fill NA values with default scores
                        results['Overall_Score'] = results['Overall_Score'].fillna(0)
                        results['Liquidity_Ratio'] = results['Liquidity_Ratio'].fillna(1)
                    else:
                        # If scanner analysis fails, add default scores
                        results['Overall_Score'] = 0
                        results['Liquidity_Ratio'] = 1
                    
                    if results is not None and not results.empty:
                        # Convert columns to numeric and apply filters
                        results['Price'] = pd.to_numeric(results['Price'], errors='coerce')
                        results['Volume'] = pd.to_numeric(results['Volume'], errors='coerce')
                        results['Overall_Score'] = pd.to_numeric(results['Overall_Score'], errors='coerce')
                        
                        # Apply filters with proper type handling
                        price_filter = results['Price'].notna() & (results['Price'] >= min_price if min_price > 0 else True)
                        volume_filter = results['Volume'].notna() & (results['Volume'] >= min_volume if min_volume > 0 else True)
                        score_filter = results['Overall_Score'].notna() & (results['Overall_Score'] >= min_score if min_score > -1 else True)
                        
                        # Combine filters
                        mask = price_filter & volume_filter & score_filter
                        filtered_results = results[mask]
                        
                        if not filtered_results.empty:
                            # Display results with timeframe info
                            st.subheader(f"📊 Kết quả quét ({len(filtered_results)} cổ phiếu)")
                            st.info(f"🕐 **Khung thời gian**: {timeframe_config.name} - {timeframe_config.description}")
                            
                            # Overview metrics
                            buy_signals = len(filtered_results[filtered_results['Signal'] == 'BUY'])
                            sell_signals = len(filtered_results[filtered_results['Signal'] == 'SELL'])
                            hold_signals = len(filtered_results[filtered_results['Signal'] == 'HOLD'])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Tổng cổ phiếu", len(filtered_results))
                            with col2:
                                st.metric("Tín hiệu MUA", buy_signals)
                            with col3:
                                st.metric("Tín hiệu BÁN", sell_signals)
                            with col4:
                                st.metric("Tín hiệu GIỮ", hold_signals)

                            # Create tabs for different views
                            tab1, tab2, tab3, tab4 = st.tabs([
                                "📊 Bảng tín hiệu", 
                                "📈 Chỉ báo kỹ thuật", 
                                "📉 Biểu đồ phân phối",
                                "⚡ Tín hiệu mạnh"
                            ])
                            
                            with tab1:
                                # Basic signals table
                                display_cols = ['Symbol', 'Signal', 'Confidence', 'Price', 'Change_Pct', 'Volume', 'Overall_Score']
                                display_df = filtered_results[display_cols].copy()
                                
                                # Format columns
                                display_df['Price'] = display_df['Price'].map('{:,.0f}'.format)
                                display_df['Change_Pct'] = display_df['Change_Pct'].map('{:+.2f}%'.format)
                                display_df['Volume'] = display_df['Volume'].map('{:,.0f}'.format)
                                display_df['Confidence'] = display_df['Confidence'].map('{:.1%}'.format)
                                display_df['Overall_Score'] = display_df['Overall_Score'].round(2)
                                
                                # Rename columns
                                display_df.columns = ['Mã', 'Tín hiệu', 'Độ tin cậy', 'Giá', '% Thay đổi', 'Khối lượng', 'Điểm']
                                
                                # Sort by score
                                display_df = display_df.sort_values('Điểm', ascending=False)
                                
                                st.dataframe(display_df, hide_index=True, width='stretch', key="signal_results_table")
                            
                            with tab2:
                                # Technical indicators table
                                st.write("**📊 Chỉ báo kỹ thuật**")
                                
                                # Start with basic columns
                                tech_cols = ['Symbol', 'Signal']
                                rename_dict = {'Symbol': 'Mã', 'Signal': 'Tín hiệu'}
                                
                                # Add all available technical indicators based on actual column names
                                available_indicators = [
                                    ('RSI', 'RSI'),
                                    ('MACD', 'MACD'),
                                    ('MACD_Signal', 'MACD Signal'),
                                    ('Stochastic', 'Stoch K'),
                                    ('Williams_R', 'Williams %R'),
                                    ('BB_Percent', 'BB %B'),
                                    ('BB_Upper', 'BB Upper'),
                                    ('BB_Lower', 'BB Lower'),
                                    ('SMA_20', 'SMA 20'),
                                    ('SMA_50', 'SMA 50'),
                                    ('EMA_12', 'EMA 12'),
                                    ('EMA_26', 'EMA 26')
                                ]
                                
                                for col_name, display_name in available_indicators:
                                    if col_name in filtered_results.columns:
                                        tech_cols.append(col_name)
                                        rename_dict[col_name] = display_name
                                
                                # Show which indicators are available
                                found_indicators = [col for col in tech_cols if col not in ['Symbol', 'Signal']]
                                st.write(f"🎯 Tìm thấy {len(found_indicators)} chỉ báo: {', '.join(found_indicators)}")
                                
                                # Only proceed if we have indicators
                                if len(tech_cols) > 2:  # More than Symbol and Signal
                                    tech_df = filtered_results[tech_cols].copy()
                                    
                                    # Format numeric columns
                                    for col in tech_cols:
                                        if col not in ['Symbol', 'Signal'] and col in tech_df.columns:
                                            tech_df[col] = pd.to_numeric(tech_df[col], errors='coerce').round(2)
                                    
                                    # Rename columns
                                    tech_df = tech_df.rename(columns=rename_dict)
                                    
                                    st.dataframe(tech_df, hide_index=True, width='stretch', key="tech_indicators_table")
                                else:
                                    st.warning("❌ Không có chỉ báo kỹ thuật nào được tính toán")
                                    st.info("💡 Vui lòng kiểm tra dữ liệu đầu vào hoặc cài đặt chỉ báo")
                            
                            with tab3:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # RSI Distribution
                                    if use_rsi and 'RSI' in filtered_results.columns:
                                        fig_rsi = go.Figure()
                                        fig_rsi.add_trace(go.Histogram(
                                            x=filtered_results['RSI'],
                                            name='RSI',
                                            nbinsx=20,
                                            marker_color='blue'
                                        ))
                                        fig_rsi.update_layout(
                                            title='Phân bố RSI',
                                            xaxis_title='RSI',
                                            yaxis_title='Số lượng cổ phiếu'
                                        )
                                        st.plotly_chart(fig_rsi, width='stretch', key="rsi_distribution_chart")
                                
                                with col2:
                                    # Price Change Distribution
                                    fig_change = go.Figure()
                                    fig_change.add_trace(go.Histogram(
                                        x=filtered_results['Change_Pct'],
                                        name='% Thay đổi',
                                        nbinsx=20,
                                        marker_color='green'
                                    ))
                                    fig_change.update_layout(
                                        title='Phân bố % thay đổi giá',
                                        xaxis_title='% Thay đổi',
                                        yaxis_title='Số lượng cổ phiếu'
                                    )
                                    st.plotly_chart(fig_change, width='stretch', key="price_change_distribution_chart")
                            
                            with tab4:
                                # Strong signals analysis
                                st.write("**⚡ Phân tích tín hiệu mạnh**")
                                
                                # Convert Overall_Score to numeric and calculate absolute values safely
                                overall_scores = pd.to_numeric(filtered_results['Overall_Score'], errors='coerce')
                                confidence_values = pd.to_numeric(filtered_results['Confidence'], errors='coerce')
                                
                                # Apply filters using values from sidebar
                                confidence_mask = confidence_values >= conf_threshold
                                score_mask = overall_scores.abs() >= score_threshold
                                strong_signals = filtered_results[confidence_mask | score_mask]  # OR instead of AND
                                
                                # Show statistics
                                total_signals = len(filtered_results)
                                strong_count = len(strong_signals)
                                st.success(f"📊 Tìm thấy {strong_count}/{total_signals} tín hiệu mạnh ({strong_count/total_signals*100:.1f}%)")
                                
                                # Show breakdown
                                high_conf_only = len(filtered_results[confidence_values >= conf_threshold])
                                high_score_only = len(filtered_results[overall_scores.abs() >= score_threshold])
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Độ tin cậy cao", f"{high_conf_only}", f"≥{conf_threshold:.0%}")
                                with col2:
                                    st.metric("Điểm tổng cao", f"{high_score_only}", f"≥{score_threshold:.0%}")
                                
                                if not strong_signals.empty:
                                    # Sort by confidence and score
                                    strong_signals_sorted = strong_signals.sort_values(
                                        ['Confidence', 'Overall_Score'], 
                                        ascending=[False, False]
                                    )
                                    
                                    for _, signal in strong_signals_sorted.iterrows():
                                        # Color coding based on signal strength
                                        conf = signal['Confidence']
                                        score = signal['Overall_Score']
                                        
                                        if conf >= 0.8 or abs(score) >= 0.8:
                                            badge = "🔥"  # Very strong
                                        elif conf >= 0.6 or abs(score) >= 0.6:
                                            badge = "⚡"  # Strong
                                        else:
                                            badge = "📊"  # Moderate
                                        
                                        with st.expander(
                                            f"{badge} {signal['Symbol']} - {signal['Signal']} "
                                            f"(Tin cậy: {conf:.0%}, Điểm: {score:.2f})"
                                        ):
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.write("**📈 Thông tin cơ bản:**")
                                                st.write(f"• Giá: {signal['Price']:,.0f} VND")
                                                st.write(f"• Thay đổi: {signal['Change_Pct']:+.2f}%")
                                                st.write(f"• Khối lượng: {signal['Volume']:,.0f}")
                                                
                                                if 'Signal_Reasons' in signal and isinstance(signal['Signal_Reasons'], list):
                                                    st.write("**🎯 Lý do tín hiệu:**")
                                                    for reason in signal['Signal_Reasons']:
                                                        st.markdown(f"• {reason}")
                                            
                                            with col2:
                                                st.write("**📊 Chỉ báo kỹ thuật:**")
                                                tech_info = []
                                                
                                                if 'RSI' in signal and not pd.isna(signal['RSI']):
                                                    rsi_val = signal['RSI']
                                                    if rsi_val < 30:
                                                        rsi_status = "🔵 Quá bán"
                                                    elif rsi_val > 70:
                                                        rsi_status = "🔴 Quá mua"
                                                    else:
                                                        rsi_status = "⚪ Trung tính"
                                                    tech_info.append(f"• RSI: {rsi_val:.1f} {rsi_status}")
                                                
                                                if 'MACD' in signal and not pd.isna(signal['MACD']):
                                                    macd_val = signal['MACD']
                                                    macd_signal_val = signal.get('MACD_Signal', 0)
                                                    if macd_val > macd_signal_val:
                                                        macd_status = "🟢 Tăng"
                                                    else:
                                                        macd_status = "🔴 Giảm"
                                                    tech_info.append(f"• MACD: {macd_val:.3f} {macd_status}")
                                                
                                                if 'Stochastic' in signal and not pd.isna(signal['Stochastic']):
                                                    stoch_val = signal['Stochastic']
                                                    if stoch_val < 20:
                                                        stoch_status = "🔵 Quá bán"
                                                    elif stoch_val > 80:
                                                        stoch_status = "🔴 Quá mua"
                                                    else:
                                                        stoch_status = "⚪ Trung tính"
                                                    tech_info.append(f"• Stochastic: {stoch_val:.1f} {stoch_status}")
                                                
                                                if 'Williams_R' in signal and not pd.isna(signal['Williams_R']):
                                                    wr_val = signal['Williams_R']
                                                    if wr_val < -80:
                                                        wr_status = "🔵 Quá bán"
                                                    elif wr_val > -20:
                                                        wr_status = "🔴 Quá mua"
                                                    else:
                                                        wr_status = "⚪ Trung tính"
                                                    tech_info.append(f"• Williams %R: {wr_val:.1f} {wr_status}")
                                                
                                                if 'BB_Percent' in signal and not pd.isna(signal['BB_Percent']):
                                                    bb_val = signal['BB_Percent']
                                                    if bb_val < 0:
                                                        bb_status = "🔵 Dưới dải"
                                                    elif bb_val > 1:
                                                        bb_status = "🔴 Trên dải"
                                                    else:
                                                        bb_status = "⚪ Trong dải"
                                                    tech_info.append(f"• BB %B: {bb_val:.2f} {bb_status}")
                                                
                                                if tech_info:
                                                    st.write("\n".join(tech_info))
                                                else:
                                                    st.write("Không có chỉ báo kỹ thuật")
                                else:
                                    st.info("Không có tín hiệu mạnh nào được tìm thấy")
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

# Main page function for st.Page
render_market_scanner_page()
