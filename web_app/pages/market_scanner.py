"""
🔍 MARKET SCANNER PAGE - Trang quét thị trường
==============================================

Trang quét thị trường để tìm cơ hội đầu tư
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_market_scanner_page():
    """
    Render trang Market Scanner
    """
    st.markdown("# 🔍 Quét thị trường")
    
    try:
        # Import here to avoid circular imports
        from src.analysis.market_scanner import MarketScanner
        from src.utils.portfolio_manager import PortfolioManager
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager()
        
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
        
        # Filters
        min_score = st.sidebar.slider(
            "Điểm tín hiệu tối thiểu:",
            min_value=-1.0,
            max_value=1.0,
            value=-0.5,
            step=0.1
        )
        
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

# Main page function for st.Page
render_market_scanner_page()
