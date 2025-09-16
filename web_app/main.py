"""
🏠 KOZY STOCK - Vietnamese Stock Analysis Dashboard
==================================================

Modern Streamlit application using st.navigation and st.Page architecture
for Vietnamese stock market analysis.
"""

import streamlit as st
import sys
import os

# Add project root to path  
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# Configure page
st.set_page_config(
    page_title="Kozy Stock - Phân tích cổ phiếu Việt Nam",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Main application using modern Streamlit navigation
    """
    
    # Define page functions
    def stock_analysis_page():
        """Stock analysis page function for st.Page"""
        exec(open("/Users/ducnp/Projects/vnstock/kozy-stock/web_app/pages/stock_analysis.py").read())
    
    def market_scanner_page():
        """Market scanner page function for st.Page"""
        exec(open("/Users/ducnp/Projects/vnstock/kozy-stock/web_app/pages/market_scanner.py").read())
    
    def stock_comparison_page():
        """Stock comparison page function for st.Page"""
        exec(open("/Users/ducnp/Projects/vnstock/kozy-stock/web_app/pages/stock_comparison.py").read())
    
    def backtest_page():
        """Backtest page function for st.Page"""
        exec(open("/Users/ducnp/Projects/vnstock/kozy-stock/web_app/pages/backtest.py").read())
    
    def help_page():
        """Help page function for st.Page"""
        exec(open("/Users/ducnp/Projects/vnstock/kozy-stock/web_app/pages/help.py").read())
    
    # Create pages using st.Page
    pages = {
        "Phân tích": [
            st.Page(stock_analysis_page, title="📈 Phân tích kỹ thuật", icon="📈"),
            st.Page(market_scanner_page, title="🔍 Quét thị trường", icon="🔍"),
            st.Page(stock_comparison_page, title="📊 So sánh cổ phiếu", icon="📊"),
        ],
        "Trading": [
            st.Page(backtest_page, title="🔄 Backtest chiến lược", icon="🔄"),
        ],
        "Hỗ trợ": [
            st.Page(help_page, title="❓ Hướng dẫn", icon="❓"),
        ]
    }
    
    # Create navigation
    pg = st.navigation(pages)
    
    # Add sidebar info
    with st.sidebar:
        st.markdown("# 🏠 Kozy Stock")
        st.markdown("### Phân tích cổ phiếu Việt Nam")
        
        st.markdown("---")
        st.markdown("### 📊 Thông tin thị trường")
        st.info("VN-Index: 1,280.5 (+0.8%)")
        st.info("HNX-Index: 235.2 (+0.3%)")
        
        st.markdown("---")
        st.markdown("### 🔗 Liên kết nhanh")
        st.markdown("- [Tin tức](#)")
        st.markdown("- [Lịch sự kiện](#)")
        st.markdown("- [Báo cáo](#)")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        📦 Kozy Stock v2.0.0<br>
        🚀 Modern Navigation
        </div>
        """, unsafe_allow_html=True)
    
    # Run the selected page
    pg.run()

if __name__ == "__main__":
    main()
