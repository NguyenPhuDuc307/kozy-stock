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

""", unsafe_allow_html=True)

def main():
    """
    Main application using modern Streamlit navigation
    """
    
    # Define page functions
    def home_page():
        """Home page function for st.Page"""
        exec(open("web_app/pages/home.py").read())
    
    def stock_analysis_page():
        """Stock analysis page function for st.Page"""
        exec(open("web_app/pages/stock_analysis.py").read())
    
    def market_scanner_page():
        """Market scanner page function for st.Page"""
        exec(open("web_app/pages/market_scanner.py").read())
    
    def stock_comparison_page():
        """Stock comparison page function for st.Page"""
        exec(open("web_app/pages/stock_comparison.py").read())
    
    def backtest_page():
        """Backtest page function for st.Page"""
        exec(open("web_app/pages/backtest.py").read())
    
    def portfolio_management_page():
        """Portfolio management page function for st.Page"""
        exec(open("web_app/pages/portfolio_management.py").read())
    
    def portfolio_tracking_page():
        """Portfolio tracking page function for st.Page"""
        exec(open("web_app/pages/portfolio_tracking.py").read())
    
    # Create pages using st.Page
    pages = {
        "Trang chủ": [
            st.Page(home_page, title="Tổng quan thị trường", icon="🏠"),
        ],
        "Phân tích": [
            st.Page(stock_analysis_page, title="Phân tích kỹ thuật", icon="📈"),
            st.Page(market_scanner_page, title="Quét thị trường", icon="🔍"),
            st.Page(stock_comparison_page, title="So sánh cổ phiếu", icon="📊"),
        ],
        "Trading": [
            st.Page(backtest_page, title="Backtest chiến lược", icon="🔄"),
            st.Page(portfolio_tracking_page, title="Theo dõi danh mục", icon="📊"),
        ],
        "Quản lý": [
            st.Page(portfolio_management_page, title="Quản lý danh mục", icon="📁"),
        ]
    }
    
    # Create navigation
    pg = st.navigation(pages)
    # Run the selected page
    pg.run()

if __name__ == "__main__":
    main()
