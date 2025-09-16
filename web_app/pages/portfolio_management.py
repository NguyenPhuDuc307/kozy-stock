"""
📁 PORTFOLIO MANAGEMENT PAGE - Trang quản lý danh mục
==================================================

Trang này cho phép người dùng tạo, sửa, xóa các danh mục cổ phiếu
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_portfolio_management():
    """
    Render trang quản lý danh mục
    """
    st.markdown("# 📁 Quản lý danh mục cổ phiếu")
    
    try:
        from src.utils.portfolio_manager import PortfolioManager
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager()
        
        # Tabs for different operations
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Xem danh mục", "➕ Thêm danh mục", "✏️ Sửa danh mục", "🗑️ Xóa danh mục"])
        
        with tab1:
            st.subheader("📋 Danh sách các danh mục hiện có")
            
            portfolios = portfolio_manager.get_portfolios()
            
            if not portfolios:
                st.info("📝 Chưa có danh mục nào. Hãy tạo danh mục đầu tiên!")
            else:
                for portfolio_name, stocks in portfolios.items():
                    with st.expander(f"📂 {portfolio_name} ({len(stocks)} cổ phiếu)", expanded=False):
                        st.write("**Danh sách cổ phiếu:**")
                        
                        # Hiển thị dưới dạng cột để dễ đọc
                        cols = st.columns(5)
                        for i, stock in enumerate(sorted(stocks)):
                            with cols[i % 5]:
                                st.write(f"• {stock}")
                        
                        # Thêm/xóa cổ phiếu nhanh
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_stock = st.text_input(f"Thêm cổ phiếu vào {portfolio_name}:", key=f"add_{portfolio_name}")
                            if st.button(f"➕ Thêm", key=f"btn_add_{portfolio_name}"):
                                if new_stock:
                                    portfolio_manager.add_stock_to_portfolio(portfolio_name, new_stock)
                                    st.rerun()
                        
                        with col2:
                            if stocks:
                                remove_stock = st.selectbox(f"Xóa cổ phiếu khỏi {portfolio_name}:", 
                                                          [""] + sorted(stocks), key=f"remove_{portfolio_name}")
                                if st.button(f"🗑️ Xóa", key=f"btn_remove_{portfolio_name}"):
                                    if remove_stock:
                                        portfolio_manager.remove_stock_from_portfolio(portfolio_name, remove_stock)
                                        st.rerun()
        
        with tab2:
            st.subheader("➕ Thêm danh mục mới")
            
            with st.form("add_portfolio_form"):
                portfolio_name = st.text_input("Tên danh mục:", placeholder="VD: Cổ phiếu nhỏ")
                
                st.write("**Nhập danh sách cổ phiếu (mỗi mã một dòng):**")
                stocks_input = st.text_area("Danh sách cổ phiếu:", 
                                           placeholder="VCB\nFPT\nVNM\n...", height=200)
                
                submitted = st.form_submit_button("✅ Tạo danh mục")
                
                if submitted:
                    if portfolio_name and stocks_input:
                        # Parse stocks from input
                        stocks = [stock.strip().upper() for stock in stocks_input.split('\n') if stock.strip()]
                        if stocks:
                            portfolio_manager.add_portfolio(portfolio_name, stocks)
                            st.rerun()
                        else:
                            st.error("❌ Vui lòng nhập ít nhất một mã cổ phiếu!")
                    else:
                        st.error("❌ Vui lòng nhập đầy đủ tên danh mục và danh sách cổ phiếu!")
        
        with tab3:
            st.subheader("✏️ Sửa danh mục")
            
            portfolio_names = portfolio_manager.get_portfolio_names()
            
            if not portfolio_names:
                st.info("📝 Chưa có danh mục nào để sửa.")
            else:
                selected_portfolio = st.selectbox("Chọn danh mục cần sửa:", portfolio_names)
                
                if selected_portfolio:
                    current_stocks = portfolio_manager.get_portfolio_stocks(selected_portfolio)
                    
                    with st.form("edit_portfolio_form"):
                        st.write(f"**Sửa danh mục: {selected_portfolio}**")
                        
                        # Hiển thị danh sách hiện tại
                        current_stocks_text = '\n'.join(current_stocks)
                        stocks_input = st.text_area("Danh sách cổ phiếu (mỗi mã một dòng):", 
                                                   value=current_stocks_text, height=200)
                        
                        submitted = st.form_submit_button("💾 Cập nhật danh mục")
                        
                        if submitted:
                            if stocks_input:
                                # Parse stocks from input
                                stocks = [stock.strip().upper() for stock in stocks_input.split('\n') if stock.strip()]
                                if stocks:
                                    portfolio_manager.update_portfolio(selected_portfolio, stocks)
                                    st.rerun()
                                else:
                                    st.error("❌ Danh mục không thể trống!")
                            else:
                                st.error("❌ Vui lòng nhập danh sách cổ phiếu!")
        
        with tab4:
            st.subheader("🗑️ Xóa danh mục")
            
            portfolio_names = portfolio_manager.get_portfolio_names()
            
            if not portfolio_names:
                st.info("📝 Chưa có danh mục nào để xóa.")
            else:
                selected_portfolio = st.selectbox("Chọn danh mục cần xóa:", [""] + portfolio_names)
                
                if selected_portfolio:
                    stocks = portfolio_manager.get_portfolio_stocks(selected_portfolio)
                    st.warning(f"⚠️ Bạn có chắc chắn muốn xóa danh mục **{selected_portfolio}** với {len(stocks)} cổ phiếu?")
                    
                    # Hiển thị preview
                    with st.expander("Xem trước danh mục sẽ bị xóa"):
                        cols = st.columns(5)
                        for i, stock in enumerate(sorted(stocks)):
                            with cols[i % 5]:
                                st.write(f"• {stock}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Xác nhận xóa", type="primary"):
                            portfolio_manager.delete_portfolio(selected_portfolio)
                            st.rerun()
                    with col2:
                        if st.button("❌ Hủy"):
                            st.rerun()
        
        # Thống kê tổng quan
        st.markdown("---")
        st.subheader("📊 Thống kê tổng quan")
        
        portfolios = portfolio_manager.get_portfolios()
        total_portfolios = len(portfolios)
        total_unique_stocks = len(portfolio_manager.get_all_stocks())
        total_stocks = sum(len(stocks) for stocks in portfolios.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số danh mục", total_portfolios)
        with col2:
            st.metric("Tổng số cổ phiếu", total_stocks)
        with col3:
            st.metric("Cổ phiếu duy nhất", total_unique_stocks)
            
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")

# Main page function for st.Page
render_portfolio_management()
