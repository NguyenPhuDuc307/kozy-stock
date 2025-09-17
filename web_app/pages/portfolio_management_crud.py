"""
📊 PORTFOLIO MANAGEMENT - Quản lý nhiều danh mục giao dịch
=========================================================

Trang này cho phép tạo, sửa, xóa và quản lý nhiều danh mục giao dịch khác nhau
"""

def render_portfolio_management_page():
    """Render trang quản lý danh mục giao dịch"""
    
    # Import tất cả trong function để tránh lỗi exec
    import streamlit as st
    import pandas as pd
    from datetime import datetime
    import sys
    import os
    
    # Thêm path để import modules
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.join(project_root, 'src'))
    
    from utils.trading_portfolio_manager import TradingPortfolioManager
    
    def render_portfolios_list(portfolio_manager):
        """Hiển thị danh sách các danh mục"""
        st.subheader("📋 Danh sách Danh mục Giao dịch")
        
        portfolios = portfolio_manager.get_portfolios()
        
        if not portfolios:
            st.info("📝 Chưa có danh mục nào. Hãy tạo danh mục đầu tiên!")
            return
        
        # Tạo DataFrame để hiển thị
        portfolio_data = []
        for portfolio in portfolios:
            # Tính lãi/lỗ từ holdings để đảm bảo consistency
            holdings_df = portfolio_manager.get_portfolio_summary(portfolio["id"])
            if holdings_df is not None and not holdings_df.empty:
                total_profit_loss_from_holdings = 0
                for _, row in holdings_df.iterrows():
                    # Parse profit_loss string (format: "+1,234" or "-1,234") 
                    profit_loss_str = row['Profit_Loss'].replace(',', '').replace('+', '')
                    total_profit_loss_from_holdings += float(profit_loss_str)
            else:
                total_profit_loss_from_holdings = 0
            
            portfolio_data.append({
                "Tên danh mục": portfolio["name"],
                "Mô tả": portfolio["description"][:50] + "..." if len(portfolio["description"]) > 50 else portfolio["description"],
                "Chiến lược": portfolio["strategy"],
                "Vốn ban đầu": f"{portfolio['initial_cash']:,.0f} VNĐ",
                "Tổng đầu tư": f"{portfolio.get('total_invested', 0):,.0f} VNĐ",
                "Lãi/Lỗ": f"{total_profit_loss_from_holdings:+,.0f} VNĐ",
                "Ngày tạo": portfolio["created_date"].split(" ")[0],
                "ID": portfolio["id"]
            })
        
        df = pd.DataFrame(portfolio_data)
        
        # Hiển thị bảng
        st.dataframe(
            df.drop(columns=["ID"]),
            use_container_width=True,
            hide_index=True
        )
        
        # Chi tiết danh mục được chọn
        st.markdown("---")
        st.subheader("🔍 Chi tiết Danh mục")
        
        portfolio_names = [f"{p['name']} ({p['id']})" for p in portfolios]
        selected_portfolio = st.selectbox(
            "Chọn danh mục để xem chi tiết:",
            portfolio_names,
            key="detail_portfolio_select"
        )
        
        if selected_portfolio:
            # Lấy ID từ tên được chọn
            portfolio_id = selected_portfolio.split("(")[-1].strip(")")
            portfolio_info = portfolio_manager.get_portfolio(portfolio_id)
            
            if portfolio_info:
                # Lấy holdings để tính toán chính xác từ cùng 1 nguồn
                holdings_df = portfolio_manager.get_portfolio_summary(portfolio_id)
                
                # Tính tổng từ holdings_df để đảm bảo consistency
                if holdings_df is not None and not holdings_df.empty:
                    total_profit_loss_from_holdings = 0
                    for _, row in holdings_df.iterrows():
                        # Parse profit_loss string (format: "+1,234" or "-1,234") 
                        profit_loss_str = row['Profit_Loss'].replace(',', '').replace('+', '')
                        total_profit_loss_from_holdings += float(profit_loss_str)
                else:
                    total_profit_loss_from_holdings = 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("💰 Vốn ban đầu", f"{portfolio_info['initial_cash']:,.0f} VNĐ")
                    st.metric("📈 Tổng đầu tư", f"{portfolio_info.get('total_invested', 0):,.0f} VNĐ")
                    st.metric("💵 Tiền mặt", f"{portfolio_info.get('current_cash', 0):,.0f} VNĐ")
                
                with col2:
                    st.metric("💎 Tổng giá trị", f"{portfolio_info.get('total_value', 0):,.0f} VNĐ")
                    st.metric("📊 Lãi/Lỗ", f"{total_profit_loss_from_holdings:,.0f} VNĐ", delta=f"{total_profit_loss_from_holdings:,.0f}")
                
                # Hiển thị holdings
                
                holdings_df = portfolio_manager.get_portfolio_summary(portfolio_id)
                if holdings_df is not None and not holdings_df.empty:
                    st.subheader("📊 Cổ phiếu đang nắm giữ")
                    
                    # Style dataframe với màu sắc
                    def color_profit_loss(val):
                        """Tô màu cho cột lãi/lỗ"""
                        if isinstance(val, str):
                            # Xử lý cho Profit_Loss
                            if val.replace(',', '').replace('+', '').replace('-', '').replace('.0', '').isdigit():
                                if val.startswith('+') or (not val.startswith('-') and float(val.replace(',', '')) > 0):
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                                elif val.startswith('-'):
                                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                            # Xử lý cho Profit_Loss_Pct  
                            elif '%' in val:
                                clean_val = val.replace('+', '').replace('%', '')
                                if clean_val.startswith('-'):
                                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                                else:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        return ''
                    
                    styled_df = holdings_df.style.map(
                        color_profit_loss, 
                        subset=['Profit_Loss', 'Profit_Loss_Pct']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
    
    def render_create_portfolio(portfolio_manager):
        """Form tạo danh mục mới"""
        st.subheader("➕ Tạo Danh mục Mới")
        
        with st.form("create_portfolio_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("📝 Tên danh mục *", placeholder="VD: Danh mục Ngân hàng")
                strategy = st.selectbox(
                    "🎯 Chiến lược đầu tư",
                    ["Value Investing", "Growth Investing", "Dividend Investing", "Swing Trading", "Day Trading", "Khác"]
                )
            
            with col2:
                initial_cash = st.number_input("💰 Vốn ban đầu (VNĐ)", min_value=0, value=10000000, step=1000000)
                description = st.text_area("📋 Mô tả", placeholder="Mô tả về danh mục và mục tiêu đầu tư...")
            
            submitted = st.form_submit_button("✅ Tạo danh mục", width='stretch')
            
            if submitted:
                if not name.strip():
                    st.error("❌ Vui lòng nhập tên danh mục!")
                else:
                    portfolio_id = portfolio_manager.create_portfolio(
                        name=name.strip(),
                        description=description.strip(),
                        initial_cash=initial_cash,
                        strategy=strategy
                    )
                    st.success(f"✅ Đã tạo danh mục '{name}' thành công!")
                    st.info(f"🆔 ID danh mục: {portfolio_id}")
                    st.rerun()
    
    def render_edit_portfolio(portfolio_manager):
        """Form chỉnh sửa danh mục"""
        st.subheader("✏️ Chỉnh sửa Danh mục")
        
        portfolios = portfolio_manager.get_portfolios()
        
        if not portfolios:
            st.info("📝 Chưa có danh mục nào để chỉnh sửa.")
            return
        
        # Chọn danh mục
        portfolio_names = [f"{p['name']} ({p['id']})" for p in portfolios]
        selected_portfolio = st.selectbox(
            "Chọn danh mục cần chỉnh sửa:",
            portfolio_names,
            key="edit_portfolio_select"
        )
        
        if selected_portfolio:
            # Lấy thông tin danh mục hiện tại
            portfolio_id = selected_portfolio.split("(")[-1].strip(")")
            portfolio_info = portfolio_manager.get_portfolio(portfolio_id)
            
            if portfolio_info:
                with st.form("edit_portfolio_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_name = st.text_input("📝 Tên danh mục", value=portfolio_info["name"])
                        strategies = ["Value Investing", "Growth Investing", "Dividend Investing", "Swing Trading", "Day Trading", "Khác"]
                        current_strategy_index = 0
                        if portfolio_info["strategy"] in strategies:
                            current_strategy_index = strategies.index(portfolio_info["strategy"])
                        
                        new_strategy = st.selectbox(
                            "🎯 Chiến lược đầu tư",
                            strategies,
                            index=current_strategy_index
                        )
                    
                    with col2:
                        new_description = st.text_area("📋 Mô tả", value=portfolio_info["description"])
                    
                    submitted = st.form_submit_button("💾 Cập nhật", width='stretch')
                    
                    if submitted:
                        if not new_name.strip():
                            st.error("❌ Vui lòng nhập tên danh mục!")
                        else:
                            success = portfolio_manager.update_portfolio(
                                portfolio_id=portfolio_id,
                                name=new_name.strip(),
                                description=new_description.strip(),
                                strategy=new_strategy
                            )
                            
                            if success:
                                st.success("✅ Đã cập nhật danh mục thành công!")
                                st.rerun()
                            else:
                                st.error("❌ Có lỗi khi cập nhật danh mục!")
    
    def render_delete_portfolio(portfolio_manager):
        """Form xóa danh mục"""
        st.subheader("🗑️ Xóa Danh mục")
        
        portfolios = portfolio_manager.get_portfolios()
        
        if not portfolios:
            st.info("📝 Chưa có danh mục nào để xóa.")
            return
        
        # Chọn danh mục
        portfolio_names = [f"{p['name']} ({p['id']})" for p in portfolios]
        selected_portfolio = st.selectbox(
            "Chọn danh mục cần xóa:",
            portfolio_names,
            key="delete_portfolio_select"
        )
        
        if selected_portfolio:
            # Lấy thông tin danh mục
            portfolio_id = selected_portfolio.split("(")[-1].strip(")")
            portfolio_info = portfolio_manager.get_portfolio(portfolio_id)
            
            if portfolio_info:
                # Hiển thị thông tin cảnh báo
                st.warning("⚠️ **Cảnh báo**: Việc xóa danh mục sẽ:")
                st.markdown("""
                - Đánh dấu danh mục là không hoạt động (soft delete)
                - Không thể khôi phục được
                - Lịch sử giao dịch vẫn được lưu trữ
                """)
                
                # Hiển thị thông tin danh mục
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Tên**: {portfolio_info['name']}")
                    st.info(f"**Mô tả**: {portfolio_info['description']}")
                with col2:
                    st.info(f"**Vốn ban đầu**: {portfolio_info['initial_cash']:,.0f} VNĐ")
                    st.info(f"**Tổng đầu tư**: {portfolio_info.get('total_invested', 0):,.0f} VNĐ")
                
                # Xác nhận xóa
                confirm_text = st.text_input(
                    f"Gõ '{portfolio_info['name']}' để xác nhận xóa:",
                    placeholder=f"Gõ '{portfolio_info['name']}' để xác nhận"
                )
                
                if st.button("🗑️ XÓA VĨNH VIỄN", type="primary", width='stretch'):
                    if confirm_text == portfolio_info['name']:
                        success = portfolio_manager.delete_portfolio(portfolio_id)
                        if success:
                            st.success("✅ Đã xóa danh mục thành công!")
                            st.rerun()
                        else:
                            st.error("❌ Có lỗi khi xóa danh mục!")
                    else:
                        st.error("❌ Tên xác nhận không đúng!")
    
    def render_portfolio_statistics(portfolio_manager):
        """Hiển thị thống kê tổng quan"""
        st.subheader("📊 Thống kê Tổng quan")
        
        portfolios = portfolio_manager.get_portfolios()
        
        if not portfolios:
            st.info("📝 Chưa có danh mục nào để thống kê.")
            return
        
        # Tính toán thống kê tổng quan
        total_portfolios = len(portfolios)
        total_initial_cash = sum(p.get('initial_cash', 0) for p in portfolios)
        total_invested = sum(p.get('total_invested', 0) for p in portfolios)
        total_profit_loss = sum(p.get('total_profit_loss', 0) for p in portfolios)
        
        # Hiển thị metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Số danh mục", total_portfolios)
        
        with col2:
            st.metric("💰 Tổng vốn ban đầu", f"{total_initial_cash:,.0f} VNĐ")
        
        with col3:
            st.metric("📈 Tổng đầu tư", f"{total_invested:,.0f} VNĐ")
        
        with col4:
            profit_loss_pct = (total_profit_loss / total_invested * 100) if total_invested > 0 else 0
            st.metric(
                "💵 Tổng lãi/lỗ", 
                f"{total_profit_loss:,.0f} VNĐ",
                f"{profit_loss_pct:.2f}%"
            )
        
        # Biểu đồ hiệu suất các danh mục
        if len(portfolios) > 0:
            st.markdown("---")
            st.subheader("📈 Hiệu suất các Danh mục")
            
            # Tạo DataFrame cho biểu đồ
            chart_data = []
            for p in portfolios:
                profit_loss = p.get('total_profit_loss', 0)
                invested = p.get('total_invested', 1)  # Tránh chia cho 0
                profit_pct = (profit_loss / invested * 100) if invested > 0 else 0
                
                chart_data.append({
                    "Danh mục": p["name"],
                    "Lãi/Lỗ (%)": profit_pct,
                    "Tổng đầu tư": invested,
                    "Lãi/Lỗ (VNĐ)": profit_loss
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            # Biểu đồ cột
            st.bar_chart(chart_df.set_index("Danh mục")["Lãi/Lỗ (%)"])
            
            # Bảng chi tiết
            st.dataframe(chart_df, use_container_width=True, hide_index=True)
    
    # Main function logic
    st.title("📊 Quản lý Danh mục Giao dịch")
    st.markdown("---")
    
    # Khởi tạo portfolio manager
    if 'trading_portfolio_manager' not in st.session_state:
        st.session_state.trading_portfolio_manager = TradingPortfolioManager()
    
    portfolio_manager = st.session_state.trading_portfolio_manager
    
    # Sidebar cho các chức năng
    with st.sidebar:
        st.header("🛠️ Chức năng")
        action = st.selectbox(
            "Chọn thao tác:",
            ["📋 Xem danh sách", "➕ Tạo mới", "✏️ Chỉnh sửa", "🗑️ Xóa", "📊 Thống kê"]
        )
        
        st.markdown("---")
        st.subheader("🔄 Bảo trì")
        
        if st.button("🔄 Cập nhật giá hiện tại", width='stretch'):
            with st.spinner("Đang cập nhật giá hiện tại cho tất cả danh mục..."):
                try:
                    success = portfolio_manager.refresh_all_portfolios()
                    if success:
                        st.success("✅ Đã cập nhật giá hiện tại thành công!")
                        st.rerun()
                    else:
                        st.error("❌ Có lỗi khi cập nhật!")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
        
        st.caption("💡 Sử dụng nút này để cập nhật giá hiện tại từ thị trường")
    
    if action == "📋 Xem danh sách":
        render_portfolios_list(portfolio_manager)
    
    elif action == "➕ Tạo mới":
        render_create_portfolio(portfolio_manager)
    
    elif action == "✏️ Chỉnh sửa":
        render_edit_portfolio(portfolio_manager)
    
    elif action == "🗑️ Xóa":
        render_delete_portfolio(portfolio_manager)
    
    elif action == "📊 Thống kê":
        render_portfolio_statistics(portfolio_manager)

# Entry point cho page
if __name__ == "__main__":
    render_portfolio_management_page()
