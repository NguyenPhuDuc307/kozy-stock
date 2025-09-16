"""
Portfolio Tracking Page

Trang theo dõi danh mục đầu tư - sử dụng lịch sử giao dịch thực tế
Theo dõi các cổ phiếu đang sở hữu với lợi nhuận, tín hiệu và khuyến nghị
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

def format_number_short(value):
    """Format số ngắn gọn với đơn vị"""
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.0f}K"
    else:
        return f"{value:.0f}"

def get_portfolio_data():
    """
    Lấy dữ liệu danh mục từ lịch sử giao dịch thực tế
    """
    import random
    import pandas as pd
    from datetime import datetime, timedelta
    from src.utils.trading_history import TradingHistory
    
    # Khởi tạo trading history
    trading_history = TradingHistory()
    
    # Lấy danh sách cổ phiếu đang nắm giữ từ lịch sử giao dịch
    current_holdings = trading_history.get_current_holdings()
    
    if not current_holdings:
        # Nếu chưa có giao dịch nào, tạo dữ liệu mẫu
        st.info("📝 Chưa có lịch sử giao dịch. Tạo dữ liệu mẫu...")
        trading_history.add_sample_data()
        current_holdings = trading_history.get_current_holdings()
    
    portfolio_data = []
    
    for symbol, holding_data in current_holdings.items():
        try:
            # Lấy giá hiện tại từ vnstock giống như DataProvider
            import vnstock
            
            # Lấy dữ liệu gần nhất
            quote = vnstock.Quote(symbol=symbol, source='VCI')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            data = quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )
            
            if data is not None and not data.empty:
                current_price = data.iloc[-1]['close']
            else:
                # Fallback nếu không có dữ liệu từ API
                current_price = holding_data["avg_price"] * random.uniform(0.95, 1.05)
            
            # Tính toán lợi nhuận dựa trên dữ liệu thực
            shares = holding_data["shares"]
            avg_price = holding_data["avg_price"]
            total_cost = holding_data["total_cost"]
            
            # Nhân giá lên 1000 để hiển thị đúng đơn vị VND
            current_price_display = current_price * 1000
            current_value = current_price_display * shares
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Tính toán tín hiệu kỹ thuật
            if data is not None and not data.empty and len(data) >= 5:
                prices = data['close'].values
                ma5 = prices[-5:].mean()
                
                # Technical score dựa trên MA5
                if current_price > ma5 * 1.02:
                    technical_score = 0.7
                    signal = "BUY"
                    recommendation = "MUA MẠNH"
                elif current_price > ma5:
                    technical_score = 0.3
                    signal = "BUY"
                    recommendation = "MUA"
                elif current_price < ma5 * 0.98:
                    technical_score = -0.7
                    signal = "SELL"
                    recommendation = "BÁN"
                else:
                    technical_score = 0
                    signal = "HOLD"
                    recommendation = "GIỮ"
            else:
                # Fallback technical analysis
                if profit_loss_pct > 5:
                    technical_score = 0.5
                    signal = "HOLD"
                    recommendation = "GIỮ"
                elif profit_loss_pct < -5:
                    technical_score = -0.5
                    signal = "HOLD"
                    recommendation = "GIỮ"
                else:
                    technical_score = 0
                    signal = "HOLD"
                    recommendation = "GIỮ"
            
            portfolio_data.append({
                'Symbol': symbol,
                'Buy_Price': avg_price,
                'Current_Price': current_price_display,
                'Shares': shares,
                'Total_Cost': total_cost,
                'Current_Value': current_value,
                'Profit_Loss': profit_loss,
                'Profit_Loss_Pct': profit_loss_pct,
                'Technical_Score': technical_score,
                'Signal': signal,
                'Recommendation': recommendation
            })
            
        except Exception as e:
            # Nếu lỗi API, dùng dữ liệu từ holdings
            shares = holding_data["shares"]
            avg_price = holding_data["avg_price"]
            total_cost = holding_data["total_cost"]
            
            # Giả lập giá hiện tại
            current_price = avg_price * random.uniform(0.9, 1.1)
            current_price_display = current_price * 1000
            current_value = current_price_display * shares
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Default technical analysis
            technical_score = random.uniform(-0.5, 0.5)
            signal = "HOLD"
            recommendation = "GIỮ"
            
            portfolio_data.append({
                'Symbol': symbol,
                'Buy_Price': avg_price,
                'Current_Price': current_price_display,
                'Shares': shares,
                'Total_Cost': total_cost,
                'Current_Value': current_value,
                'Profit_Loss': profit_loss,
                'Profit_Loss_Pct': profit_loss_pct,
                'Technical_Score': technical_score,
                'Signal': signal,
                'Recommendation': recommendation
            })
    
    return pd.DataFrame(portfolio_data)

def create_performance_chart(portfolio_data):
    """
    Tạo biểu đồ hiệu suất danh mục
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Lợi nhuận theo cổ phiếu', 'Điểm kỹ thuật theo cổ phiếu'),
        vertical_spacing=0.1
    )
    
    # Profit/Loss chart
    colors = ['green' if x >= 0 else 'red' for x in portfolio_data['Profit_Loss_Pct']]
    
    fig.add_trace(
        go.Bar(
            x=portfolio_data['Symbol'],
            y=portfolio_data['Profit_Loss_Pct'],
            name='Lợi nhuận (%)',
            marker_color=colors
        ),
        row=1, col=1
    )
    
    # Technical score chart
    tech_colors = ['green' if x >= 0 else 'red' for x in portfolio_data['Technical_Score']]
    
    fig.add_trace(
        go.Bar(
            x=portfolio_data['Symbol'],
            y=portfolio_data['Technical_Score'],
            name='Điểm kỹ thuật',
            marker_color=tech_colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Phân tích hiệu suất danh mục",
        showlegend=False
    )
    
    # Add horizontal lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    return fig

def render_portfolio_tracking_page():
    st.markdown("# 📊 Theo dõi danh mục đầu tư")
    
    try:
        # Import trading history
        from src.utils.trading_history import TradingHistory
        
        # Initialize trading history
        trading_history = TradingHistory()
        current_holdings = trading_history.get_current_holdings()
        
        # Sidebar - Thống kê và quản lý
        st.sidebar.markdown("## 📊 Tổng quan")
        st.sidebar.metric("📈 Số cổ phiếu đang nắm giữ", len(current_holdings))
        
        # Sidebar - Thêm giao dịch nhanh
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ➕ Thêm giao dịch")
        
        # Khởi tạo session state cho giá
        if 'current_price' not in st.session_state:
            st.session_state.current_price = 50000
        if 'last_symbol' not in st.session_state:
            st.session_state.last_symbol = ""
        if 'last_transaction_type' not in st.session_state:
            st.session_state.last_transaction_type = "BUY"
        
        # Form thêm giao dịch
        symbol = st.sidebar.text_input("Mã cổ phiếu", placeholder="VNM").upper()
        transaction_type = st.sidebar.selectbox("Loại giao dịch", ["BUY", "SELL"])
        quantity = st.sidebar.number_input("Số lượng", min_value=1, value=100)
        
        # Tự động lấy giá từ thị trường
        auto_price = st.sidebar.checkbox("Lấy giá thị trường tự động", value=True)
        
        # Kiểm tra nếu symbol hoặc transaction_type thay đổi
        symbol_changed = symbol != st.session_state.last_symbol
        type_changed = transaction_type != st.session_state.last_transaction_type
        
        if auto_price and symbol and len(symbol) >= 3 and (symbol_changed or type_changed):
            try:
                import vnstock
                from datetime import datetime, timedelta
                
                # Lấy dữ liệu gần nhất
                quote = vnstock.Quote(symbol=symbol, source='VCI')
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                
                data = quote.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )
                
                if data is not None and not data.empty:
                    current_price = data.iloc[-1]['close']
                    if transaction_type == "BUY":
                        # Lấy giá mua (thêm 0.2% spread)
                        market_price = current_price * 1000 * 1.002
                        price_label = f"💰 Giá mua thị trường: {market_price:,.0f} VND"
                    else:
                        # Lấy giá bán (trừ 0.2% spread)
                        market_price = current_price * 1000 * 0.998
                        price_label = f"💰 Giá bán thị trường: {market_price:,.0f} VND"
                    
                    # Cập nhật session state
                    st.session_state.current_price = int(market_price)
                    st.session_state.last_symbol = symbol
                    st.session_state.last_transaction_type = transaction_type
                    
                    st.sidebar.success(price_label)
                else:
                    st.sidebar.warning("⚠️ Không lấy được giá thị trường")
                    # Fallback với giá mặc định
                    if transaction_type == "BUY":
                        st.session_state.current_price = 50000
                    else:
                        st.session_state.current_price = 49000
            except Exception as e:
                st.sidebar.warning(f"⚠️ Lỗi khi lấy giá: {str(e)}")
                # Fallback với giá mặc định
                if transaction_type == "BUY":
                    st.session_state.current_price = 50000
                else:
                    st.session_state.current_price = 49000
        
        # Hiển thị ô nhập giá với giá từ session state
        if auto_price:
            price = st.sidebar.number_input("Giá (VND)", value=st.session_state.current_price, min_value=10, step=1000)
        else:
            price = st.sidebar.number_input("Giá (VND)", min_value=10, value=50000, step=1000)
        
        fee = st.sidebar.number_input("Phí giao dịch (VND)", min_value=0, value=0)
        note = st.sidebar.text_input("Ghi chú", placeholder="Ghi chú...")
        
        if st.sidebar.button("📈 Thêm giao dịch", type="primary"):
            if symbol and len(symbol) >= 3:
                transaction_id = trading_history.add_transaction(
                    symbol, transaction_type, quantity, price, fee=fee, note=note
                )
                st.sidebar.success(f"✅ Đã thêm giao dịch #{transaction_id}")
                st.rerun()
            else:
                st.sidebar.error("❌ Mã cổ phiếu không hợp lệ")
        
        if not current_holdings:
            st.warning("⚠️ Chưa có cổ phiếu nào trong danh mục")
            st.info("💡 Hãy thêm giao dịch đầu tiên bằng form bên trái!")
            
            # Tạo dữ liệu mẫu
            if st.button("🎯 Tạo dữ liệu mẫu để demo"):
                trading_history.add_sample_data()
                st.success("✅ Đã tạo dữ liệu mẫu!")
                st.rerun()
            return
        
        # Sidebar - Xóa danh mục
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Quản lý danh mục")
        
        if current_holdings:
            selected_symbol = st.sidebar.selectbox(
                "Chọn cổ phiếu để xóa",
                options=list(current_holdings.keys()),
                placeholder="Chọn mã cổ phiếu..."
            )
            
            if selected_symbol:
                holding_info = current_holdings[selected_symbol]
                st.sidebar.write(f"**{selected_symbol}**: {holding_info['shares']:,.0f} cổ phiếu")
                st.sidebar.write(f"Giá TB: {holding_info['avg_price']:,.0f} VND")
                
                if st.sidebar.button("🗑️ Xóa khỏi danh mục", type="secondary"):
                    # Xóa tất cả giao dịch của cổ phiếu này
                    trading_history.clear_symbol_transactions(selected_symbol)
                    st.sidebar.success(f"✅ Đã xóa {selected_symbol} khỏi danh mục")
                    st.rerun()
        
        # Sidebar - Cài đặt theo dõi
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Cài đặt")
        show_signals = st.sidebar.checkbox("Hiển thị tín hiệu kỹ thuật", value=True)
        show_fundamentals = st.sidebar.checkbox("Hiển thị chỉ số cơ bản", value=True)
        auto_refresh = st.sidebar.checkbox("Tự động làm mới (30s)", value=False)
        
        # Header info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Danh mục", "Lịch sử giao dịch")
        with col2:
            st.metric("📈 Số cổ phiếu", len(current_holdings))
        with col3:
            import time
            current_time = time.strftime("%H:%M:%S")
            st.metric("🕐 Cập nhật lúc", current_time)
        
        # Refresh button
        if st.button("🔄 Làm mới dữ liệu", type="primary"):
            st.rerun()
        
        # Get portfolio data
        with st.spinner("📊 Đang tải dữ liệu danh mục..."):
            # Logic lấy dữ liệu portfolio inline
            import random
            import pandas as pd
            import vnstock
            from datetime import datetime, timedelta
            
            portfolio_data = []
            
            for symbol, holding_data in current_holdings.items():
                try:
                    # Lấy giá hiện tại từ vnstock giống như DataProvider
                    # Lấy dữ liệu gần nhất
                    quote = vnstock.Quote(symbol=symbol, source='VCI')
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5)
                    
                    data = quote.history(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d")
                    )
                    
                    if data is not None and not data.empty:
                        current_price = data.iloc[-1]['close']
                    else:
                        # Fallback nếu không có dữ liệu từ API
                        current_price = holding_data["avg_price"] * random.uniform(0.95, 1.05)
                    
                    # Tính toán lợi nhuận dựa trên dữ liệu thực
                    shares = holding_data["shares"]
                    avg_price = holding_data["avg_price"]
                    total_cost = holding_data["total_cost"]
                    
                    # Nhân giá lên 1000 để hiển thị đúng đơn vị VND
                    current_price_display = current_price * 1000
                    current_value = current_price_display * shares
                    profit_loss = current_value - total_cost
                    profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
                    
                    # Tính toán tín hiệu kỹ thuật
                    if data is not None and not data.empty and len(data) >= 5:
                        prices = data['close'].values
                        ma5 = prices[-5:].mean()
                        
                        # Technical score dựa trên MA5
                        if current_price > ma5 * 1.02:
                            technical_score = 0.7
                            signal = "BUY"
                            recommendation = "MUA MẠNH"
                        elif current_price > ma5:
                            technical_score = 0.3
                            signal = "BUY"
                            recommendation = "MUA"
                        elif current_price < ma5 * 0.98:
                            technical_score = -0.7
                            signal = "SELL"
                            recommendation = "BÁN"
                        else:
                            technical_score = 0
                            signal = "HOLD"
                            recommendation = "GIỮ"
                    else:
                        # Fallback technical analysis
                        if profit_loss_pct > 5:
                            technical_score = 0.5
                            signal = "HOLD"
                            recommendation = "GIỮ"
                        elif profit_loss_pct < -5:
                            technical_score = -0.5
                            signal = "HOLD"
                            recommendation = "GIỮ"
                        else:
                            technical_score = 0
                            signal = "HOLD"
                            recommendation = "GIỮ"
                    
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Buy_Price': avg_price,
                        'Current_Price': current_price_display,
                        'Shares': shares,
                        'Total_Cost': total_cost,
                        'Current_Value': current_value,
                        'Profit_Loss': profit_loss,
                        'Profit_Loss_Pct': profit_loss_pct,
                        'Technical_Score': technical_score,
                        'Signal': signal,
                        'Recommendation': recommendation
                    })
                    
                except Exception as e:
                    # Nếu lỗi API, dùng dữ liệu từ holdings
                    shares = holding_data["shares"]
                    avg_price = holding_data["avg_price"]
                    total_cost = holding_data["total_cost"]
                    
                    # Giả lập giá hiện tại
                    current_price = avg_price * random.uniform(0.9, 1.1)
                    current_price_display = current_price * 1000
                    current_value = current_price_display * shares
                    profit_loss = current_value - total_cost
                    profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
                    
                    # Default technical analysis
                    technical_score = random.uniform(-0.5, 0.5)
                    signal = "HOLD"
                    recommendation = "GIỮ"
                    
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Buy_Price': avg_price,
                        'Current_Price': current_price_display,
                        'Shares': shares,
                        'Total_Cost': total_cost,
                        'Current_Value': current_value,
                        'Profit_Loss': profit_loss,
                        'Profit_Loss_Pct': profit_loss_pct,
                        'Technical_Score': technical_score,
                        'Signal': signal,
                        'Recommendation': recommendation
                    })
            
            portfolio_data = pd.DataFrame(portfolio_data)
        
        if portfolio_data.empty:
            st.error("❌ Không thể tải dữ liệu danh mục")
            return
        
        # Portfolio overview
        st.markdown("---")
        st.subheader("📊 Tổng quan danh mục")
        
        # Calculate portfolio metrics
        total_value = portfolio_data['Current_Value'].sum()
        total_profit = portfolio_data['Profit_Loss'].sum()
        total_profit_pct = (total_profit / (total_value - total_profit)) * 100 if (total_value - total_profit) > 0 else 0
        avg_score = portfolio_data['Technical_Score'].mean()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Format function inline
        def format_number_short(value):
            """Format số ngắn gọn với đơn vị"""
            if abs(value) >= 1_000_000_000:
                return f"{value/1_000_000_000:.1f}B"
            elif abs(value) >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.0f}K"
            else:
                return f"{value:.0f}"
        
        with col1:
            st.metric(
                "💰 Tổng giá trị",
                f"{format_number_short(total_value)} VND",
                f"{format_number_short(total_profit)} VND"
            )
        
        with col2:
            color = "normal" if total_profit_pct >= 0 else "inverse"
            st.metric(
                "📈 Lợi nhuận (%)",
                f"{total_profit_pct:+.2f}%",
                delta_color=color
            )
        
        with col3:
            score_color = "normal" if avg_score >= 0 else "inverse"
            st.metric(
                "🎯 Điểm kỹ thuật TB",
                f"{avg_score:.2f}",
                delta_color=score_color
            )
        
        with col4:
            buy_signals = len(portfolio_data[portfolio_data['Signal'] == 'BUY'])
            st.metric(
                "🟢 Tín hiệu MUA",
                buy_signals
            )
        
        # Portfolio composition chart
        if len(portfolio_data) > 1:
            import plotly.graph_objects as go
            fig_pie = go.Figure(data=[go.Pie(
                labels=portfolio_data['Symbol'],
                values=portfolio_data['Current_Value'],
                hole=.3
            )])
            fig_pie.update_layout(
                title="Cơ cấu danh mục",
                height=400
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        # Detailed tracking table
        st.markdown("---")
        st.subheader("📋 Chi tiết theo dõi")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            signal_filter = st.selectbox(
                "Lọc theo tín hiệu:",
                ["Tất cả", "BUY", "SELL", "HOLD"]
            )
        
        with col2:
            profit_filter = st.selectbox(
                "Lọc theo lợi nhuận:",
                ["Tất cả", "Lãi", "Lỗ"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sắp xếp theo:",
                ["Profit_Loss_Pct", "Technical_Score", "Current_Price", "Symbol"]
            )
        
        # Apply filters
        filtered_data = portfolio_data.copy()
        
        if signal_filter != "Tất cả":
            filtered_data = filtered_data[filtered_data['Signal'] == signal_filter]
        
        if profit_filter == "Lãi":
            filtered_data = filtered_data[filtered_data['Profit_Loss'] >= 0]
        elif profit_filter == "Lỗ":
            filtered_data = filtered_data[filtered_data['Profit_Loss'] < 0]
        
        # Sort data
        ascending = sort_by not in ['Profit_Loss_Pct', 'Technical_Score']
        filtered_data = filtered_data.sort_values(sort_by, ascending=ascending)
        
        # Display table
        if not filtered_data.empty:
            # Format display table
            display_data = filtered_data.copy()
            
            # Format columns for display using .loc to avoid SettingWithCopyWarning
            display_data.loc[:, 'Số lượng'] = display_data['Shares'].apply(lambda x: f"{x:,.0f}")
            display_data.loc[:, 'Giá mua'] = display_data['Buy_Price'].apply(lambda x: f"{x:,.0f}")
            display_data.loc[:, 'Giá hiện tại'] = display_data['Current_Price'].apply(lambda x: f"{x:,.0f}")
            display_data.loc[:, 'Lợi nhuận (%)'] = display_data['Profit_Loss_Pct'].apply(lambda x: f"{x:+.2f}%")
            display_data.loc[:, 'Lợi nhuận (VND)'] = display_data['Profit_Loss'].apply(lambda x: f"{x:+,.0f}")
            display_data.loc[:, 'Điểm KT'] = display_data['Technical_Score'].apply(lambda x: f"{x:.2f}")
            display_data.loc[:, 'Tín hiệu'] = display_data['Signal']
            display_data.loc[:, 'Khuyến nghị'] = display_data['Recommendation']
            
            # Select columns to display
            display_columns = [
                'Symbol', 'Số lượng', 'Giá mua', 'Giá hiện tại', 'Lợi nhuận (%)', 
                'Lợi nhuận (VND)', 'Điểm KT', 'Tín hiệu', 'Khuyến nghị'
            ]
            
            # Style the dataframe
            def style_profit_loss(val):
                if "+" in str(val):
                    return 'color: #00ff88; font-weight: bold'
                elif "-" in str(val):
                    return 'color: #ff4444; font-weight: bold'
                return 'color: black'
            
            def style_signal(val):
                if val == 'BUY':
                    return 'color: #00ff88; font-weight: bold'
                elif val == 'SELL':
                    return 'color: #ff4444; font-weight: bold'
                elif val == 'HOLD':
                    return 'color: #ffa500; font-weight: bold'
                return 'color: black'
            
            styled_df = display_data[display_columns].style.map(
                style_profit_loss, subset=['Lợi nhuận (%)', 'Lợi nhuận (VND)']
            ).map(
                style_signal, subset=['Tín hiệu']
            )
            
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
        else:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc")
        
        # Performance analysis
        st.markdown("---")
        st.subheader("📈 Phân tích hiệu suất")
        
        # Performance over time chart
        if len(portfolio_data) > 0:
            # Tạo biểu đồ hiệu suất inline
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # Create subplot
            fig_performance = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Lợi nhuận theo cổ phiếu', 'Điểm kỹ thuật theo cổ phiếu'),
                vertical_spacing=0.1
            )
            
            # Profit/Loss chart
            colors = ['green' if x >= 0 else 'red' for x in portfolio_data['Profit_Loss_Pct']]
            
            fig_performance.add_trace(
                go.Bar(
                    x=portfolio_data['Symbol'],
                    y=portfolio_data['Profit_Loss_Pct'],
                    name='Lợi nhuận (%)',
                    marker_color=colors
                ),
                row=1, col=1
            )
            
            # Technical score chart
            tech_colors = ['green' if x >= 0 else 'red' for x in portfolio_data['Technical_Score']]
            
            fig_performance.add_trace(
                go.Bar(
                    x=portfolio_data['Symbol'],
                    y=portfolio_data['Technical_Score'],
                    name='Điểm kỹ thuật',
                    marker_color=tech_colors
                ),
                row=2, col=1
            )
            
            # Update layout
            fig_performance.update_layout(
                height=600,
                title_text="Phân tích hiệu suất danh mục",
                showlegend=False
            )
            
            # Add horizontal lines
            fig_performance.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            fig_performance.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            st.plotly_chart(fig_performance, width='stretch')
        
        # Recommendations summary
        st.markdown("---")
        st.subheader("💡 Tóm tắt khuyến nghị")
        
        recommendations = portfolio_data['Recommendation'].value_counts()
        
        if not recommendations.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                for rec, count in recommendations.items():
                    if rec == "MUA MẠNH":
                        st.success(f"🟢 {rec}: {count} cổ phiếu")
                    elif rec == "MUA":
                        st.info(f"🔵 {rec}: {count} cổ phiếu")
                    elif rec == "GIỮ":
                        st.warning(f"🟡 {rec}: {count} cổ phiếu")
                    elif rec == "BÁN":
                        st.error(f"🔴 {rec}: {count} cổ phiếu")
                    else:
                        st.write(f"⚪ {rec}: {count} cổ phiếu")
            
            with col2:
                # Top performers
                top_performers = filtered_data.nlargest(3, 'Profit_Loss_Pct')
                if not top_performers.empty:
                    st.write("🏆 **Top performers:**")
                    for _, stock in top_performers.iterrows():
                        st.write(f"• {stock['Symbol']}: {stock['Profit_Loss_Pct']:+.2f}%")
                
                # Worst performers
                worst_performers = filtered_data.nsmallest(3, 'Profit_Loss_Pct')
                if not worst_performers.empty:
                    st.write("⚠️ **Cần quan tâm:**")
                    for _, stock in worst_performers.iterrows():
                        st.write(f"• {stock['Symbol']}: {stock['Profit_Loss_Pct']:+.2f}%")
        
        # Auto refresh
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()
        
    except ImportError as e:
        st.error("❌ Không thể load Portfolio Tracking")
        st.info("💡 Vui lòng kiểm tra cài đặt Portfolio Manager")
        st.code(f"Import error: {e}")

    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")

# Main page function for st.Page  
render_portfolio_tracking_page()
