"""
📊 STOCK COMPARISON PAGE - Trang so sánh cổ phiếu
=================================================

Trang so sánh hiệu suất giữa các cổ phiếu
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_comparison_page():
    """
    Render trang so sánh cổ phiếu
    """
    st.markdown("# 📊 So sánh cổ phiếu")
    
    try:
        # Import portfolio manager
        from src.utils.portfolio_manager import PortfolioManager
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager()
        all_stocks = sorted(portfolio_manager.get_all_stocks())
        
        if not all_stocks:
            # Fallback if no portfolios
            all_stocks = ["VCB", "CTG", "BID", "ACB", "VIC", "FPT", "MSN", "VNM", "PLX", "TCB"]
            st.warning("⚠️ Chưa có danh mục nào. Sử dụng danh sách mặc định.")
            st.info("💡 Hãy vào 'Quản lý danh mục' để tạo danh mục!")
        
        # Basic comparison interface
        st.markdown("## Chọn cổ phiếu để so sánh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stock1 = st.selectbox("Cổ phiếu 1:", all_stocks, index=0, key="stock1")
        
        with col2:
            # Ensure stock2 is different from stock1
            stock2_options = [s for s in all_stocks if s != stock1]
            stock2 = st.selectbox("Cổ phiếu 2:", stock2_options, index=0, key="stock2")
    
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {str(e)}")
        # Fallback
        all_stocks = ["VCB", "CTG", "BID", "ACB", "VIC", "FPT", "MSN", "VNM", "PLX", "TCB"]
        
        # Basic comparison interface
        st.markdown("## Chọn cổ phiếu để so sánh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stock1 = st.selectbox("Cổ phiếu 1:", all_stocks, index=0, key="stock1")
        
        with col2:
            stock2 = st.selectbox("Cổ phiếu 2:", all_stocks, index=1, key="stock2")
    
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
                from datetime import datetime, timedelta
                import pandas as pd
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                from src.data.data_provider import DataProvider
                from src.utils.config import ConfigManager
                
                # Simple config class for DataProvider
                class SimpleConfig:
                    CACHE_ENABLED = True
                    CACHE_DURATION = 300
                
                # Get data for both stocks
                config = ConfigManager()
                data_provider = DataProvider(SimpleConfig())
                
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
                
                st.plotly_chart(fig, width='stretch')
                
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
                
                st.dataframe(metrics_df, width='stretch')
                
                # Correlation analysis
                st.markdown("## 🔗 Phân tích tương quan")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Hệ số tương quan", f"{correlation:.3f}")
                
                with col2:
                    if correlation > 0.7:
                        correlation_desc = "Cao"
                        color = "🔴"
                    elif correlation > 0.3:
                        correlation_desc = "Trung bình"
                        color = "🟡"
                    else:
                        correlation_desc = "Thấp"
                        color = "🟢"
                    st.metric("Mức độ", f"{color} {correlation_desc}")
                
                with col3:
                    diversification = "Tốt" if correlation < 0.5 else "Kém"
                    st.metric("Đa dạng hóa", diversification)
                
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

# Main page function for st.Page
render_comparison_page()
