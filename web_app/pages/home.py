"""
🏠 HOME PAGE - Trang chủ
========================

Trang hiển thị tổng quan thị trường với biểu đồ nến các sàn
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_home_page():
    """
    Render trang chủ với tổng quan thị trường
    """
    import pandas as pd  # Import pandas trong function để tránh lỗi
    
    st.markdown("# 🏠 Tổng quan thị trường")
    
    try:
        # Import các thư viện cần thiết
        from datetime import datetime, timedelta
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Danh sách các chỉ số chính
        indices = {
            "VNINDEX": {"name": "VN-Index (HOSE)", "color": "#1f77b4", "symbol": "VNINDEX", "source": "VCI"},
            "VN30": {"name": "VN30", "color": "#ff7f0e", "symbol": "VN30", "source": "VCI"},
            "HNXINDEX": {"name": "HNX-Index", "color": "#2ca02c", "symbol": "HNXINDEX", "source": "VCI"}, 
            "HNX30": {"name": "HNX30", "color": "#d62728", "symbol": "HNX30", "source": "VCI"},
            # Tạm thời comment UPCOM do symbol không hỗ trợ
            # "UPCOM": {"name": "UPCOM-Index", "color": "#9467bd", "symbol": "UPCOM", "source": "VCI"}
        }
        
        # Lấy thời gian (30 ngày gần nhất)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Hiển thị trạng thái các sàn
        st.markdown("### 📈 Trạng thái các sàn giao dịch")
        
        # Container cho metrics
        cols = st.columns(len(indices))
        
        # Lấy dữ liệu cho từng chỉ số
        indices_data = {}
        for i, (key, info) in enumerate(indices.items()):
            try:
                with st.spinner(f"Đang tải dữ liệu {info['name']}..."):
                    # Sử dụng vnstock API mới
                    from vnstock import Vnstock
                    
                    # Khởi tạo vnstock với source từ config
                    stock = Vnstock().stock(symbol=info['symbol'], source=info.get('source', 'VCI'))
                    
                    # Lấy dữ liệu lịch sử
                    df = stock.quote.history(
                        start=start_str,
                        end=end_str,
                        interval='1D'
                    )
                    
                    if df is not None and not df.empty:
                        # Sắp xếp theo thời gian
                        df = df.sort_index()
                        
                        # Lấy giá trị mới nhất
                        latest = df.iloc[-1]
                        previous = df.iloc[-2] if len(df) > 1 else latest
                        
                        # Tính toán thay đổi
                        change = latest['close'] - previous['close']
                        change_pct = (change / previous['close']) * 100
                        
                        # Hiển thị metric
                        with cols[i]:
                            st.metric(
                                label=info['name'],
                                value=f"{latest['close']:,.2f}",
                                delta=f"{change_pct:+.2f}%"
                            )
                        
                        # Lưu dữ liệu để vẽ biểu đồ
                        indices_data[key] = {
                            'data': df,
                            'info': info,
                            'latest': latest,
                            'change_pct': change_pct
                        }
                    else:
                        with cols[i]:
                            st.error(f"❌ Không có dữ liệu")
                            st.caption(info['name'])
                    
            except Exception as e:
                with cols[i]:
                    st.error(f"❌ Lỗi {info['name']}")
                    st.caption(f"Symbol: {info['symbol']}")
                
                # Log lỗi chi tiết
                error_msg = str(e)
                if "Invalid derivative or bond symbol" in error_msg:
                    st.warning(f"⚠️ {info['name']}: Symbol '{info['symbol']}' không hợp lệ cho vnstock API")
                else:
                    st.error(f"Lỗi tải dữ liệu {info['name']}: {error_msg}")
        
        # Hiển thị biểu đồ nến
        if indices_data:
            st.markdown("### 📈 Biểu đồ nến các sàn giao dịch")
            
            # Tabs cho từng sàn
            tab_names = [data['info']['name'] for data in indices_data.values()]
            tabs = st.tabs(tab_names)
            
            for tab, (key, data) in zip(tabs, indices_data.items()):
                with tab:
                    df = data['data']
                    info = data['info']
                    
                    # Tạo biểu đồ nến
                    fig = go.Figure()
                    
                    # Thêm biểu đồ nến
                    fig.add_trace(go.Candlestick(
                        x=df.index,  # Sử dụng index thay vì 'time'
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=info['name'],
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ))
                    
                    # Thêm đường MA20
                    if len(df) >= 20:
                        df['ma20'] = df['close'].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(
                            x=df.index,  # Sử dụng index thay vì 'time'
                            y=df['ma20'],
                            mode='lines',
                            name='MA20',
                            line=dict(color='orange', width=1.5)
                        ))
                    
                    # Cập nhật layout
                    fig.update_layout(
                        title=f"Biểu đồ nến {info['name']} - 30 ngày gần nhất",
                        yaxis_title="Giá",
                        xaxis_title="Thời gian",
                        template='plotly_white',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Loại bỏ rangeslider
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    
                    # Hiển thị biểu đồ
                    st.plotly_chart(fig, width='stretch')
                    
                    # Thông tin chi tiết
                    col1, col2, col3, col4 = st.columns(4)
                    latest = data['latest']
                    
                    with col1:
                        st.metric("Mở cửa", f"{latest['open']:,.2f}")
                    with col2:
                        st.metric("Cao nhất", f"{latest['high']:,.2f}")
                    with col3:
                        st.metric("Thấp nhất", f"{latest['low']:,.2f}")
                    with col4:
                        st.metric("Khối lượng", f"{latest['volume']:,.0f}")
        
        # Phần thống kê tổng quan
        st.markdown("---")
        st.markdown("### 📊 Thống kê tổng quan")
        
        if indices_data:
            # Bảng tóm tắt
            summary_data = []
            for key, data in indices_data.items():
                latest = data['latest']
                info = data['info']
                change_pct = data['change_pct']
                
                summary_data.append({
                    "Sàn": info['name'],
                    "Giá đóng cửa": f"{latest['close']:,.2f}",
                    "Thay đổi (%)": f"{change_pct:+.2f}%",
                    "Khối lượng": f"{latest['volume']:,.0f}",
                    "Cao nhất": f"{latest['high']:,.2f}",
                    "Thấp nhất": f"{latest['low']:,.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Định dạng bảng với màu sắc
            def color_change(val):
                if "+" in val:
                    return 'color: green'
                elif "-" in val:
                    return 'color: red'
                return 'color: black'
            
            styled_df = summary_df.style.map(color_change, subset=['Thay đổi (%)'])
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
    except ImportError:
        st.error("❌ Không thể import vnstock. Vui lòng cài đặt: `pip install vnstock`")
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        
        # Hiển thị dữ liệu mẫu nếu không kết nối được
        st.warning("⚠️ Hiển thị dữ liệu mẫu do không thể kết nối")
        
        # Dữ liệu mẫu
        sample_data = {
            "Sàn": ["VN-Index (HOSE)", "VN30", "HNX-Index", "HNX30", "UPCOM-Index"],
            "Giá đóng cửa": ["1,240.50", "1,285.20", "230.15", "450.80", "85.25"],
            "Thay đổi (%)": ["+0.85%", "+1.20%", "-0.45%", "+0.65%", "-0.12%"],
            "Khối lượng": ["485,250,000", "125,180,000", "45,680,000", "25,450,000", "12,350,000"]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, width='stretch', hide_index=True)

# Main page function for st.Page  
render_home_page()
