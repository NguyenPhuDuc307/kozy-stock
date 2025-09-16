"""
❓ HELP PAGE - Trang hướng dẫn sử dụng
===================================

Trang hướng dẫn và hỗ trợ người dùng
"""

import streamlit as st

def render_help_page():
    """
    Render trang trợ giúp
    """
    st.markdown("# ❓ Hướng dẫn sử dụng")
    
    # Navigation menu for help sections
    help_section = st.selectbox(
        "Chọn chủ đề:",
        [
            "🚀 Bắt đầu",
            "📈 Phân tích kỹ thuật", 
            "🔍 Quét thị trường",
            "📊 So sánh cổ phiếu",
            "🔄 Backtest chiến lược",
            "💡 Mẹo sử dụng",
            "🛠️ Khắc phục sự cố",
            "📞 Liên hệ hỗ trợ"
        ]
    )
    
    if help_section == "🚀 Bắt đầu":
        st.markdown("## 🚀 Bắt đầu với Kozy Stock")
        
        st.markdown("""
        ### Chào mừng bạn đến với Kozy Stock! 👋
        
        Kozy Stock là công cụ phân tích cổ phiếu Việt Nam toàn diện, giúp bạn:
        
        ✅ **Phân tích kỹ thuật chuyên sâu** với hơn 20 chỉ báo
        ✅ **Quét thị trường** tìm cơ hội đầu tư
        ✅ **So sánh hiệu suất** giữa các cổ phiếu
        ✅ **Backtest chiến lược** kiểm tra hiệu quả
        
        ### 🎯 Các tính năng chính:
        
        **1. 📈 Phân tích kỹ thuật**
        - Biểu đồ candlestick tương tác
        - 20+ chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands...)
        - Tín hiệu mua/bán tự động
        - Phân tích xu hướng và mô hình giá
        
        **2. 🔍 Quét thị trường**
        - Tìm cổ phiếu breakout
        - Lọc theo chỉ báo kỹ thuật
        - Xếp hạng theo momentum
        - Cảnh báo cơ hội đầu tư
        
        **3. 📊 So sánh cổ phiếu**
        - So sánh hiệu suất 2 cổ phiếu
        - Phân tích rủi ro/lợi nhuận
        - Tính toán correlation
        - Khuyến nghị đa dạng hóa
        
        **4. 🔄 Backtest chiến lược**
        - Kiểm tra hiệu quả chiến lược
        - Tính toán Sharpe ratio
        - Phân tích drawdown
        - So sánh với Buy & Hold
        """)
    
    elif help_section == "📈 Phân tích kỹ thuật":
        st.markdown("## 📈 Hướng dẫn phân tích kỹ thuật")
        
        st.markdown("""
        ### 🎯 Cách sử dụng trang phân tích:
        
        **1. Chọn cổ phiếu và khung thời gian**
        - Nhập mã cổ phiếu (VD: VCB, FPT, VIC...)
        - Chọn khung thời gian: 1D (ngày), 1H (giờ), 4H
        - Chọn khoảng thời gian hiển thị
        
        **2. Các chỉ báo kỹ thuật:**
        
        **📊 Chỉ báo xu hướng:**
        - **SMA/EMA**: Đường trung bình động - xác định xu hướng
        - **Bollinger Bands**: Dải giá dao động - tìm điểm vào/ra
        - **Parabolic SAR**: Điểm dừng và đảo chiều
        
        **📈 Chỉ báo momentum:**
        - **RSI (14)**: 0-100, <30 oversold, >70 overbought
        - **MACD**: Đường tín hiệu crossover
        - **Stochastic**: %K và %D crossover
        
        **📊 Chỉ báo khối lượng:**
        - **Volume**: Khối lượng giao dịch
        - **OBV**: On-Balance Volume - xác nhận xu hướng
        
        **3. Tín hiệu giao dịch:**
        
        **🟢 Tín hiệu MUA:**
        - RSI < 30 và tăng
        - MACD cắt lên signal line
        - Giá breakout khỏi Bollinger upper band
        - Golden Cross (MA ngắn cắt lên MA dài)
        
        **🔴 Tín hiệu BÁN:**
        - RSI > 70 và giảm  
        - MACD cắt xuống signal line
        - Giá breakdown khỏi Bollinger lower band
        - Death Cross (MA ngắn cắt xuống MA dài)
        
        **4. Cách đọc biểu đồ:**
        - **Nến xanh**: Giá đóng cửa > mở cửa (tăng)
        - **Nến đỏ**: Giá đóng cửa < mở cửa (giảm)
        - **Doji**: Mở cửa ≈ đóng cửa (phân vân)
        - **Hammer**: Thân ngắn, râu dưới dài (đảo chiều tăng)
        - **Shooting Star**: Thân ngắn, râu trên dài (đảo chiều giảm)
        """)
    
    elif help_section == "🔍 Quét thị trường":
        st.markdown("## 🔍 Hướng dẫn quét thị trường")
        
        st.markdown("""
        ### 🎯 Tính năng Market Scanner:
        
        **1. Mục đích:**
        - Tìm cổ phiếu có tín hiệu kỹ thuật tốt
        - Phát hiện breakout và momentum
        - Lọc cơ hội đầu tư tiềm năng
        
        **2. Các bộ lọc:**
        
        **📊 Lọc theo RSI:**
        - Oversold (RSI < 30): Cổ phiếu bị bán quá mức
        - Overbought (RSI > 70): Cổ phiếu mua quá mức
        - Neutral (30-70): Vùng cân bằng
        
        **📈 Lọc theo MACD:**
        - Bullish: MACD > Signal line (xu hướng tăng)
        - Bearish: MACD < Signal line (xu hướng giảm)
        - Crossover: Tín hiệu giao cắt mới
        
        **📊 Lọc theo Bollinger Bands:**
        - Upper breakout: Giá vượt dải trên
        - Lower breakdown: Giá xuống dưới dải dưới
        - Squeeze: Dải co hẹp (chuẩn bị bùng nổ)
        
        **🔥 Lọc theo Volume:**
        - High volume: Khối lượng cao bất thường
        - Volume spike: Tăng đột biến khối lượng
        
        **3. Cách sử dụng:**
        - Chọn tiêu chí lọc phù hợp
        - Xem danh sách cổ phiếu phù hợp
        - Click vào cổ phiếu để phân tích chi tiết
        - Kiểm tra tín hiệu bằng nhiều chỉ báo
        
        **4. Mẹo sử dụng hiệu quả:**
        - Kết hợp nhiều bộ lọc
        - Ưu tiên cổ phiếu có volume cao
        - Kiểm tra news trước khi vào lệnh
        - Đặt stop-loss bảo vệ vốn
        """)
    
    elif help_section == "📊 So sánh cổ phiếu":
        st.markdown("## 📊 Hướng dẫn so sánh cổ phiếu")
        
        st.markdown("""
        ### 🎯 Tính năng so sánh:
        
        **1. Mục đích:**
        - So sánh hiệu suất 2 cổ phiếu
        - Đánh giá rủi ro/lợi nhuận
        - Quyết định đa dạng hóa danh mục
        
        **2. Các chỉ số so sánh:**
        
        **📈 Hiệu suất:**
        - **Tổng lợi nhuận**: % thay đổi giá trong kỳ
        - **Volatility**: Độ biến động giá (rủi ro)
        - **Sharpe Ratio**: Lợi nhuận/rủi ro
        - **Max Drawdown**: Thua lỗ tối đa
        
        **🔗 Correlation:**
        - **> 0.7**: Tương quan cao (không nên kết hợp)
        - **0.3-0.7**: Tương quan trung bình
        - **< 0.3**: Tương quan thấp (tốt cho đa dạng hóa)
        
        **3. Cách đọc kết quả:**
        
        **🏆 Winner Analysis:**
        - Cổ phiếu có return cao hơn
        - Cổ phiếu có Sharpe ratio tốt hơn
        
        **💎 Risk-Adjusted Performance:**
        - Xem xét cả lợi nhuận và rủi ro
        - Ưu tiên Sharpe ratio cao
        
        **🔄 Diversification:**
        - Correlation thấp = đa dạng hóa tốt
        - Giảm rủi ro tổng thể danh mục
        
        **4. Khuyến nghị:**
        - **Correlation < 0.3**: ✅ Kết hợp tốt
        - **Correlation 0.3-0.7**: ⚠️ Cân nhắc tỷ trọng  
        - **Correlation > 0.7**: ❌ Không nên kết hợp
        """)
    
    elif help_section == "🔄 Backtest chiến lược":
        st.markdown("## 🔄 Hướng dẫn Backtest")
        
        st.markdown("""
        ### 🎯 Backtest là gì?
        
        Backtest là quá trình kiểm tra hiệu quả của chiến lược trading bằng dữ liệu lịch sử.
        
        **1. Các chiến lược có sẵn:**
        
        **📈 Golden Cross:**
        - MA ngắn hạn cắt lên MA dài hạn → Mua
        - MA ngắn hạn cắt xuống MA dài hạn → Bán
        
        **📊 RSI Oversold/Overbought:**
        - RSI < 30 → Mua (oversold)
        - RSI > 70 → Bán (overbought)
        
        **📈 MACD Signal:**
        - MACD cắt lên Signal line → Mua
        - MACD cắt xuống Signal line → Bán
        
        **📊 Bollinger Bands:**
        - Giá chạm dải dưới → Mua
        - Giá chạm dải trên → Bán
        
        **2. Tham số quản lý rủi ro:**
        
        **💰 Vốn ban đầu:**
        - Số tiền khởi điểm cho backtest
        
        **📊 Position Size:**
        - % vốn sử dụng cho mỗi lệnh
        - Khuyến nghị: 5-20%
        
        **🛡️ Stop Loss:**
        - % cắt lỗ tự động
        - Khuyến nghị: 3-10%
        
        **💸 Commission:**
        - Phí giao dịch (thường 0.15%)
        
        **3. Các chỉ số đánh giá:**
        
        **📈 Total Return:**
        - Lợi nhuận tổng cộng (%)
        
        **📊 Sharpe Ratio:**
        - Lợi nhuận điều chỉnh rủi ro
        - > 1: Tốt, > 2: Rất tốt
        
        **📉 Max Drawdown:**
        - Thua lỗ tối đa liên tiếp (%)
        - Càng thấp càng tốt
        
        **🎯 Win Rate:**
        - Tỷ lệ lệnh thắng (%)
        - > 60%: Tốt
        
        **📊 Alpha:**
        - Hiệu suất so với Buy & Hold
        - Dương: Chiến lược tốt hơn
        
        **4. Cách đọc kết quả:**
        
        **✅ Chiến lược tốt:**
        - Total Return > Buy & Hold
        - Sharpe Ratio > 1
        - Max Drawdown < 20%
        - Win Rate > 50%
        
        **⚠️ Cần cải thiện:**
        - Alpha âm
        - Sharpe Ratio < 0.5
        - Max Drawdown > 30%
        - Win Rate < 40%
        """)
    
    elif help_section == "💡 Mẹo sử dụng":
        st.markdown("## 💡 Mẹo sử dụng hiệu quả")
        
        st.markdown("""
        ### 🎯 Mẹo phân tích:
        
        **1. Kết hợp nhiều chỉ báo:**
        - Không dựa vào 1 chỉ báo duy nhất
        - RSI + MACD + Volume = tín hiệu mạnh
        - Xác nhận bằng price action
        
        **2. Chọn khung thời gian phù hợp:**
        - **Day trading**: 1H, 4H
        - **Swing trading**: 1D
        - **Long-term**: 1W, 1M
        
        **3. Quản lý rủi ro:**
        - Luôn đặt stop-loss
        - Không all-in 1 cổ phiếu
        - Đa dạng hóa danh mục
        
        ### 🔍 Mẹo quét thị trường:
        
        **1. Thời điểm quét tốt nhất:**
        - Sau giờ đóng cửa (15h)
        - Trước giờ mở cửa (8h30)
        
        **2. Ưu tiên cổ phiếu:**
        - Volume cao bất thường
        - Có tin tức tích cực
        - Thuộc ngành hot
        
        ### 📊 Mẹo so sánh cổ phiếu:
        
        **1. So sánh cùng ngành:**
        - Banking: VCB vs CTG vs BID
        - Tech: FPT vs CMG vs ELC
        
        **2. Đa dạng hóa ngành:**
        - Kết hợp các ngành khác nhau
        - Correlation thấp = tốt
        
        ### 🔄 Mẹo backtest:
        
        **1. Test nhiều khoảng thời gian:**
        - Bull market vs Bear market
        - Kết quả ổn định qua nhiều năm
        
        **2. Tối ưu tham số:**
        - Thử nhiều setting khác nhau
        - Tránh over-fitting
        
        **3. Luôn có plan B:**
        - Chiến lược dự phòng
        - Điều chỉnh khi thất bại
        """)
    
    elif help_section == "🛠️ Khắc phục sự cố":
        st.markdown("## 🛠️ Khắc phục sự cố thường gặp")
        
        st.markdown("""
        ### ❌ Lỗi thường gặp:
        
        **1. "Không thể tải dữ liệu"**
        
        **Nguyên nhân:**
        - Mã cổ phiếu không đúng
        - Kết nối internet yếu
        - Server bảo trì
        
        **Giải pháp:**
        - ✅ Kiểm tra mã cổ phiếu (VD: VCB, không phải vcb)
        - ✅ Refresh trang web
        - ✅ Thử lại sau 5-10 phút
        - ✅ Chọn khoảng thời gian ngắn hơn
        
        **2. "Biểu đồ không hiển thị"**
        
        **Nguyên nhân:**
        - Browser không hỗ trợ
        - Cache browser lỗi
        - JavaScript bị block
        
        **Giải pháp:**
        - ✅ Dùng Chrome, Firefox, Safari mới nhất
        - ✅ Xóa cache browser (Ctrl+F5)
        - ✅ Tắt ad blocker
        - ✅ Enable JavaScript
        
        **3. "Tính toán chỉ báo lỗi"**
        
        **Nguyên nhân:**
        - Dữ liệu không đủ
        - Tham số không hợp lệ
        
        **Giải pháp:**
        - ✅ Chọn khoảng thời gian dài hơn
        - ✅ Kiểm tra tham số chỉ báo
        - ✅ Reset về mặc định
        
        **4. "App chạy chậm"**
        
        **Nguyên nhân:**
        - Quá nhiều tab mở
        - RAM không đủ
        - Kết nối mạng chậm
        
        **Giải pháp:**
        - ✅ Đóng các tab không cần thiết
        - ✅ Restart browser
        - ✅ Chọn ít chỉ báo hơn
        - ✅ Giảm khoảng thời gian hiển thị
        
        ### 🔧 Tối ưu hiệu suất:
        
        **1. Browser khuyến nghị:**
        - Chrome 90+
        - Firefox 88+
        - Safari 14+
        - Edge 90+
        
        **2. Cấu hình tối thiểu:**
        - RAM: 4GB
        - CPU: Dual-core
        - Internet: 10Mbps
        
        **3. Mẹo tăng tốc:**
        - Đóng app khác
        - Dùng window nhỏ
        - Chọn ít chỉ báo
        """)
    
    elif help_section == "📞 Liên hệ hỗ trợ":
        st.markdown("## 📞 Liên hệ & Hỗ trợ")
        
        st.markdown("""
        ### 🤝 Cần hỗ trợ?
        
        **1. 💬 Cộng đồng:**
        - Discord: [Kozy Stock Community]
        - Telegram: [@KozyStock]
        - Facebook Group: [Kozy Stock Users]
        
        **2. 📧 Email hỗ trợ:**
        - support@kozystock.com
        - Thời gian phản hồi: 24-48h
        
        **3. 📱 Hotline:**
        - 📞 1900-xxxx (8h-17h, T2-T6)
        
        **4. 📖 Tài liệu:**
        - Wiki: [docs.kozystock.com]
        - Video tutorials: [YouTube Channel]
        - Blog: [blog.kozystock.com]
        
        ### 🐛 Báo lỗi:
        
        **Khi báo lỗi, vui lòng cung cấp:**
        - Browser và version
        - Mô tả chi tiết lỗi
        - Screenshot nếu có
        - Các bước tái hiện lỗi
        
        ### 💡 Góp ý tính năng:
        
        **Chúng tôi luôn lắng nghe:**
        - Tính năng mới
        - Cải thiện giao diện
        - Thêm chỉ báo
        - Tối ưu hiệu suất
        
        ### 🔄 Cập nhật:
        
        **Theo dõi phiên bản mới:**
        - Newsletter: [Đăng ký nhận tin]
        - Release notes: [Github releases]
        - Social media: [@KozyStock]
        
        ### ⭐ Đánh giá:
        
        **Nếu hài lòng, hãy:**
        - ⭐ Rate 5 sao
        - 📝 Viết review
        - 🔄 Share cho bạn bè
        - 💝 Donate ủng hộ
        """)
    
    # Quick links
    st.markdown("---")
    st.markdown("## 🔗 Liên kết nhanh")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Phân tích ngay", type="primary"):
            st.switch_page("stock_analysis")
    
    with col2:
        if st.button("🔍 Quét thị trường"):
            st.switch_page("market_scanner")
    
    with col3:
        if st.button("📊 So sánh CP"):
            st.switch_page("stock_comparison")
    
    with col4:
        if st.button("🔄 Backtest"):
            st.switch_page("backtest")
    
    # Version info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    📦 Kozy Stock v2.0.0 | 🐍 Python 3.13 | 🚀 Streamlit 1.49<br>
    Made with ❤️ for Vietnamese investors
    </div>
    """, unsafe_allow_html=True)

# Main page function for st.Page
render_help_page()
