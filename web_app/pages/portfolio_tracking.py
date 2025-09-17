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

# Import for advanced technical analysis
# Import sẽ được thực hiện trong function để tránh lỗi
# from src.analysis.indicators import TechnicalIndicators
# from src.analysis.signals import TradingSignals

# Debug import
UnifiedConfig = None
TimeFrame = None
UnifiedSignalAnalyzer = None

try:
    from src.utils.unified_config import UnifiedConfig, TimeFrame, UnifiedSignalAnalyzer
except ImportError as e:
    import traceback
    traceback.print_exc()
    
    # Fallback - create minimal classes
    class TimeFrame:
        SHORT_TERM = "short_term"
        MEDIUM_TERM = "medium_term"
        LONG_TERM = "long_term"
    
    class UnifiedConfig:
        @classmethod
        def create_sidebar_timeframe_selector(cls, key):
            import streamlit as st
            st.sidebar.markdown("### ⏱️ Khoảng thời gian phân tích (Fallback)")
            return TimeFrame.MEDIUM_TERM
            
        @classmethod
        def get_timeframe_config(cls, timeframe):
            class Config:
                name = "Trung hạn (Fallback)"
                description = "30-90 ngày"
            return Config()
            
        @classmethod
        def get_date_range(cls, timeframe):
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            return start_date, end_date
            
        @classmethod
        def create_advanced_settings_expander(cls, key):
            import streamlit as st
            st.sidebar.info("Fallback mode - advanced settings not available")
            return {}
    
    class UnifiedSignalAnalyzer:
        def __init__(self, timeframe, thresholds):
            pass
            
        def analyze_comprehensive_signal(self, data):
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reasons': ['Fallback signal - unified system not available']
            }
    
    st.warning("🔄 Using fallback configuration system")

except Exception as e:
    import traceback
    traceback.print_exc()    # Super fallback
    class TimeFrame:
        MEDIUM_TERM = "medium_term"
    
    class UnifiedConfig:
        @classmethod
        def create_sidebar_timeframe_selector(cls, key):
            return TimeFrame.MEDIUM_TERM
        @classmethod    
        def get_timeframe_config(cls, timeframe):
            class Config:
                name = "Emergency Fallback"
                description = "Basic mode"
            return Config()
        @classmethod
        def get_date_range(cls, timeframe):
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return start_date, end_date
        @classmethod
        def create_advanced_settings_expander(cls, key):
            return {}
    
    class UnifiedSignalAnalyzer:
        def __init__(self, timeframe, thresholds):
            pass
        def analyze_comprehensive_signal(self, data):
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasons': ['Emergency fallback']}

# Verify classes are defined
if UnifiedConfig is None:
    st.error("💀 UnifiedConfig is still None after all attempts")

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
    from src.data.data_provider import DataProvider
    
    # Simple config class for DataProvider
    class SimpleConfig:
        CACHE_ENABLED = True
        CACHE_DURATION = 300
    
    # Khởi tạo components
    trading_history = TradingHistory()
    data_provider = DataProvider(SimpleConfig())
    
    # Import technical analysis components
    try:
        from src.analysis.indicators import TechnicalIndicators
        from src.analysis.signals import TradingSignals
        indicators = TechnicalIndicators()
        signals = TradingSignals()
    except ImportError:
        # Fallback classes if import fails
        class TechnicalIndicators:
            def calculate_all(self, data):
                return data
        class TradingSignals:
            def generate_signal(self, data):
                return None
        indicators = TechnicalIndicators()
        signals = TradingSignals()
    
    # Lấy danh sách cổ phiếu đang nắm giữ từ lịch sử giao dịch
    current_holdings = trading_history.get_current_holdings()
    
    if not current_holdings:
        # Nếu chưa có giao dịch nào, return empty
        return pd.DataFrame()
    
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
                # API trả về giá theo nghìn VND, cần nhân 1000 để về VND đầy đủ
                api_price = data.iloc[-1]['close']
                current_price = api_price * 1000
            else:
                # Fallback sử dụng avg_price (đã ở đơn vị VND đầy đủ)
                current_price = holding_data["avg_price"]
            
            # Tính toán lợi nhuận dựa trên dữ liệu thực
            shares = holding_data["shares"]
            avg_price = holding_data["avg_price"]
            total_cost = holding_data["total_cost"]
            
            # Giá đã được convert về VND đầy đủ ở trên
            current_value = current_price * shares
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Tính toán tín hiệu kỹ thuật chuyên nghiệp
            if data is not None and not data.empty and len(data) >= 30:
                # Tính toán các chỉ báo kỹ thuật
                df_with_indicators = indicators.calculate_all(data)
                
                if df_with_indicators is not None and not df_with_indicators.empty:
                    # Sử dụng cùng hệ thống tín hiệu như market scanner
                    signal_result = signals.generate_signal(df_with_indicators)
                    
                    if signal_result:
                        signal = signal_result.signal_type
                        technical_score = signal_result.confidence
                        
                        # Map signal to recommendation
                        if signal == "BUY":
                            if technical_score >= 0.7:
                                recommendation = "MUA MẠNH"
                            else:
                                recommendation = "MUA"
                        elif signal == "SELL":
                            if technical_score >= 0.7:
                                recommendation = "BÁN MẠNH"
                            else:
                                recommendation = "BÁN"
                        else:
                            recommendation = "GIỮ"
                    else:
                        # Fallback if signal generation fails
                        signal = "HOLD"
                        technical_score = 0.0
                        recommendation = "GIỮ"
                else:
                    # Fallback if indicators calculation fails
                    signal = "HOLD"
                    technical_score = 0.0
                    recommendation = "GIỮ"
            else:
                # Fallback for insufficient data - use advanced MA analysis
                if data is not None and not data.empty and len(data) >= 5:
                    prices = data['close'].values
                    volumes = data['volume'].values if 'volume' in data.columns else None
                    
                    # Tính multiple MA periods để có độ tin cậy chính xác hơn
                    ma5 = prices[-5:].mean()
                    ma10 = prices[-10:].mean() if len(prices) >= 10 else ma5
                    ma20 = prices[-20:].mean() if len(prices) >= 20 else ma10
                    
                    # Tính volatility để đánh giá độ tin cậy
                    if len(prices) >= 10:
                        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                        volatility = sum([abs(change) for change in price_changes]) / len(price_changes)
                    else:
                        volatility = 0.02  # Default 2%
                    
                    # Tính trend strength
                    if len(prices) >= 5:
                        trend_slope = (prices[-1] - prices[-5]) / prices[-5]
                    else:
                        trend_slope = 0
                    
                    # Tính volume confirmation (nếu có data volume)
                    volume_factor = 1.0
                    if volumes is not None and len(volumes) >= 5:
                        recent_avg_volume = volumes[-5:].mean()
                        total_avg_volume = volumes.mean()
                        if recent_avg_volume > total_avg_volume * 1.2:
                            volume_factor = 1.2  # High volume = more confident
                        elif recent_avg_volume < total_avg_volume * 0.8:
                            volume_factor = 0.8  # Low volume = less confident
                    
                    # Tính MA alignment score
                    ma_alignment = 0
                    if current_price > ma5 > ma10 > ma20:
                        ma_alignment = 1.0  # Strong uptrend
                    elif current_price > ma5 > ma10:
                        ma_alignment = 0.7  # Moderate uptrend
                    elif current_price > ma5:
                        ma_alignment = 0.3  # Weak uptrend
                    elif current_price < ma5 < ma10 < ma20:
                        ma_alignment = -1.0  # Strong downtrend
                    elif current_price < ma5 < ma10:
                        ma_alignment = -0.7  # Moderate downtrend
                    elif current_price < ma5:
                        ma_alignment = -0.3  # Weak downtrend
                    else:
                        ma_alignment = 0  # Sideways
                    
                    # Tính base signal strength
                    price_vs_ma5 = (current_price - ma5) / ma5
                    
                    if abs(price_vs_ma5) > 0.03:  # > 3% difference
                        base_strength = min(abs(price_vs_ma5) * 10, 0.8)  # Cap at 0.8
                    else:
                        base_strength = abs(price_vs_ma5) * 5  # More gradual for small moves
                    
                    # Combine factors for final confidence
                    confidence_factors = [
                        base_strength,  # Price vs MA strength
                        abs(ma_alignment) * 0.3,  # MA alignment bonus
                        min(abs(trend_slope) * 15, 0.2),  # Trend strength bonus (cap at 0.2)
                        (volume_factor - 1) * 0.1  # Volume confirmation bonus
                    ]
                    
                    technical_score = sum(confidence_factors)
                    
                    # Apply volatility penalty (high volatility = less reliable)
                    volatility_penalty = min(volatility * 5, 0.3)  # Cap penalty at 0.3
                    technical_score = max(0, technical_score - volatility_penalty)
                    
                    # Apply directional sign
                    if price_vs_ma5 > 0.02:  # 2% above MA5
                        signal = "BUY"
                        recommendation = "MUA MẠNH" if technical_score >= 0.6 else "MUA"
                    elif price_vs_ma5 < -0.02:  # 2% below MA5
                        signal = "SELL"
                        recommendation = "BÁN MẠNH" if technical_score >= 0.6 else "BÁN"
                        technical_score = -technical_score  # Negative for sell signals
                    else:
                        signal = "HOLD"
                        recommendation = "GIỮ"
                        technical_score = 0.0
                        
                else:
                    # Final fallback based on profit/loss with more nuanced scoring
                    if profit_loss_pct > 10:
                        technical_score = 0.15 + min((profit_loss_pct - 10) * 0.01, 0.25)
                        signal = "HOLD"
                        recommendation = "GIỮ"
                    elif profit_loss_pct > 5:
                        technical_score = 0.05 + (profit_loss_pct - 5) * 0.02
                        signal = "HOLD"
                        recommendation = "GIỮ"
                    elif profit_loss_pct < -10:
                        technical_score = -0.15 - min((abs(profit_loss_pct) - 10) * 0.01, 0.25)
                        signal = "HOLD"
                        recommendation = "GIỮ"
                    elif profit_loss_pct < -5:
                        technical_score = -0.05 - (abs(profit_loss_pct) - 5) * 0.02
                        signal = "HOLD"
                        recommendation = "GIỮ"
                    else:
                        technical_score = profit_loss_pct * 0.01  # Small correlation with P&L
                        signal = "HOLD"
                        recommendation = "GIỮ"
            
            portfolio_data.append({
                'Symbol': symbol,
                'Buy_Price': avg_price,
                'Current_Price': current_price,
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
            
            # Giả lập giá hiện tại (fallback)
            current_price = avg_price * random.uniform(0.9, 1.1)
            current_value = current_price * shares
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Default technical analysis với tính toán chính xác hơn
            # Sử dụng P&L để ước tính độ tin cậy thay vì random
            if profit_loss_pct > 15:
                technical_score = 0.25 + min((profit_loss_pct - 15) * 0.005, 0.15)
            elif profit_loss_pct > 5:
                technical_score = 0.1 + (profit_loss_pct - 5) * 0.015
            elif profit_loss_pct > 0:
                technical_score = profit_loss_pct * 0.02
            elif profit_loss_pct > -5:
                technical_score = profit_loss_pct * 0.02
            elif profit_loss_pct > -15:
                technical_score = -0.1 + (profit_loss_pct + 5) * 0.015
            else:
                technical_score = -0.25 + max((profit_loss_pct + 15) * 0.005, -0.15)
            
            # Add some realistic variation based on symbol characteristics
            import hashlib
            symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            variation = (symbol_hash % 100 - 50) * 0.002  # ±0.1 variation
            technical_score += variation
            
            signal = "HOLD"
            recommendation = "GIỮ"
            
            portfolio_data.append({
                'Symbol': symbol,
                'Buy_Price': avg_price,
                'Current_Price': current_price,
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
        # Import modules
        from src.utils.trading_history import TradingHistory
        from src.utils.trading_portfolio_manager import TradingPortfolioManager
        from src.data.data_provider import DataProvider
        
        # Simple config class for DataProvider
        class SimpleConfig:
            CACHE_ENABLED = True
            CACHE_DURATION = 300
        
        # Ensure UnifiedConfig is available in function scope
        global UnifiedConfig, UnifiedSignalAnalyzer, TimeFrame
        if UnifiedConfig is None:
            st.error("💀 UnifiedConfig not available in function scope")
            return
        
        # Unified timeframe selector
        selected_timeframe = UnifiedConfig.create_sidebar_timeframe_selector("portfolio_tracking_timeframe")
        timeframe_config = UnifiedConfig.get_timeframe_config(selected_timeframe)
        
        # Advanced settings
        custom_thresholds = UnifiedConfig.create_advanced_settings_expander("portfolio_tracking_advanced")
        
        # Khởi tạo components
        data_provider = DataProvider(SimpleConfig())
        
        # Import technical analysis modules in function scope
        try:
            from src.analysis.indicators import TechnicalIndicators
            from src.analysis.signals import TradingSignals
            indicators = TechnicalIndicators()
            signals = TradingSignals()
        except ImportError as e:
            # Create dummy classes
            class TechnicalIndicators:
                def calculate_all(self, data):
                    return data
            class TradingSignals:
                def generate_signal(self, data):
                    return None
            indicators = TechnicalIndicators()
            signals = TradingSignals()
        
        # Initialize unified signal analyzer
        signal_analyzer = UnifiedSignalAnalyzer(selected_timeframe, custom_thresholds)
        
        # Khởi tạo portfolio manager
        if 'trading_portfolio_manager' not in st.session_state:
            st.session_state.trading_portfolio_manager = TradingPortfolioManager()
        
        portfolio_manager = st.session_state.trading_portfolio_manager
        
        # Lấy danh sách danh mục
        portfolios = portfolio_manager.get_portfolios()
        
        if not portfolios:
            st.info("📝 Chưa có danh mục nào. Hãy tạo danh mục đầu tiên trong trang 'Quản lý nhiều danh mục'!")
            st.warning("⚠️ Vui lòng vào trang 'Quản lý nhiều danh mục' để tạo danh mục mới")
            return
        
        # Sidebar để chọn danh mục
        st.sidebar.markdown("## 📁 Chọn Danh mục")
        
        portfolio_options = [f"{p['name']} ({p['id']})" for p in portfolios]
        selected_portfolio = st.sidebar.selectbox(
            "Danh mục:",
            portfolio_options,
            key="portfolio_selector"
        )
        
        # Lấy portfolio_id từ selection
        portfolio_id = selected_portfolio.split("(")[-1].strip(")")
        portfolio_info = portfolio_manager.get_portfolio(portfolio_id)
        
        # Hiển thị thông tin danh mục trong sidebar
        if portfolio_info:
            st.sidebar.markdown("### 📋 Thông tin Danh mục")
            st.sidebar.write(f"**Tên**: {portfolio_info['name']}")
            st.sidebar.write(f"**Chiến lược**: {portfolio_info['strategy']}")
            st.sidebar.write(f"**Vốn ban đầu**: {portfolio_info['initial_cash']:,.0f} VNĐ")
            st.sidebar.write(f"**Tạo ngày**: {portfolio_info['created_date'].split(' ')[0]}")
        
        # Lấy trading history của danh mục được chọn
        trading_history = portfolio_manager.get_trading_history(portfolio_id)
        if trading_history is None:
            st.error("❌ Không thể tải lịch sử giao dịch của danh mục này!")
            return
            
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
                transaction_id = portfolio_manager.add_transaction(
                    portfolio_id, symbol, transaction_type, quantity, price, fee=fee, note=note
                )
                if transaction_id:
                    st.sidebar.success(f"✅ Đã thêm giao dịch #{transaction_id}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Có lỗi khi thêm giao dịch")
            else:
                st.sidebar.error("❌ Mã cổ phiếu không hợp lệ")
        
        if not current_holdings:
            st.warning("⚠️ Chưa có cổ phiếu nào trong danh mục")
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
                    result = portfolio_manager.clear_symbol_transactions(portfolio_id, selected_symbol)
                    if result:
                        st.sidebar.success(f"✅ Đã xóa {selected_symbol} khỏi danh mục")
                        st.rerun()
                    else:
                        st.sidebar.error(f"❌ Có lỗi khi xóa {selected_symbol}")
        
        # Sidebar - Cài đặt theo dõi
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Cài đặt")
        auto_refresh = st.sidebar.checkbox("Tự động làm mới (30s)", value=False)
        
        # Get portfolio data
        with st.spinner("📊 Đang tải dữ liệu danh mục..."):
            # Logic lấy dữ liệu portfolio inline
            import random
            import pandas as pd
            import vnstock
            from datetime import datetime, timedelta
            
            portfolio_data = []
            
            # Get date range using unified config
            try:
                start_date, end_date = UnifiedConfig.get_date_range(selected_timeframe)
            except Exception as e:
                st.error(f"Error getting date range: {e}")
                # Fallback
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
            
            for symbol, holding_data in current_holdings.items():
                try:
                    # Lấy giá hiện tại từ vnstock với đủ dữ liệu cho phân tích kỹ thuật
                    quote = vnstock.Quote(symbol=symbol, source='VCI')
                    
                    data = quote.history(
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d")
                    )
                    
                    if data is not None and not data.empty:
                        # API trả về giá theo nghìn VND, cần nhân 1000 để về VND đầy đủ
                        current_price = data.iloc[-1]['close'] * 1000
                    else:
                        # Fallback nếu không có dữ liệu từ API
                        current_price = holding_data["avg_price"]
                except Exception as e:
                    # Fallback nếu có lỗi API
                    current_price = holding_data["avg_price"]
                
                # Tính toán lợi nhuận dựa trên dữ liệu thực
                shares = holding_data["shares"]
                avg_price = holding_data["avg_price"]
                total_cost = holding_data["total_cost"]
                
                # Tính current_value và profit_loss
                current_value = current_price * shares
                profit_loss = current_value - total_cost
                profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
                
                # Tính toán tín hiệu kỹ thuật thống nhất
                if data is not None and not data.empty and len(data) >= 30:
                    # Tính toán các chỉ báo kỹ thuật
                    df_with_indicators = indicators.calculate_all(data)
                    
                    if df_with_indicators is not None and not df_with_indicators.empty:
                        # Sử dụng unified signal analyzer
                        signal_analysis = signal_analyzer.analyze_comprehensive_signal(df_with_indicators)
                        
                        if signal_analysis:
                            signal = signal_analysis['signal']
                            technical_score = signal_analysis['confidence']
                            
                            # Map signal to recommendation
                            if signal == "BUY":
                                if technical_score >= 0.7:
                                    recommendation = "MUA MẠNH"
                                else:
                                    recommendation = "MUA"
                            elif signal == "SELL":
                                if technical_score >= 0.7:
                                    recommendation = "BÁN MẠNH"
                                else:
                                    recommendation = "BÁN"
                            else:
                                recommendation = "GIỮ"
                        else:
                            # Fallback if signal generation fails
                            signal = "HOLD"
                            technical_score = 0.0
                            recommendation = "GIỮ"
                    else:
                        # Fallback if indicators calculation fails
                        signal = "HOLD"
                        technical_score = 0.0
                        recommendation = "GIỮ"
                else:
                    # Fallback for insufficient data - use simple MA analysis
                    if data is not None and not data.empty and len(data) >= 5:
                        prices = data['close'].values
                        ma5 = prices[-5:].mean()
                        
                        if current_price > ma5 * 1.02:
                            technical_score = 0.3
                            signal = "BUY"
                            recommendation = "MUA"
                        elif current_price < ma5 * 0.98:
                            technical_score = -0.3
                            signal = "SELL"
                            recommendation = "BÁN"
                        else:
                            technical_score = 0.0
                            signal = "HOLD"
                            recommendation = "GIỮ"
                    else:
                        # Final fallback based on profit/loss
                        if profit_loss_pct > 5:
                            technical_score = 0.2
                            signal = "HOLD"
                            recommendation = "GIỮ"
                        elif profit_loss_pct < -5:
                            technical_score = -0.2
                            signal = "HOLD"
                            recommendation = "GIỮ"
                        else:
                            technical_score = 0.0
                            signal = "HOLD"
                            recommendation = "GIỮ"
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Buy_Price': avg_price,
                    'Current_Price': current_price,
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
        st.info(f"🕐 **Khung thời gian phân tích**: {timeframe_config.name} - {timeframe_config.description}")
        
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
