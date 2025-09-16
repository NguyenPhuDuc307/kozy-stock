"""
🔄 BACKTEST PAGE - Trang kiểm tra lại chiến lược
===============================================

Trang kiểm tra hiệu quả của các chiến lược trading
"""

import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def render_backtest_page():
    """
    Render trang backtest chiến lược
    """
    st.markdown("# 🔄 Backtest chiến lược")
    
    # Strategy selection
    st.markdown("## 🎯 Chọn chiến lược")
    
    strategy = st.selectbox(
        "Chiến lược trading:",
        [
            "Golden Cross (MA20 x MA50)",
            "RSI Oversold/Overbought", 
            "MACD Signal",
            "Bollinger Bands Bounce",
            "Moving Average Crossover",
            "Custom Strategy"
        ],
        index=0
    )
    
    # Stock selection
    st.markdown("## 📈 Chọn cổ phiếu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Cổ phiếu:",
            ["VCB", "CTG", "BID", "ACB", "VIC", "FPT", "MSN", "VNM", "PLX", "TCB"],
            index=0
        )
    
    with col2:
        timeframe = st.selectbox(
            "Khung thời gian:",
            ["1D", "1H", "4H"],
            index=0
        )
    
    # Backtest period
    st.markdown("## ⏰ Khoảng thời gian backtest")
    
    period = st.selectbox(
        "Thời gian:",
        ["6 tháng", "1 năm", "2 năm", "3 năm"],
        index=1
    )
    
    # Strategy parameters
    st.markdown("## ⚙️ Tham số chiến lược")
    
    if strategy == "Golden Cross (MA20 x MA50)":
        col1, col2 = st.columns(2)
        with col1:
            fast_ma = st.number_input("MA nhanh:", min_value=5, max_value=50, value=20)
        with col2:
            slow_ma = st.number_input("MA chậm:", min_value=20, max_value=200, value=50)
    
    elif strategy == "RSI Oversold/Overbought":
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.number_input("RSI Period:", min_value=5, max_value=50, value=14)
        with col2:
            oversold = st.number_input("Oversold level:", min_value=10, max_value=40, value=30)
        with col3:
            overbought = st.number_input("Overbought level:", min_value=60, max_value=90, value=70)
    
    elif strategy == "MACD Signal":
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_ema = st.number_input("Fast EMA:", min_value=5, max_value=20, value=12)
        with col2:
            slow_ema = st.number_input("Slow EMA:", min_value=20, max_value=50, value=26)
        with col3:
            signal_ema = st.number_input("Signal EMA:", min_value=5, max_value=20, value=9)
    
    # Risk management
    st.markdown("## 🛡️ Quản lý rủi ro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "Vốn ban đầu (VND):",
            min_value=10000000,
            max_value=10000000000,
            value=100000000,
            step=10000000,
            format="%d"
        )
    
    with col2:
        position_size = st.slider(
            "Tỷ lệ vốn mỗi lệnh (%):",
            min_value=1,
            max_value=100,
            value=10
        )
    
    with col3:
        stop_loss = st.slider(
            "Stop Loss (%):",
            min_value=1,
            max_value=20,
            value=5
        )
    
    # Commission settings
    commission = st.slider(
        "Phí giao dịch (%):",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05
    ) / 100
    
    # Run backtest button
    if st.button("🚀 Chạy Backtest", type="primary"):
        with st.spinner("🔄 Đang chạy backtest..."):
            try:
                # Import libraries
                from datetime import datetime, timedelta
                import pandas as pd
                import numpy as np
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                from src.data.data_provider import DataProvider
                from src.utils.config import ConfigManager
                from src.analysis.indicators import TechnicalIndicators
                
                # Simple config for DataProvider
                class SimpleConfig:
                    CACHE_ENABLED = True
                    CACHE_DURATION = 300
                
                # Calculate backtest period
                period_map = {
                    "6 tháng": 180,
                    "1 năm": 365,
                    "2 năm": 730,
                    "3 năm": 1095
                }
                days = period_map[period]
                
                # Get data
                config = ConfigManager()
                data_provider = DataProvider(SimpleConfig())
                tech_indicators = TechnicalIndicators()
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                data = data_provider.get_historical_data(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    resolution=timeframe
                )
                
                if data is None or data.empty:
                    st.error("❌ Không thể lấy dữ liệu!")
                    return
                
                # Calculate indicators based on strategy
                if strategy == "Golden Cross (MA20 x MA50)":
                    data['ma_fast'] = tech_indicators.sma(data['close'], fast_ma)
                    data['ma_slow'] = tech_indicators.sma(data['close'], slow_ma)
                    
                    # Generate signals
                    data['signal'] = 0
                    data['signal'][fast_ma:] = np.where(
                        data['ma_fast'][fast_ma:] > data['ma_slow'][fast_ma:], 1, 0
                    )
                    data['position'] = data['signal'].diff()
                
                elif strategy == "RSI Oversold/Overbought":
                    data['rsi'] = tech_indicators.rsi(data['close'], rsi_period)
                    
                    # Generate signals
                    data['signal'] = 0
                    data.loc[data['rsi'] < oversold, 'signal'] = 1  # Buy signal
                    data.loc[data['rsi'] > overbought, 'signal'] = -1  # Sell signal
                    data['position'] = data['signal'].diff()
                
                elif strategy == "MACD Signal":
                    macd_data = tech_indicators.macd(data['close'], fast_ema, slow_ema, signal_ema)
                    data['macd'] = macd_data['macd']
                    data['macd_signal'] = macd_data['signal']
                    data['macd_histogram'] = macd_data['histogram']
                    
                    # Generate signals
                    data['signal'] = 0
                    data['signal'] = np.where(data['macd'] > data['macd_signal'], 1, 0)
                    data['position'] = data['signal'].diff()
                
                else:
                    # Default MA crossover
                    data['ma_fast'] = tech_indicators.sma(data['close'], 20)
                    data['ma_slow'] = tech_indicators.sma(data['close'], 50)
                    data['signal'] = 0
                    data['signal'][20:] = np.where(
                        data['ma_fast'][20:] > data['ma_slow'][20:], 1, 0
                    )
                    data['position'] = data['signal'].diff()
                
                # Run backtest
                portfolio_value = initial_capital
                position = 0
                shares = 0
                trades = []
                portfolio_values = [initial_capital]
                
                for i in range(1, len(data)):
                    current_price = data['close'].iloc[i]
                    
                    # Buy signal
                    if data['position'].iloc[i] == 1 and position == 0:
                        position_value = portfolio_value * (position_size / 100)
                        shares = position_value / current_price
                        commission_cost = position_value * commission
                        portfolio_value -= (position_value + commission_cost)
                        position = 1
                        
                        trades.append({
                            'date': data.index[i],
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'value': position_value,
                            'commission': commission_cost
                        })
                    
                    # Sell signal
                    elif data['position'].iloc[i] == -1 and position == 1:
                        sell_value = shares * current_price
                        commission_cost = sell_value * commission
                        portfolio_value += (sell_value - commission_cost)
                        
                        trades.append({
                            'date': data.index[i],
                            'type': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': sell_value,
                            'commission': commission_cost
                        })
                        
                        shares = 0
                        position = 0
                    
                    # Calculate current portfolio value
                    if position == 1:
                        current_portfolio = portfolio_value + (shares * current_price)
                    else:
                        current_portfolio = portfolio_value
                    
                    portfolio_values.append(current_portfolio)
                
                # Final portfolio value
                if position == 1:
                    final_value = portfolio_value + (shares * data['close'].iloc[-1])
                else:
                    final_value = portfolio_value
                
                # Calculate metrics
                total_return = (final_value - initial_capital) / initial_capital * 100
                
                # Buy and hold return
                buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
                
                # Calculate portfolio returns
                portfolio_df = pd.DataFrame({
                    'date': data.index,
                    'portfolio_value': portfolio_values
                })
                portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
                
                # Sharpe ratio (giả sử risk-free rate = 5%)
                excess_returns = portfolio_df['returns'].dropna() - (0.05 / 252)
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * (252**0.5) if excess_returns.std() > 0 else 0
                
                # Max drawdown
                rolling_max = portfolio_df['portfolio_value'].cummax()
                drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                # Win rate
                profitable_trades = 0
                total_trades = len([t for t in trades if t['type'] == 'SELL'])
                
                if total_trades > 0:
                    for i in range(0, len(trades) - 1, 2):
                        if i + 1 < len(trades):
                            buy_price = trades[i]['price']
                            sell_price = trades[i + 1]['price']
                            if sell_price > buy_price:
                                profitable_trades += 1
                    
                    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
                else:
                    win_rate = 0
                
                # Display results
                st.success("✅ Backtest hoàn thành!")
                
                # Performance summary
                st.markdown("## 📊 Kết quả Backtest")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Tổng lợi nhuận", f"{total_return:.2f}%")
                
                with col2:
                    st.metric("Buy & Hold", f"{buy_hold_return:.2f}%")
                
                with col3:
                    st.metric("Vốn cuối kỳ", f"{final_value:,.0f} VND")
                
                with col4:
                    st.metric("Số giao dịch", f"{len(trades)}")
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                with col2:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                with col3:
                    st.metric("Tỷ lệ thắng", f"{win_rate:.1f}%")
                
                with col4:
                    alpha = total_return - buy_hold_return
                    st.metric("Alpha", f"{alpha:.2f}%")
                
                # Portfolio value chart
                st.markdown("## 📈 Biểu đồ giá trị danh mục")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Giá trị danh mục', 'Giá cổ phiếu & Tín hiệu'),
                    row_weights=[0.6, 0.4]
                )
                
                # Portfolio value
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_df['date'],
                        y=portfolio_df['portfolio_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Buy & Hold comparison
                buy_hold_values = [initial_capital * (1 + (data['close'].iloc[i] - data['close'].iloc[0]) / data['close'].iloc[0]) for i in range(len(data))]
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=buy_hold_values,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Stock price
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['close'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(color='black', width=1)
                    ),
                    row=2, col=1
                )
                
                # Buy signals
                buy_signals = data[data['position'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals['close'],
                            mode='markers',
                            name='Buy Signal',
                            marker=dict(color='green', size=8, symbol='triangle-up')
                        ),
                        row=2, col=1
                    )
                
                # Sell signals
                sell_signals = data[data['position'] == -1]
                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_signals['close'],
                            mode='markers',
                            name='Sell Signal',
                            marker=dict(color='red', size=8, symbol='triangle-down')
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    title=f'Backtest Results: {strategy} on {symbol}',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade history
                if trades:
                    st.markdown("## 📋 Lịch sử giao dịch")
                    
                    trades_df = pd.DataFrame(trades)
                    trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                    trades_df['price'] = trades_df['price'].round(2)
                    trades_df['value'] = trades_df['value'].round(0)
                    trades_df['commission'] = trades_df['commission'].round(0)
                    
                    st.dataframe(trades_df, use_container_width=True)
                
                # Strategy analysis
                st.markdown("## 💡 Phân tích chiến lược")
                
                if total_return > buy_hold_return:
                    st.success(f"✅ Chiến lược {strategy} có hiệu suất tốt hơn Buy & Hold với alpha {alpha:.2f}%")
                else:
                    st.warning(f"⚠️ Chiến lược {strategy} không hiệu quả bằng Buy & Hold. Alpha âm: {alpha:.2f}%")
                
                if sharpe_ratio > 1:
                    st.success(f"✅ Sharpe Ratio tốt ({sharpe_ratio:.2f}) - Hiệu suất điều chỉnh rủi ro cao")
                elif sharpe_ratio > 0.5:
                    st.info(f"ℹ️ Sharpe Ratio ổn ({sharpe_ratio:.2f}) - Hiệu suất điều chỉnh rủi ro trung bình")
                else:
                    st.warning(f"⚠️ Sharpe Ratio thấp ({sharpe_ratio:.2f}) - Cần cải thiện")
                
                if win_rate >= 60:
                    st.success(f"✅ Tỷ lệ thắng cao ({win_rate:.1f}%) - Chiến lược ổn định")
                elif win_rate >= 40:
                    st.info(f"ℹ️ Tỷ lệ thắng trung bình ({win_rate:.1f}%)")
                else:
                    st.warning(f"⚠️ Tỷ lệ thắng thấp ({win_rate:.1f}%) - Cần điều chỉnh")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi chạy backtest: {str(e)}")
                # Fallback simple interface
                st.info("🔄 Hiển thị giao diện đơn giản...")
                
                st.markdown("### 📊 Kết quả sẽ hiển thị ở đây")
                st.info(f"Chiến lược: {strategy}")
                st.info(f"Cổ phiếu: {symbol}")
                st.info(f"Thời gian: {period}")
                st.info(f"Vốn ban đầu: {initial_capital:,} VND")

# Main page function for st.Page
render_backtest_page()
