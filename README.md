# 📊 Hệ thống phân tích chứng khoán Việt Nam

Ứng dụng web tích hợp phân tích chứng khoán, quét thị trường và backtesting chiến lược.

## 🚀 Cài đặt và chạy

### 1. Setup môi trường
```bash
./setup.sh
```

### 2. Chạy ứng dụng
```bash
./run.sh
```

### 3. Truy cập ứng dụng
Mở trình duyệt và truy cập: http://localhost:8501

## ✨ Tính năng

### 📈 Phân tích cổ phiếu
- Biểu đồ nến tương tác
- 40+ chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands...)
- Tín hiệu mua/bán tự động
- Thông tin công ty chi tiết

### 🔍 Quét thị trường
- Quét toàn thị trường real-time
- Xếp hạng cổ phiếu theo tín hiệu
- Phân tích thanh khoản
- Lọc theo ngành (Banks, Real Estate, Technology...)

### 📋 Backtest chiến lược
- 4 chiến lược giao dịch:
  - Moving Average Crossover
  - Mean Reversion  
  - Momentum Strategy
  - Bollinger Bands Strategy
- Quản lý rủi ro (Stop Loss, Take Profit)
- Phân tích hiệu suất chi tiết
- Sharpe Ratio, Win Rate, Max Drawdown

### 📊 So sánh cổ phiếu
- So sánh hiệu suất nhiều cổ phiếu
- Phân tích tương quan

## 📋 Cấu trúc dự án

```
kozy-stock/
├── setup.sh              # Script cài đặt
├── run.sh                 # Script chạy ứng dụng
├── web_app/
│   └── main.py        # Ứng dụng web chính
├── src/                   # Mã nguồn hệ thống
├── examples/              # Ví dụ sử dụng
├── tests/                 # Test cases
└── docs/                  # Tài liệu
```

## 🎯 Hướng dẫn sử dụng

### Phân tích cổ phiếu
1. Chọn tab "📈 Phân tích cổ phiếu"
2. Chọn mã cổ phiếu và thời gian
3. Nhấn "📊 Phân tích"
4. Xem biểu đồ, chỉ báo và tín hiệu

### Quét thị trường
1. Chọn tab "🔍 Quét thị trường"
2. Chọn loại quét (VN30, Banks, Real Estate...)
3. Điều chỉnh bộ lọc
4. Nhấn "🔍 Quét thị trường"
5. Xem kết quả và top picks

### Backtest chiến lược
1. Chọn tab "📋 Backtest chiến lược"
2. Chọn chiến lược và cổ phiếu
3. Điều chỉnh tham số
4. Nhấn "� Chạy Backtest"
5. Xem kết quả hiệu suất

## ⚠️ Lưu ý quan trọng

- Hệ thống chỉ mang tính chất tham khảo
- Không phải lời khuyên đầu tư
- Luôn tự nghiên cứu trước khi đầu tư
- Quản lý rủi ro là ưu tiên hàng đầu

## 📞 Hỗ trợ

- 📧 Email: support@stockanalysis.vn (demo)
- 📱 Hotline: 1900-xxx-xxx (demo)
- 💬 Telegram: @vnstock_support (demo)

---

**Phát triển bởi:** Vietnam Stock Analysis Team
**Phiên bản:** 2.0.0
**Cập nhật:** September 2025

## 🏗️ Kiến trúc hệ thống

```
kozy-stock/
├── 📁 src/                         # Mã nguồn chính
│   ├── 🐍 main.py                  # StockAnalysisSystem - Lớp chính
│   ├── 📁 data/                    # Quản lý dữ liệu
│   │   └── 🐍 data_provider.py     # DataProvider - Lấy & cache dữ liệu
│   ├── 📁 analysis/                # Phân tích kỹ thuật  
│   │   ├── 🐍 indicators.py        # TechnicalIndicators - 30+ chỉ báo
│   │   └── 🐍 signals.py           # TradingSignals - Tạo tín hiệu
│   ├── 📁 visualization/           # Biểu đồ & trực quan hóa
│   │   └── 🐍 charts.py            # InteractiveCharts - Plotly charts
│   ├── � strategies/              # Chiến lược trading
│   │   └── 🐍 backtest.py          # BacktestEngine + Strategy Library
│   └── 📁 utils/                   # Tiện ích & cấu hình
│       └── 🐍 config.py            # ConfigManager - Quản lý cấu hình
├── 📁 web_app/                     # Ứng dụng web Streamlit
│   └── 🐍 app.py                   # Streamlit web application
├── 📁 examples/                    # Ví dụ & hướng dẫn
│   ├── 🐍 basic_analysis.py        # Ví dụ phân tích cơ bản
│   └── 🐍 backtest_example.py      # Ví dụ backtest chiến lược
├── � tests/                       # Test cases & validation
│   └── 🐍 test_main.py             # Unit tests cho hệ thống
├── 📁 docs/                        # Tài liệu & hướng dẫn
├── 📁 logs/                        # Log files (tự tạo)
├── � cache/                       # Cache dữ liệu (tự tạo)
├── 📄 requirements.txt             # Dependencies
├── 📄 setup.sh                     # Script cài đặt tự động
├── 📄 config.json                  # File cấu hình (tự tạo)
└── 📄 README.md                    # Tài liệu này
```

## 🚀 Tính năng chính

### 📊 Thu thập & Xử lý dữ liệu
- ✅ Lấy dữ liệu realtime từ vnstock
- ✅ Cache thông minh để tối ưu hiệu suất
- ✅ Xử lý và làm sạch dữ liệu
- ✅ Hỗ trợ multiple timeframes

### 📈 Phân tích kỹ thuật
- ✅ 20+ chỉ báo kỹ thuật (MA, RSI, MACD, Bollinger...)
- ✅ Nhận dạng patterns (Head & Shoulders, Triangles...)
- ✅ Tìm Support & Resistance levels
- ✅ Phân tích volume và momentum

### 🎯 Chiến lược giao dịch
- ✅ Trend Following (Golden Cross, MACD...)
- ✅ Mean Reversion (RSI, Bollinger Bands...)
- ✅ Breakout strategies
- ✅ Multi-timeframe analysis

### ⚖️ Quản lý rủi ro
- ✅ Position sizing (Kelly Criterion, Fixed %)
- ✅ Stop loss strategies (ATR, Technical, Fixed)
- ✅ Risk-reward optimization
- ✅ Portfolio risk metrics

### 📊 Backtesting
- ✅ Historical backtesting engine
- ✅ Performance metrics (Sharpe, Sortino, Max DD...)
- ✅ Monte Carlo simulation
- ✅ Strategy comparison

### 🎨 Visualization
- ✅ Interactive candlestick charts
- ✅ Technical indicators overlay
- ✅ Pattern recognition visualization
- ✅ Performance dashboards

### 🌐 Web Interface
- ✅ Streamlit-based web app
- ✅ Real-time charting
- ✅ Strategy backtesting UI
- ✅ Portfolio tracking

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd kozy-stock
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình
```bash
cp src/utils/config.py.example src/utils/config.py
# Chỉnh sửa config.py theo nhu cầu
```

## 🚀 Sử dụng

### 1. Chạy phân tích cổ phiếu đơn lẻ
```bash
python examples/basic_analysis.py
```

### 2. Chạy Market Scanner
```bash
# Quét thị trường qua terminal
./run_market_scanner.sh

# Hoặc chạy example
python examples/market_scanner_example.py
```

### 3. Chạy web applications
```bash
# Web app phân tích cổ phiếu
streamlit run web_app/app.py

# Market Scanner web app  
./run_market_scanner_web.sh

# Main app với navigation
./run_main_app.sh
```

### 4. Market Scanner - Tính năng mới 🔍

Market Scanner là tính năng mạnh mẽ để quét và phân tích toàn bộ thị trường:

#### 🎯 Các loại quét:
- **Quét nhanh**: Phân tích 10 cổ phiếu hàng đầu
- **Quét toàn thị trường**: Phân tích 100+ cổ phiếu
- **Quét theo ngành**: Phân tích từng ngành riêng biệt
- **Quét tùy chỉnh**: Phân tích danh sách cổ phiếu tự chọn

#### 📊 Kết quả cung cấp:
- Tín hiệu giao dịch (BUY/SELL/HOLD) cho từng cổ phiếu
- Điểm số tín hiệu (-1.0 đến +1.0)
- Xếp hạng theo thanh khoản
- Phân tích theo ngành
- Báo cáo Excel chi tiết

#### 🚀 Cách sử dụng Market Scanner:

**Terminal:**
```bash
./run_market_scanner.sh
# Chọn: 1=Nhanh, 2=Toàn thị trường, 3=Theo ngành
```

**Web App:**
```bash
./run_market_scanner_web.sh
# Truy cập: http://localhost:8501
```

**Python Code:**
```python
from src.analysis.market_scanner import MarketScanner

scanner = MarketScanner()

# Quét nhanh 10 cổ phiếu
results, _ = scanner.scan_top_stocks()

# Quét toàn thị trường
results, _ = scanner.scan_full_market()

# Quét theo ngành
sector_results = scanner.scan_by_sectors()
```

### 5. Sử dụng như một library
```python
from src.main import StockAnalysisSystem

# Khởi tạo hệ thống
system = StockAnalysisSystem()

# Phân tích một cổ phiếu
analysis = system.analyze_stock('VCB', period='6M')

# Backtest một chiến lược
results = system.backtest_strategy('golden_cross', 'VCB', '2023-01-01', '2024-01-01')

# Tạo báo cáo
system.generate_report('VCB', output='pdf')
```

## 📊 Ví dụ sử dụng

### Phân tích cơ bản
```python
# Lấy dữ liệu và tính chỉ báo
data = system.get_stock_data('VCB', '1Y')
indicators = system.calculate_indicators(data)

# Tìm tín hiệu
signals = system.find_signals(data, strategy='multi_indicator')

# Vẽ biểu đồ
chart = system.create_chart(data, indicators, signals)
chart.show()
```

### Backtesting
```python
# Định nghĩa chiến lược
strategy = {
    'name': 'RSI_Bollinger',
    'entry': {'RSI': '<30', 'Price': 'touches_lower_bb'},
    'exit': {'RSI': '>70', 'Price': 'touches_upper_bb'},
    'stop_loss': 0.05,
    'take_profit': 0.10
}

# Chạy backtest
results = system.backtest(strategy, ['VCB', 'FPT', 'VHM'], '2022-01-01', '2024-01-01')
print(results.summary())
```

### So sánh chiến lược
```python
strategies = ['golden_cross', 'rsi_mean_reversion', 'breakout']
comparison = system.compare_strategies(strategies, 'VCB', '2Y')
system.plot_strategy_comparison(comparison)
```

## 🎯 Chiến lược có sẵn

### 1. Trend Following
- **Golden Cross**: SMA50 > SMA200
- **MACD Crossover**: MACD > Signal
- **EMA Ribbon**: Multiple EMA alignment

### 2. Mean Reversion
- **RSI Contrarian**: Buy RSI<30, Sell RSI>70
- **Bollinger Bands**: Buy at lower band, sell at upper
- **Support/Resistance**: Trade bounces

### 3. Breakout
- **Volume Breakout**: High volume + price breakout
- **Pattern Breakout**: Triangle, flag patterns
- **ATR Breakout**: Volatility-based entries

### 4. Multi-timeframe
- **Trend + Entry**: Weekly trend, daily entry
- **Momentum Confluence**: Multiple timeframe RSI
- **Support/Resistance Multi-TF**: Key levels across timeframes

## 📈 Supported Indicators

### Trend Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- Rate of Change (ROC)
- Momentum

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Standard Deviation
- Keltner Channels

### Volume Indicators
- Volume Moving Average
- On Balance Volume (OBV)
- Volume Rate of Change
- Accumulation/Distribution Line

## 🔧 Cấu hình

### config.py
```python
# Data settings
DATA_SOURCE = 'VCI'  # VCI, VND, TCBS
CACHE_ENABLED = True
CACHE_DURATION = 300  # seconds

# Analysis settings
DEFAULT_TIMEFRAME = '1D'
INDICATORS_PERIODS = {
    'SMA_SHORT': 20,
    'SMA_LONG': 50,
    'RSI': 14,
    'MACD': (12, 26, 9)
}

# Risk management
DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.10
MAX_POSITION_SIZE = 0.1  # 10% of portfolio

# Visualization
CHART_THEME = 'plotly_white'
FIGURE_SIZE = (1200, 800)
```

## 🧪 Testing

```bash
# Chạy tất cả tests
python -m pytest tests/

# Chạy tests cho module cụ thể
python -m pytest tests/test_indicators.py

# Chạy với coverage
python -m pytest --cov=src tests/
```

## 📚 Tài liệu

- [📖 Hướng dẫn phân tích kỹ thuật](docs/technical_analysis_guide.md)
- [🔧 API Reference](docs/api_reference.md)
- [👤 User Guide](docs/user_guide.md)
- [📝 Ví dụ](examples/)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

MIT License - xem [LICENSE](LICENSE) file.

## ⚠️ Disclaimer

Hệ thống này chỉ mang tính chất giáo dục và nghiên cứu. Không phải lời khuyên đầu tư. Luôn có kế hoạch quản lý rủi ro khi giao dịch thực tế.

## 📞 Liên hệ

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**🚀 Happy Trading! 📈**
