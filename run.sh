#!/bin/bash

# 🚀 STOCK ANALYSIS SYSTEM LAUNCHER
# ==================================
# Ứng dụng web phân tích chứng khoán Việt Nam tích hợp

echo "🚀 Khởi động hệ thống phân tích chứng khoán..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "💡 Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🌐 Starting web application..."
echo "📱 Access URL: http://localhost:8501"
echo "⚠️  Press Ctrl+C to stop"
echo ""

# Run the main integrated app
streamlit run web_app/main.py --server.port 8501 --server.address localhost

echo ""
echo "👋 Application stopped."
