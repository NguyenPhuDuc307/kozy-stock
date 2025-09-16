#!/bin/bash
echo "🚀 Khởi động ứng dụng web..."
source venv/bin/activate
streamlit run web_app/app.py --server.port 8501 --server.headless true
