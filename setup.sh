#!/bin/bash

# 🚀 SETUP SCRIPT - Thiết lập hệ thống phân tích chứng khoán
# ========================================================

echo "🚀 Đang thiết lập hệ thống phân tích chứng khoán Việt Nam..."

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 chưa được cài đặt. Vui lòng cài đặt Python 3.8+"
    exit 1
fi

echo "✅ Python đã được cài đặt: $(python3 --version)"

# Tạo virtual environment
echo "📦 Tạo virtual environment..."
python3 -m venv venv

# Kích hoạt virtual environment
echo "🔄 Kích hoạt virtual environment..."
source venv/bin/activate

# Cập nhật pip
echo "⬆️ Cập nhật pip..."
pip install --upgrade pip

# Cài đặt requirements
echo "📚 Cài đặt các thư viện cần thiết..."
pip install -r requirements.txt

# Tạo thư mục logs và cache
echo "📁 Tạo thư mục cần thiết..."
mkdir -p logs
mkdir -p cache
mkdir -p data

# Tạo file .env mẫu
echo "⚙️ Tạo file cấu hình..."
cat > .env << EOF
# Cấu hình môi trường
ENVIRONMENT=development
LOG_LEVEL=INFO
CACHE_ENABLED=true
EOF

# Tạo file run script
echo "📝 Tạo script chạy ứng dụng..."
cat > run_app.sh << 'EOF'
#!/bin/bash
echo "🚀 Khởi động ứng dụng web..."
source venv/bin/activate
streamlit run web_app/app.py --server.port 8501 --server.headless true
EOF

chmod +x run_app.sh

# Tạo file run examples
cat > run_examples.sh << 'EOF'
#!/bin/bash
echo "🧪 Chạy các ví dụ..."
source venv/bin/activate
cd examples
python basic_analysis.py
EOF

chmod +x run_examples.sh

echo ""
echo "✅ Thiết lập hoàn thành!"
echo ""
echo "📋 Các bước tiếp theo:"
echo "1. Kích hoạt virtual environment: source venv/bin/activate"
echo "2. Chạy ứng dụng web: ./run_app.sh"
echo "3. Hoặc chạy ví dụ: ./run_examples.sh"
echo ""
echo "🌐 Ứng dụng web sẽ chạy tại: http://localhost:8501"
echo ""
echo "📖 Xem thêm tài liệu trong README.md"
