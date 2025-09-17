"""
📊 TRADING PORTFOLIO MANAGER
============================

Quản lý nhiều danh mục giao dịch (CRUD) để tổ chức các giao dịch theo từng chiến lược khác nhau
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from .trading_history import TradingHistory

class TradingPortfolioManager:
    """
    Quản lý nhiều danh mục giao dịch khác nhau
    """
    
    def __init__(self, data_file: str = "trading_portfolios.json"):
        """
        Khởi tạo trading portfolio manager
        
        Args:
            data_file: Tên file lưu trữ thông tin các danh mục giao dịch
        """
        self.data_file = data_file
        self.portfolios_data = self._load_portfolios()
        self._portfolio_instances = {}  # Cache cho các instance TradingHistory
    
    def _load_portfolios(self) -> Dict:
        """Tải thông tin các danh mục từ file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"portfolios": {}}
        else:
            return {"portfolios": {}}
    
    def _save_portfolios(self):
        """Lưu thông tin các danh mục vào file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.portfolios_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi lưu file danh mục: {e}")
    
    def create_portfolio(self, name: str, description: str = "", 
                        initial_cash: float = 0.0, strategy: str = "") -> str:
        """
        Tạo danh mục giao dịch mới
        
        Args:
            name: Tên danh mục
            description: Mô tả danh mục
            initial_cash: Số tiền ban đầu
            strategy: Chiến lược đầu tư
            
        Returns:
            portfolio_id: ID của danh mục vừa tạo
        """
        # Tạo ID duy nhất cho danh mục
        portfolio_id = f"portfolio_{len(self.portfolios_data['portfolios']) + 1}_{int(datetime.now().timestamp())}"
        
        # Tạo file riêng cho lịch sử giao dịch của danh mục này
        trading_file = f"trading_history_{portfolio_id}.json"
        
        portfolio_info = {
            "id": portfolio_id,
            "name": name,
            "description": description,
            "initial_cash": initial_cash,
            "current_cash": initial_cash,
            "strategy": strategy,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trading_file": trading_file,
            "is_active": True,
            "total_invested": 0.0,
            "total_profit_loss": 0.0,
            "total_value": initial_cash
        }
        
        # Lưu thông tin danh mục
        self.portfolios_data["portfolios"][portfolio_id] = portfolio_info
        self._save_portfolios()
        
        # Tạo instance TradingHistory cho danh mục này
        self._portfolio_instances[portfolio_id] = TradingHistory(trading_file)
        
        return portfolio_id
    
    def get_portfolios(self) -> List[Dict]:
        """Lấy danh sách tất cả danh mục"""
        portfolios = []
        for portfolio_id, info in self.portfolios_data["portfolios"].items():
            if info.get("is_active", True):
                # Cập nhật thống kê
                self._update_portfolio_stats(portfolio_id)
                portfolios.append(info.copy())
        
        return portfolios
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Dict]:
        """
        Lấy thông tin một danh mục cụ thể
        
        Args:
            portfolio_id: ID của danh mục
            
        Returns:
            Thông tin danh mục hoặc None nếu không tìm thấy
        """
        if portfolio_id in self.portfolios_data["portfolios"]:
            portfolio_info = self.portfolios_data["portfolios"][portfolio_id].copy()
            if portfolio_info.get("is_active", True):
                self._update_portfolio_stats(portfolio_id)
                return portfolio_info
        return None
    
    def update_portfolio(self, portfolio_id: str, name: str = None, 
                        description: str = None, strategy: str = None) -> bool:
        """
        Cập nhật thông tin danh mục
        
        Args:
            portfolio_id: ID của danh mục
            name: Tên mới (nếu có)
            description: Mô tả mới (nếu có)
            strategy: Chiến lược mới (nếu có)
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        if portfolio_id not in self.portfolios_data["portfolios"]:
            return False
        
        portfolio = self.portfolios_data["portfolios"][portfolio_id]
        
        if name is not None:
            portfolio["name"] = name
        if description is not None:
            portfolio["description"] = description
        if strategy is not None:
            portfolio["strategy"] = strategy
        
        portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self._save_portfolios()
        return True
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """
        Xóa danh mục (soft delete - đánh dấu is_active = False)
        
        Args:
            portfolio_id: ID của danh mục cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        if portfolio_id not in self.portfolios_data["portfolios"]:
            return False
        
        # Soft delete
        self.portfolios_data["portfolios"][portfolio_id]["is_active"] = False
        self.portfolios_data["portfolios"][portfolio_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Xóa khỏi cache
        if portfolio_id in self._portfolio_instances:
            del self._portfolio_instances[portfolio_id]
        
        self._save_portfolios()
        return True
    
    def get_trading_history(self, portfolio_id: str) -> Optional[TradingHistory]:
        """
        Lấy instance TradingHistory của một danh mục
        
        Args:
            portfolio_id: ID của danh mục
            
        Returns:
            Instance TradingHistory hoặc None nếu không tìm thấy
        """
        if portfolio_id not in self.portfolios_data["portfolios"]:
            return None
        
        portfolio = self.portfolios_data["portfolios"][portfolio_id]
        if not portfolio.get("is_active", True):
            return None
        
        # Kiểm tra cache
        if portfolio_id not in self._portfolio_instances:
            trading_file = portfolio["trading_file"]
            self._portfolio_instances[portfolio_id] = TradingHistory(trading_file)
        
        return self._portfolio_instances[portfolio_id]
    
    def add_transaction(self, portfolio_id: str, symbol: str, transaction_type: str,
                       quantity: int, price: float, date: str = None,
                       fee: float = 0.0, note: str = "") -> Optional[int]:
        """
        Thêm giao dịch vào danh mục cụ thể
        
        Args:
            portfolio_id: ID của danh mục
            symbol: Mã cổ phiếu
            transaction_type: 'BUY' hoặc 'SELL'
            quantity: Số lượng
            price: Giá
            date: Ngày giao dịch
            fee: Phí giao dịch
            note: Ghi chú
            
        Returns:
            Transaction ID nếu thành công, None nếu thất bại
        """
        trading_history = self.get_trading_history(portfolio_id)
        if trading_history is None:
            return None
        
        # Thêm giao dịch
        transaction_id = trading_history.add_transaction(
            symbol, transaction_type, quantity, price, date, fee, note
        )
        
        # Cập nhật thống kê danh mục
        self._update_portfolio_stats(portfolio_id)
        
        return transaction_id
    
    def clear_symbol_transactions(self, portfolio_id: str, symbol: str) -> bool:
        """
        Xóa tất cả giao dịch của một mã cổ phiếu trong danh mục cụ thể
        
        Args:
            portfolio_id: ID của danh mục
            symbol: Mã cổ phiếu cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        trading_history = self.get_trading_history(portfolio_id)
        if trading_history is None:
            return False
        
        # Xóa giao dịch của symbol
        result = trading_history.clear_symbol_transactions(symbol)
        
        if result:
            # Cập nhật thống kê danh mục
            self._update_portfolio_stats(portfolio_id)
        
        return result
    
    def _update_portfolio_stats(self, portfolio_id: str):
        """Cập nhật thống kê của danh mục"""
        if portfolio_id not in self.portfolios_data["portfolios"]:
            return
        
        trading_history = self.get_trading_history(portfolio_id)
        if trading_history is None:
            return
        
        portfolio = self.portfolios_data["portfolios"][portfolio_id]
        
        # Tính tổng đầu tư và số dư hiện tại
        total_invested = 0.0
        total_current_value = 0.0
        
        holdings = trading_history.get_current_holdings()
        for symbol, data in holdings.items():
            total_invested += data["total_cost"]
            
            # Lấy giá hiện tại thực tế từ API
            try:
                # Import vnstock để lấy giá hiện tại
                import vnstock as vn
                from datetime import datetime as dt, timedelta
                
                # Lấy dữ liệu giá gần nhất
                end_date = dt.now()
                start_date = end_date - timedelta(days=7)  # Lấy 7 ngày gần nhất
                
                # Sử dụng API đúng của vnstock 4.x
                quote = vn.Quote(symbol=symbol, source='VCI')
                stock_data = quote.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if stock_data is not None and not stock_data.empty:
                    # API trả về giá theo nghìn VND, cần nhân 1000 để về VND đầy đủ
                    api_price = stock_data['close'].iloc[-1]
                    current_price = api_price * 1000
                    total_current_value += data["shares"] * current_price
                else:
                    # Fallback sử dụng avg_price nếu không lấy được giá mới
                    total_current_value += data["shares"] * data["avg_price"]
                    
            except Exception as e:
                # Fallback sử dụng avg_price nếu có lỗi
                print(f"Lỗi lấy giá hiện tại cho {symbol}: {e}")
                total_current_value += data["shares"] * data["avg_price"]
        
        portfolio["total_invested"] = total_invested
        portfolio["total_profit_loss"] = total_current_value - total_invested
        
        # Tính current_cash = initial_cash - total_invested
        current_cash = portfolio["initial_cash"] - total_invested
        portfolio["current_cash"] = current_cash
        portfolio["total_value"] = total_current_value + current_cash
        portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self._save_portfolios()
    
    def get_portfolio_summary(self, portfolio_id: str) -> Optional[pd.DataFrame]:
        """
        Lấy tóm tắt danh mục dạng DataFrame
        
        Args:
            portfolio_id: ID của danh mục
            
        Returns:
            DataFrame chứa thông tin danh mục hoặc None
        """
        trading_history = self.get_trading_history(portfolio_id)
        if trading_history is None:
            return None
        
        return trading_history.get_portfolio_summary()
    
    def clone_portfolio(self, source_portfolio_id: str, new_name: str, 
                       new_description: str = "") -> Optional[str]:
        """
        Nhân bản danh mục (chỉ sao chép thông tin, không sao chép giao dịch)
        
        Args:
            source_portfolio_id: ID danh mục nguồn
            new_name: Tên danh mục mới
            new_description: Mô tả danh mục mới
            
        Returns:
            ID danh mục mới hoặc None nếu thất bại
        """
        source_portfolio = self.get_portfolio(source_portfolio_id)
        if source_portfolio is None:
            return None
        
        # Tạo danh mục mới với thông tin tương tự
        new_portfolio_id = self.create_portfolio(
            name=new_name,
            description=new_description,
            initial_cash=source_portfolio["initial_cash"],
            strategy=source_portfolio["strategy"]
        )
        
        return new_portfolio_id
    
    def get_portfolio_performance(self, portfolio_id: str) -> Optional[Dict]:
        """
        Lấy thống kê hiệu suất của danh mục
        
        Args:
            portfolio_id: ID của danh mục
            
        Returns:
            Dict chứa các thông số hiệu suất
        """
        portfolio = self.get_portfolio(portfolio_id)
        trading_history = self.get_trading_history(portfolio_id)
        
        if portfolio is None or trading_history is None:
            return None
        
        holdings = trading_history.get_current_holdings()
        transactions = trading_history.get_transactions_history()
        
        # Tính các thông số cơ bản
        total_stocks = len(holdings)
        total_transactions = len(transactions)
        total_invested = portfolio["total_invested"]
        
        # Tính số ngày đầu tư
        if transactions:
            first_transaction_date = min(t["date"] for t in transactions)
            first_date = datetime.strptime(first_transaction_date.split(" ")[0], "%Y-%m-%d")
            days_invested = (datetime.now() - first_date).days
        else:
            days_invested = 0
        
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio["name"],
            "total_stocks": total_stocks,
            "total_transactions": total_transactions,
            "total_invested": total_invested,
            "current_value": portfolio.get("total_value", 0),
            "profit_loss": portfolio.get("total_profit_loss", 0),
            "days_invested": days_invested,
            "created_date": portfolio["created_date"],
            "last_updated": portfolio["last_updated"]
        }
    
    def refresh_all_portfolios(self):
        """Cập nhật lại thống kê cho tất cả các danh mục"""
        updated_count = 0
        for portfolio_id in self.portfolios_data["portfolios"].keys():
            portfolio = self.portfolios_data["portfolios"][portfolio_id]
            if portfolio.get("is_active", True):  # Chỉ cập nhật những portfolio đang hoạt động
                print(f"Đang cập nhật danh mục: {portfolio['name']}")
                self._update_portfolio_stats(portfolio_id)
                updated_count += 1
        
        print(f"✅ Đã cập nhật {updated_count} danh mục thành công!")
        return True

    def add_sample_portfolios(self):
        """Thêm dữ liệu mẫu cho demo"""
        # Danh mục cổ phiếu ngân hàng
        banking_id = self.create_portfolio(
            name="Danh mục Ngân hàng",
            description="Đầu tư vào các cổ phiếu ngân hàng lớn",
            initial_cash=100000000,  # 100 triệu
            strategy="Value Investing - Banking Sector"
        )
        
        # Thêm giao dịch mẫu cho danh mục ngân hàng
        banking_trades = [
            {"symbol": "VCB", "type": "BUY", "quantity": 100, "price": 78000, "date": "2024-01-15"},
            {"symbol": "TCB", "type": "BUY", "quantity": 200, "price": 25000, "date": "2024-01-20"},
            {"symbol": "BID", "type": "BUY", "quantity": 150, "price": 45000, "date": "2024-02-01"},
        ]
        
        for trade in banking_trades:
            self.add_transaction(banking_id, **trade)
        
        # Danh mục cổ phiếu công nghệ
        tech_id = self.create_portfolio(
            name="Danh mục Công nghệ",
            description="Đầu tư vào các cổ phiếu công nghệ và viễn thông",
            initial_cash=50000000,  # 50 triệu
            strategy="Growth Investing - Technology"
        )
        
        # Thêm giao dịch mẫu cho danh mục công nghệ
        tech_trades = [
            {"symbol": "FPT", "type": "BUY", "quantity": 100, "price": 120000, "date": "2024-01-10"},
            {"symbol": "CMG", "type": "BUY", "quantity": 200, "price": 35000, "date": "2024-01-25"},
        ]
        
        for trade in tech_trades:
            self.add_transaction(tech_id, **trade)
        
        # Danh mục cổ phiếu tiêu dùng
        consumer_id = self.create_portfolio(
            name="Danh mục Tiêu dùng",
            description="Đầu tư vào các cổ phiếu hàng tiêu dùng và thực phẩm",
            initial_cash=75000000,  # 75 triệu
            strategy="Dividend Investing - Consumer Goods"
        )
        
        # Thêm giao dịch mẫu cho danh mục tiêu dùng
        consumer_trades = [
            {"symbol": "VNM", "type": "BUY", "quantity": 200, "price": 85000, "date": "2024-01-05"},
            {"symbol": "MSN", "type": "BUY", "quantity": 100, "price": 125000, "date": "2024-01-30"},
        ]
        
        for trade in consumer_trades:
            self.add_transaction(consumer_id, **trade)
        
        print("✅ Đã tạo 3 danh mục giao dịch mẫu với các giao dịch demo")
