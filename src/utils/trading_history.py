"""
📈 TRADING HISTORY MANAGER
=========================

Quản lý lịch sử mua bán cổ phiếu để theo dõi danh mục đầu tư
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class TradingHistory:
    """
    Quản lý lịch sử giao dịch mua bán cổ phiếu
    """
    
    def __init__(self, data_file: str = "trading_history.json"):
        """
        Khởi tạo trading history manager
        
        Args:
            data_file: Tên file lưu trữ lịch sử giao dịch
        """
        self.data_file = data_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Tải lịch sử giao dịch từ file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"transactions": [], "holdings": {}}
        else:
            return {"transactions": [], "holdings": {}}
    
    def _save_history(self):
        """Lưu lịch sử giao dịch vào file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi lưu file: {e}")
    
    def add_transaction(self, symbol: str, transaction_type: str, 
                       quantity: int, price: float, date: str = None,
                       fee: float = 0.0, note: str = ""):
        """
        Thêm giao dịch mua/bán
        
        Args:
            symbol: Mã cổ phiếu
            transaction_type: 'BUY' hoặc 'SELL'
            quantity: Số lượng cổ phiếu
            price: Giá giao dịch
            date: Ngày giao dịch (nếu None thì dùng ngày hiện tại)
            fee: Phí giao dịch
            note: Ghi chú
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        transaction = {
            "id": len(self.history["transactions"]) + 1,
            "symbol": symbol.upper(),
            "type": transaction_type.upper(),
            "quantity": quantity,
            "price": price,
            "total_value": quantity * price,
            "fee": fee,
            "net_value": quantity * price + fee,
            "date": date,
            "note": note
        }
        
        # Thêm giao dịch vào lịch sử
        self.history["transactions"].append(transaction)
        
        # Cập nhật holdings
        self._update_holdings(symbol, transaction_type, quantity, price, fee)
        
        # Lưu file
        self._save_history()
        
        return transaction["id"]
    
    def _update_holdings(self, symbol: str, transaction_type: str, 
                        quantity: int, price: float, fee: float):
        """Cập nhật số dư cổ phiếu đang nắm giữ"""
        symbol = symbol.upper()
        
        if symbol not in self.history["holdings"]:
            self.history["holdings"][symbol] = {
                "total_shares": 0,
                "total_cost": 0.0,
                "avg_price": 0.0,
                "transactions": []
            }
        
        holding = self.history["holdings"][symbol]
        
        if transaction_type.upper() == "BUY":
            # Mua thêm
            old_total_cost = holding["total_cost"]
            old_shares = holding["total_shares"]
            
            new_shares = old_shares + quantity
            new_cost = old_total_cost + (quantity * price) + fee
            
            holding["total_shares"] = new_shares
            holding["total_cost"] = new_cost
            holding["avg_price"] = new_cost / new_shares if new_shares > 0 else 0
            
        elif transaction_type.upper() == "SELL":
            # Bán bớt
            if holding["total_shares"] >= quantity:
                # Giảm số lượng cổ phiếu
                holding["total_shares"] -= quantity
                
                # Giảm cost theo tỷ lệ
                cost_ratio = quantity / (holding["total_shares"] + quantity)
                holding["total_cost"] = holding["total_cost"] * (1 - cost_ratio)
                
                # Cập nhật avg_price
                if holding["total_shares"] > 0:
                    holding["avg_price"] = holding["total_cost"] / holding["total_shares"]
                else:
                    holding["avg_price"] = 0
            else:
                print(f"Cảnh báo: Không đủ cổ phiếu {symbol} để bán")
        
        # Lưu chi tiết giao dịch vào holding
        holding["transactions"].append({
            "type": transaction_type,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_current_holdings(self) -> Dict:
        """Lấy danh sách cổ phiếu đang nắm giữ"""
        current_holdings = {}
        
        for symbol, holding in self.history["holdings"].items():
            if holding["total_shares"] > 0:
                current_holdings[symbol] = {
                    "shares": holding["total_shares"],
                    "avg_price": holding["avg_price"],
                    "total_cost": holding["total_cost"],
                    "transactions": holding["transactions"]
                }
        
        return current_holdings
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Lấy tóm tắt danh mục đầu tư dạng DataFrame"""
        holdings = self.get_current_holdings()
        
        if not holdings:
            return pd.DataFrame()
        
        portfolio_data = []
        for symbol, data in holdings.items():
            portfolio_data.append({
                "Symbol": symbol,
                "Shares": data["shares"],
                "Avg_Price": data["avg_price"],
                "Total_Cost": data["total_cost"],
                "Current_Value": 0,  # Sẽ được cập nhật từ API
                "Profit_Loss": 0,    # Sẽ được tính toán
                "Profit_Loss_Pct": 0  # Sẽ được tính toán
            })
        
        return pd.DataFrame(portfolio_data)
    
    def get_transactions_history(self, symbol: str = None) -> List[Dict]:
        """
        Lấy lịch sử giao dịch
        
        Args:
            symbol: Mã cổ phiếu (nếu None thì lấy tất cả)
        """
        transactions = self.history["transactions"]
        
        if symbol:
            symbol = symbol.upper()
            transactions = [t for t in transactions if t["symbol"] == symbol]
        
        return transactions
    
    def delete_transaction(self, transaction_id: int):
        """Xóa giao dịch theo ID"""
        # Tìm và xóa giao dịch
        transaction_to_delete = None
        for i, trans in enumerate(self.history["transactions"]):
            if trans["id"] == transaction_id:
                transaction_to_delete = self.history["transactions"].pop(i)
                break
        
        if transaction_to_delete:
            # Tính toán lại holdings
            self._recalculate_holdings()
            self._save_history()
            return True
        
        return False
    
    def clear_symbol_transactions(self, symbol: str):
        """
        Xóa tất cả giao dịch của một mã cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu cần xóa
        """
        symbol = symbol.upper()
        
        # Lọc bỏ tất cả giao dịch của symbol này
        original_count = len(self.history["transactions"])
        self.history["transactions"] = [
            trans for trans in self.history["transactions"] 
            if trans["symbol"] != symbol
        ]
        
        removed_count = original_count - len(self.history["transactions"])
        
        if removed_count > 0:
            # Tính toán lại holdings
            self._recalculate_holdings()
            self._save_history()
            return True
        
        return False
    
    def _recalculate_holdings(self):
        """Tính toán lại toàn bộ holdings từ lịch sử giao dịch"""
        # Reset holdings
        self.history["holdings"] = {}
        
        # Duyệt lại tất cả giao dịch
        for trans in self.history["transactions"]:
            self._update_holdings(
                trans["symbol"], 
                trans["type"], 
                trans["quantity"], 
                trans["price"], 
                trans["fee"]
            )
    
    def get_stock_performance(self, symbol: str, current_price: float) -> Dict:
        """
        Tính toán hiệu suất của một cổ phiếu cụ thể
        
        Args:
            symbol: Mã cổ phiếu
            current_price: Giá hiện tại
        """
        holdings = self.get_current_holdings()
        symbol = symbol.upper()
        
        if symbol not in holdings:
            return None
        
        data = holdings[symbol]
        current_value = data["shares"] * current_price
        profit_loss = current_value - data["total_cost"]
        profit_loss_pct = (profit_loss / data["total_cost"]) * 100 if data["total_cost"] > 0 else 0
        
        return {
            "symbol": symbol,
            "shares": data["shares"],
            "avg_price": data["avg_price"],
            "current_price": current_price,
            "total_cost": data["total_cost"],
            "current_value": current_value,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        }
    
    def add_sample_data(self):
        """Thêm dữ liệu mẫu để demo"""
        sample_transactions = [
            {"symbol": "VNM", "type": "BUY", "quantity": 100, "price": 85000, "date": "2024-01-15"},
            {"symbol": "VCB", "type": "BUY", "quantity": 50, "price": 78000, "date": "2024-01-20"},
            {"symbol": "HPG", "type": "BUY", "quantity": 200, "price": 32000, "date": "2024-02-01"},
            {"symbol": "VNM", "type": "BUY", "quantity": 50, "price": 87000, "date": "2024-02-10"},
            {"symbol": "TCB", "type": "BUY", "quantity": 100, "price": 25000, "date": "2024-02-15"},
            {"symbol": "VCB", "type": "SELL", "quantity": 10, "price": 82000, "date": "2024-03-01"},
        ]
        
        for trans in sample_transactions:
            self.add_transaction(
                trans["symbol"], 
                trans["type"], 
                trans["quantity"], 
                trans["price"], 
                trans["date"]
            )
