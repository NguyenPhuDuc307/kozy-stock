"""
📁 PORTFOLIO MANAGER - Quản lý danh mục cổ phiếu
===============================================

Module này quản lý các danh mục cổ phiếu có thể được thêm/sửa/xóa
"""

import json
import os
from typing import Dict, List, Optional
import streamlit as st

class PortfolioManager:
    """
    Lớp quản lý danh mục cổ phiếu
    """
    
    def __init__(self, config_path: str = "portfolios.json"):
        """
        Khởi tạo PortfolioManager
        
        Args:
            config_path: Đường dẫn đến file cấu hình danh mục
        """
        self.config_path = config_path
        self.portfolios = self._load_portfolios()
    
    def _load_portfolios(self) -> Dict[str, List[str]]:
        """
        Tải danh mục từ file hoặc tạo mặc định
        
        Returns:
            Dict chứa các danh mục cổ phiếu
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Tạo danh mục mặc định
                default_portfolios = {
                    "VN30": [
                        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", 
                        "HDB", "HPG", "KDH", "KHG", "MBB", "MSN", "MWG", "PLX", 
                        "POW", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", 
                        "VCB", "VHM", "VIC", "VJC", "VNM", "VPB"
                    ],
                    "VN50": [
                        "ACB", "BCM", "BID", "BVH", "CTG", "DGC", "DPM", "EIB", 
                        "FPT", "GAS", "GVR", "HDB", "HNG", "HPG", "KBC", "KDH", 
                        "KHG", "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", 
                        "REE", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", 
                        "VCB", "VGC", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE"
                    ],
                    "Ngân hàng": [
                        "ACB", "BID", "CTG", "EIB", "HDB", "MBB", "MSB", 
                        "NAB", "OCB", "SHB", "SSB", "STB", "TCB", "TPB", 
                        "VCB", "VIB", "VPB"
                    ],
                    "Bất động sản": [
                        "DIG", "DXG", "FLC", "HDC", "IDC", "IJC", "KBC", 
                        "KDH", "KHG", "NLG", "NVL", "PDR", "QCG", "SCR", 
                        "SJS", "TCH", "VHM", "VIC", "VRE"
                    ],
                    "Công nghệ": [
                        "CMG", "DTD", "ELC", "FPT", "ITD", "MFS", "SAM", 
                        "SFI", "SGT", "STG", "TTN", "VGI"
                    ],
                    "Dầu khí": [
                        "BSR", "CNG", "DCM", "DGC", "DPM", "GAS", "OIL", 
                        "PLC", "PLX", "PVC", "PVD", "PVS", "PVT"
                    ],
                    "Thép": [
                        "HPG", "HSG", "NKG", "POM", "SMC", "TIS", "VCA"
                    ],
                    "Tiêu dùng": [
                        "BBC", "BHN", "CII", "KDC", "MCH", "MSN", "MWG", 
                        "PET", "PNJ", "SAB", "SBT", "TNG", "VNM", "VTO"
                    ]
                }
                self._save_portfolios(default_portfolios)
                return default_portfolios
        except Exception as e:
            st.error(f"Lỗi khi tải danh mục: {e}")
            return {}
    
    def _save_portfolios(self, portfolios: Dict[str, List[str]]) -> None:
        """
        Lưu danh mục vào file
        
        Args:
            portfolios: Dict chứa các danh mục cổ phiếu
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(portfolios, f, ensure_ascii=False, indent=2)
            self.portfolios = portfolios
        except Exception as e:
            st.error(f"Lỗi khi lưu danh mục: {e}")
    
    def get_portfolios(self) -> Dict[str, List[str]]:
        """
        Lấy tất cả danh mục
        
        Returns:
            Dict chứa các danh mục cổ phiếu
        """
        return self.portfolios
    
    def get_portfolio_names(self) -> List[str]:
        """
        Lấy danh sách tên các danh mục
        
        Returns:
            List các tên danh mục
        """
        return list(self.portfolios.keys())
    
    def get_portfolio_stocks(self, portfolio_name: str) -> List[str]:
        """
        Lấy danh sách cổ phiếu trong một danh mục
        
        Args:
            portfolio_name: Tên danh mục
            
        Returns:
            List các mã cổ phiếu
        """
        return self.portfolios.get(portfolio_name, [])
    
    def add_portfolio(self, name: str, stocks: List[str]) -> bool:
        """
        Thêm danh mục mới
        
        Args:
            name: Tên danh mục
            stocks: Danh sách mã cổ phiếu
            
        Returns:
            True nếu thành công
        """
        try:
            if name in self.portfolios:
                st.warning(f"Danh mục '{name}' đã tồn tại!")
                return False
            
            # Chuẩn hóa mã cổ phiếu (uppercase, remove duplicates)
            cleaned_stocks = list(set([stock.upper().strip() for stock in stocks if stock.strip()]))
            
            self.portfolios[name] = cleaned_stocks
            self._save_portfolios(self.portfolios)
            st.success(f"✅ Đã thêm danh mục '{name}' với {len(cleaned_stocks)} cổ phiếu")
            return True
        except Exception as e:
            st.error(f"Lỗi khi thêm danh mục: {e}")
            return False
    
    def update_portfolio(self, name: str, stocks: List[str]) -> bool:
        """
        Cập nhật danh mục
        
        Args:
            name: Tên danh mục
            stocks: Danh sách mã cổ phiếu mới
            
        Returns:
            True nếu thành công
        """
        try:
            if name not in self.portfolios:
                st.error(f"Danh mục '{name}' không tồn tại!")
                return False
            
            # Chuẩn hóa mã cổ phiếu
            cleaned_stocks = list(set([stock.upper().strip() for stock in stocks if stock.strip()]))
            
            self.portfolios[name] = cleaned_stocks
            self._save_portfolios(self.portfolios)
            st.success(f"✅ Đã cập nhật danh mục '{name}' với {len(cleaned_stocks)} cổ phiếu")
            return True
        except Exception as e:
            st.error(f"Lỗi khi cập nhật danh mục: {e}")
            return False
    
    def delete_portfolio(self, name: str) -> bool:
        """
        Xóa danh mục
        
        Args:
            name: Tên danh mục
            
        Returns:
            True nếu thành công
        """
        try:
            if name not in self.portfolios:
                st.error(f"Danh mục '{name}' không tồn tại!")
                return False
            
            del self.portfolios[name]
            self._save_portfolios(self.portfolios)
            st.success(f"✅ Đã xóa danh mục '{name}'")
            return True
        except Exception as e:
            st.error(f"Lỗi khi xóa danh mục: {e}")
            return False
    
    def add_stock_to_portfolio(self, portfolio_name: str, stock: str) -> bool:
        """
        Thêm cổ phiếu vào danh mục
        
        Args:
            portfolio_name: Tên danh mục
            stock: Mã cổ phiếu
            
        Returns:
            True nếu thành công
        """
        try:
            if portfolio_name not in self.portfolios:
                st.error(f"Danh mục '{portfolio_name}' không tồn tại!")
                return False
            
            stock = stock.upper().strip()
            if stock in self.portfolios[portfolio_name]:
                st.warning(f"Cổ phiếu '{stock}' đã có trong danh mục!")
                return False
            
            self.portfolios[portfolio_name].append(stock)
            self._save_portfolios(self.portfolios)
            st.success(f"✅ Đã thêm '{stock}' vào danh mục '{portfolio_name}'")
            return True
        except Exception as e:
            st.error(f"Lỗi khi thêm cổ phiếu: {e}")
            return False
    
    def remove_stock_from_portfolio(self, portfolio_name: str, stock: str) -> bool:
        """
        Xóa cổ phiếu khỏi danh mục
        
        Args:
            portfolio_name: Tên danh mục
            stock: Mã cổ phiếu
            
        Returns:
            True nếu thành công
        """
        try:
            if portfolio_name not in self.portfolios:
                st.error(f"Danh mục '{portfolio_name}' không tồn tại!")
                return False
            
            stock = stock.upper().strip()
            if stock not in self.portfolios[portfolio_name]:
                st.warning(f"Cổ phiếu '{stock}' không có trong danh mục!")
                return False
            
            self.portfolios[portfolio_name].remove(stock)
            self._save_portfolios(self.portfolios)
            st.success(f"✅ Đã xóa '{stock}' khỏi danh mục '{portfolio_name}'")
            return True
        except Exception as e:
            st.error(f"Lỗi khi xóa cổ phiếu: {e}")
            return False
    
    def get_all_stocks(self) -> List[str]:
        """
        Lấy tất cả mã cổ phiếu từ tất cả danh mục
        
        Returns:
            List tất cả mã cổ phiếu (không trùng lặp)
        """
        all_stocks = set()
        for stocks in self.portfolios.values():
            all_stocks.update(stocks)
        return sorted(list(all_stocks))
