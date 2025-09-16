"""
🔧 CONFIGURATION MANAGER - Quản lý cấu hình hệ thống
=================================================

Module này quản lý tất cả cấu hình của hệ thống phân tích chứng khoán
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    Quản lý cấu hình hệ thống
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Khởi tạo ConfigManager
        
        Args:
            config_file: Đường dẫn file cấu hình
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Load config from file if exists
        if os.path.exists(config_file):
            self.load_config()
        else:
            self.save_config()
        
        self.logger.info(f"🔧 ConfigManager đã được khởi tạo với file: {config_file}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Tải cấu hình mặc định
        
        Returns:
            Dict cấu hình mặc định
        """
        return {
            # API Configuration
            "api": {
                "vnstock": {
                    "base_url": "https://finfo-api.vndirect.com.vn",
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 1
                }
            },
            
            # Data Configuration
            "data": {
                "cache_enabled": True,
                "cache_duration_hours": 24,
                "cache_directory": "cache",
                "default_period": "1y",
                "default_interval": "1d",
                "supported_symbols": [
                    "VCB", "BID", "CTG", "VPB", "TCB", "MBB", "TPB", "LPB", "VIB", "SHB",
                    "VNM", "SAB", "MSN", "MWG", "VRE", "VIC", "VHM", "PDR", "NVL", "KDH",
                    "HPG", "HSG", "NKG", "TLG", "VGC", "DXG", "POM", "DCM", "DPM", "NT2",
                    "GAS", "PLX", "BSR", "PVS", "PVD", "PVC", "PVB", "CIG", "OIL", "PSH",
                    "CTD", "BCM", "HBC", "LCG", "CRE", "IJC", "SCS", "PC1", "C32", "TV2",
                    "REE", "POW", "GEG", "SBA", "QCG", "BWE", "TDM", "VSH", "HNG", "PC1"
                ]
            },
            
            # Technical Analysis Configuration
            "technical_analysis": {
                "indicators": {
                    "moving_averages": {
                        "sma_periods": [5, 10, 20, 50, 100, 200],
                        "ema_periods": [5, 10, 12, 20, 26, 50],
                        "wma_periods": [20]
                    },
                    "momentum": {
                        "rsi_period": 14,
                        "stochastic_k": 14,
                        "stochastic_d": 3,
                        "williams_r_period": 14,
                        "roc_period": 12
                    },
                    "trend": {
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "macd_signal": 9,
                        "adx_period": 14,
                        "parabolic_sar": {
                            "af_start": 0.02,
                            "af_increment": 0.02,
                            "af_max": 0.2
                        }
                    },
                    "volatility": {
                        "bollinger_period": 20,
                        "bollinger_std": 2,
                        "atr_period": 14,
                        "keltner_period": 20,
                        "keltner_multiplier": 2
                    },
                    "volume": {
                        "volume_ma_periods": [10, 20, 50],
                        "volume_roc_period": 12
                    }
                },
                "signals": {
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "stoch_oversold": 20,
                    "stoch_overbought": 80,
                    "bb_squeeze_threshold": 0.2
                }
            },
            
            # Backtesting Configuration
            "backtesting": {
                "initial_capital": 100000000,  # 100M VND
                "commission": 0.0015,  # 0.15%
                "slippage": 0.001,  # 0.1%
                "position_sizing": {
                    "max_position_size": 0.2,  # Max 20% per position
                    "risk_per_trade": 0.02  # Max 2% risk per trade
                },
                "risk_management": {
                    "max_drawdown": 0.2,  # 20%
                    "stop_loss": 0.05,  # 5%
                    "take_profit": 0.1  # 10%
                }
            },
            
            # Charting Configuration
            "charting": {
                "default_height": 800,
                "colors": {
                    "bullish": "#26a69a",
                    "bearish": "#ef5350",
                    "volume_up": "green",
                    "volume_down": "red",
                    "ma_colors": ["blue", "orange", "red", "purple", "brown"],
                    "bb_color": "rgba(173,204,255,0.5)"
                },
                "indicators_to_show": ["sma_20", "sma_50", "rsi", "macd"],
                "template": "plotly_white"
            },
            
            # Logging Configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_enabled": True,
                "file_path": "logs/stock_analysis.log",
                "max_file_size": "10MB",
                "backup_count": 5
            },
            
            # Web App Configuration
            "web_app": {
                "title": "Hệ thống phân tích chứng khoán Việt Nam",
                "page_icon": "📊",
                "layout": "wide",
                "sidebar_state": "expanded",
                "theme": {
                    "primaryColor": "#FF6B6B",
                    "backgroundColor": "#FFFFFF",
                    "secondaryBackgroundColor": "#F0F2F6",
                    "textColor": "#262730"
                }
            },
            
            # Performance Configuration
            "performance": {
                "max_data_points": 10000,
                "parallel_processing": True,
                "max_workers": 4,
                "memory_limit_mb": 1024
            }
        }
    
    def load_config(self) -> bool:
        """
        Tải cấu hình từ file
        
        Returns:
            True nếu tải thành công
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            
            self.logger.info(f"✅ Đã tải cấu hình từ {self.config_file}")
            return True
            
        except FileNotFoundError:
            self.logger.warning(f"⚠️ File cấu hình {self.config_file} không tồn tại")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Lỗi parse JSON: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tải cấu hình: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Lưu cấu hình ra file
        
        Returns:
            True nếu lưu thành công
        """
        try:
            # Tạo thư mục nếu chưa có
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Đã lưu cấu hình vào {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi lưu cấu hình: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị cấu hình
        
        Args:
            key: Key cấu hình (dạng dot notation: "api.vnstock.timeout")
            default: Giá trị mặc định
            
        Returns:
            Giá trị cấu hình
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Đặt giá trị cấu hình
        
        Args:
            key: Key cấu hình (dạng dot notation)
            value: Giá trị mới
            
        Returns:
            True nếu đặt thành công
        """
        keys = key.split('.')
        config = self.config
        
        try:
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            self.logger.info(f"✅ Đã cập nhật cấu hình: {key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi đặt cấu hình {key}: {e}")
            return False
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình API
        
        Returns:
            Dict cấu hình API
        """
        return self.get("api", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình dữ liệu
        
        Returns:
            Dict cấu hình dữ liệu
        """
        return self.get("data", {})
    
    def get_technical_analysis_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình phân tích kỹ thuật
        
        Returns:
            Dict cấu hình phân tích kỹ thuật
        """
        return self.get("technical_analysis", {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình backtesting
        
        Returns:
            Dict cấu hình backtesting
        """
        return self.get("backtesting", {})
    
    def get_charting_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình biểu đồ
        
        Returns:
            Dict cấu hình biểu đồ
        """
        return self.get("charting", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình logging
        
        Returns:
            Dict cấu hình logging
        """
        return self.get("logging", {})
    
    def get_web_app_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình web app
        
        Returns:
            Dict cấu hình web app
        """
        return self.get("web_app", {})
    
    def get_supported_symbols(self) -> list:
        """
        Lấy danh sách mã cổ phiếu được hỗ trợ
        
        Returns:
            List mã cổ phiếu
        """
        return self.get("data.supported_symbols", [])
    
    def is_cache_enabled(self) -> bool:
        """
        Kiểm tra cache có được bật không
        
        Returns:
            True nếu cache được bật
        """
        return self.get("data.cache_enabled", False)
    
    def get_cache_duration(self) -> int:
        """
        Lấy thời gian cache (giờ)
        
        Returns:
            Số giờ cache
        """
        return self.get("data.cache_duration_hours", 24)
    
    def get_initial_capital(self) -> float:
        """
        Lấy vốn ban đầu cho backtesting
        
        Returns:
            Vốn ban đầu (VND)
        """
        return self.get("backtesting.initial_capital", 100000000)
    
    def get_commission(self) -> float:
        """
        Lấy phí giao dịch
        
        Returns:
            Phí giao dịch (%)
        """
        return self.get("backtesting.commission", 0.0015)
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Cập nhật nhiều cấu hình cùng lúc
        
        Args:
            updates: Dict các cấu hình cần cập nhật
            
        Returns:
            True nếu cập nhật thành công
        """
        try:
            for key, value in updates.items():
                self.set(key, value)
            
            # Save to file
            self.save_config()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi cập nhật cấu hình: {e}")
            return False
    
    def reset_to_default(self) -> bool:
        """
        Reset về cấu hình mặc định
        
        Returns:
            True nếu reset thành công
        """
        try:
            self.config = self._load_default_config()
            self.save_config()
            self.logger.info("✅ Đã reset về cấu hình mặc định")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi reset cấu hình: {e}")
            return False
    
    def export_config(self, export_path: str) -> bool:
        """
        Xuất cấu hình ra file khác
        
        Args:
            export_path: Đường dẫn file xuất
            
        Returns:
            True nếu xuất thành công
        """
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Đã xuất cấu hình ra {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi xuất cấu hình: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """
        Nhập cấu hình từ file khác
        
        Args:
            import_path: Đường dẫn file nhập
            
        Returns:
            True nếu nhập thành công
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
                self.config.update(imported_config)
            
            self.save_config()
            self.logger.info(f"✅ Đã nhập cấu hình từ {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi nhập cấu hình: {e}")
            return False
    
    def validate_config(self) -> bool:
        """
        Kiểm tra tính hợp lệ của cấu hình
        
        Returns:
            True nếu cấu hình hợp lệ
        """
        required_keys = [
            "api.vnstock.timeout",
            "data.cache_enabled",
            "technical_analysis.indicators",
            "backtesting.initial_capital",
            "logging.level"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                self.logger.error(f"❌ Thiếu cấu hình bắt buộc: {key}")
                return False
        
        # Validate specific values
        if self.get("backtesting.initial_capital", 0) <= 0:
            self.logger.error("❌ Vốn ban đầu phải > 0")
            return False
        
        if not 0 <= self.get("backtesting.commission", 0) <= 1:
            self.logger.error("❌ Phí giao dịch phải từ 0-1")
            return False
        
        self.logger.info("✅ Cấu hình hợp lệ")
        return True

# Global config instance
config = ConfigManager()

def load_config(config_file: str = "config.json") -> ConfigManager:
    """
    Load configuration from file
    
    Args:
        config_file: Path to config file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_file)

def get_config() -> ConfigManager:
    """
    Get global config instance
    
    Returns:
        Global ConfigManager instance
    """
    return config

# Test module
if __name__ == "__main__":
    """
    Test ConfigManager
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing ConfigManager...")
    
    # Test basic operations
    test_config = ConfigManager("test_config.json")
    
    # Test get/set
    print(f"✅ Initial capital: {test_config.get_initial_capital():,.0f}")
    print(f"✅ Commission: {test_config.get_commission():.4f}")
    print(f"✅ Cache enabled: {test_config.is_cache_enabled()}")
    
    # Test update
    test_config.set("backtesting.initial_capital", 200000000)
    print(f"✅ Updated capital: {test_config.get_initial_capital():,.0f}")
    
    # Test validation
    is_valid = test_config.validate_config()
    print(f"✅ Config valid: {is_valid}")
    
    # Test supported symbols
    symbols = test_config.get_supported_symbols()
    print(f"✅ Supported symbols: {len(symbols)} symbols")
    
    # Cleanup test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("✅ Test completed!")
