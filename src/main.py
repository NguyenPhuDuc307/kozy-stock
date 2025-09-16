"""
📊 MAIN ENTRY POINT - Vietnam Stock Analysis System
=================================================

Hệ thống phân tích chứng khoán Việt Nam tổng hợp
Author: GitHub Copilot
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data.data_provider import DataProvider
from src.analysis.indicators import TechnicalIndicators
from src.analysis.signals import TradingSignals
from src.utils.config import config

class StockAnalysisSystem:
    """
    Hệ thống phân tích chứng khoán chính
    """
    
    def __init__(self):
        """
        Khởi tạo hệ thống
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Khởi tạo Vietnam Stock Analysis System")
        
        # Load config
        self.config = config
        
        # Initialize components
        self.data_provider = DataProvider(self.config)
        self.indicators = TechnicalIndicators()
        self.trading_signals = TradingSignals()
        
        self.logger.info("✅ Hệ thống đã được khởi tạo thành công")
    
    def get_stock_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """
        Lấy dữ liệu cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu (VD: 'VCB')
            period: Khoảng thời gian ('1mo', '3mo', '6mo', '1y', '2y')
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        try:
            self.logger.info(f"📊 Lấy dữ liệu cho {symbol}, period: {period}")
            
            # Calculate dates
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            if period == '1mo':
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == '3mo':
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif period == '6mo':
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif period == '1y':
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            elif period == '2y':
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Get data from data provider
            df = self.data_provider.get_historical_data(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                self.logger.info(f"✅ Lấy được {len(df)} records cho {symbol}")
            else:
                self.logger.warning(f"❌ Không thể lấy dữ liệu cho {symbol}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy dữ liệu {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Lấy thông tin cơ bản về cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            
        Returns:
            Dict chứa thông tin cơ bản
        """
        try:
            # Get company info
            company_info = self.data_provider.get_company_info(symbol)
            
            # Get realtime price
            price_info = self.data_provider.get_realtime_price(symbol)
            
            # Get recent data for additional metrics
            df = self.get_stock_data(symbol, '1mo')
            
            result = {}
            
            if company_info:
                result.update(company_info)
            
            if price_info:
                result.update(price_info)
            
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                result.update({
                    'current_price': latest['close'],
                    'volume': latest['volume'],
                    'high_52w': df['high'].max(),
                    'low_52w': df['low'].min(),
                })
                
                # Calculate change
                if len(df) > 1:
                    prev_close = df.iloc[-2]['close']
                    change = latest['close'] - prev_close
                    change_percent = (change / prev_close) * 100
                    result.update({
                        'change': change,
                        'change_percent': change_percent
                    })
            
            return result if result else None
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy thông tin {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Tính toán các chỉ báo kỹ thuật
        
        Args:
            df: DataFrame chứa dữ liệu OHLCV
            
        Returns:
            DataFrame với các chỉ báo đã được tính
        """
        try:
            if df is None or len(df) == 0:
                return None
                
            self.logger.info(f"📈 Tính toán chỉ báo kỹ thuật cho {len(df)} điểm dữ liệu")
            
            # Calculate all indicators
            df_with_indicators = self.indicators.calculate_all(df)
            
            self.logger.info(f"✅ Đã tính toán chỉ báo, tổng {len(df_with_indicators.columns)} cột")
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tính chỉ báo: {e}")
            return None
    
    def generate_signals(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Tạo tín hiệu giao dịch
        
        Args:
            df: DataFrame chứa dữ liệu và chỉ báo
            
        Returns:
            DataFrame với các tín hiệu giao dịch
        """
        try:
            if df is None or len(df) == 0:
                return None
                
            self.logger.info(f"🎯 Tạo tín hiệu giao dịch cho {len(df)} điểm dữ liệu")
            
            # Generate signals
            df_with_signals = self.trading_signals.generate_all_signals(df)
            
            self.logger.info(f"✅ Đã tạo tín hiệu giao dịch")
            return df_with_signals
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tạo tín hiệu: {e}")
            return None
    
    def create_chart(self, df: pd.DataFrame, symbol: str, indicators: List[str] = None) -> Optional[object]:
        """
        Tạo biểu đồ tương tác với Plotly
        
        Args:
            df: DataFrame chứa dữ liệu
            symbol: Mã cổ phiếu
            indicators: Danh sách indicators muốn hiển thị
            
        Returns:
            Plotly figure object hoặc None
        """
        try:
            if df is None or len(df) == 0:
                return None
                
            self.logger.info(f"📊 Tạo biểu đồ cho {symbol}")
            
            # Import chart generator
            from src.visualization.charts import InteractiveCharts
            chart_gen = InteractiveCharts()
            
            # Create candlestick chart
            fig = chart_gen.create_candlestick_chart(df, symbol, indicators or [])
            
            if fig is not None:
                self.logger.info(f"✅ Đã tạo biểu đồ cho {symbol}")
            else:
                self.logger.warning(f"⚠️ Không thể tạo biểu đồ cho {symbol}")
                
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tạo biểu đồ: {e}")
            return None
    
    def analyze_stock(self, symbol: str, period: str = '3mo') -> Dict:
        """
        Phân tích tổng hợp một cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            period: Khoảng thời gian
            
        Returns:
            Dict chứa kết quả phân tích tổng hợp
        """
        try:
            self.logger.info(f"🔍 Bắt đầu phân tích tổng hợp {symbol}")
            
            result = {
                'symbol': symbol,
                'period': period,
                'analysis_time': datetime.now().isoformat(),
                'success': False
            }
            
            # 1. Get stock data
            df = self.get_stock_data(symbol, period)
            if df is None:
                result['error'] = 'Không thể lấy dữ liệu'
                return result
            
            result['data_points'] = len(df)
            
            # 2. Get stock info
            info = self.get_stock_info(symbol)
            if info:
                result['stock_info'] = info
            
            # 3. Calculate indicators
            df_indicators = self.calculate_indicators(df)
            if df_indicators is None:
                result['error'] = 'Không thể tính chỉ báo'
                return result
            
            # 4. Generate signals
            df_signals = self.generate_signals(df_indicators)
            if df_signals is None:
                result['error'] = 'Không thể tạo tín hiệu'
                return result
            
            # 5. Extract latest analysis
            latest = df_signals.iloc[-1]
            
            result.update({
                'latest_price': latest['close'],
                'latest_data': latest.to_dict(),
                'indicators_count': len([col for col in df_indicators.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume']]),
                'success': True
            })
            
            # 6. Performance metrics
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_return = ((last_price / first_price) - 1) * 100
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * (252**0.5) * 100  # Annualized
            
            result['performance'] = {
                'total_return': total_return,
                'volatility': volatility,
                'max_price': df['high'].max(),
                'min_price': df['low'].min()
            }
            
            self.logger.info(f"✅ Hoàn thành phân tích {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi phân tích {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # Test
    system = StockAnalysisSystem()
    result = system.analyze_stock("VCB", "1mo")
    print(f"Kết quả phân tích: {result}")
