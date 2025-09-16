"""
🧪 TEST MAIN SYSTEM - Test hệ thống chính
=======================================

Test cases cho StockAnalysisSystem
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from src.main import StockAnalysisSystem
from src.utils.config import config

class TestStockAnalysisSystem:
    """
    Test cases for StockAnalysisSystem
    """
    
    @pytest.fixture
    def system(self):
        """
        Fixture để tạo instance của system
        """
        return StockAnalysisSystem()
    
    @pytest.fixture
    def sample_data(self):
        """
        Fixture tạo dữ liệu mẫu
        """
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 500)
        high_prices = close_prices + np.random.rand(100) * 1000
        low_prices = close_prices - np.random.rand(100) * 1000
        open_prices = close_prices + np.random.randn(100) * 200
        volumes = np.random.randint(100000, 1000000, 100)
        
        return pd.DataFrame({
            'time': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
    
    def test_system_initialization(self, system):
        """
        Test khởi tạo hệ thống
        """
        assert system is not None
        assert hasattr(system, 'data_provider')
        assert hasattr(system, 'indicators')
        assert hasattr(system, 'signals')
        assert hasattr(system, 'charts')
    
    def test_get_stock_info(self, system):
        """
        Test lấy thông tin cổ phiếu
        """
        # Test với symbol hợp lệ
        info = system.get_stock_info("VCB")
        
        if info:  # Chỉ test nếu có kết nối mạng
            assert isinstance(info, dict)
            assert 'current_price' in info or 'error' in info
    
    @pytest.mark.slow
    def test_get_stock_data(self, system):
        """
        Test lấy dữ liệu cổ phiếu (slow test)
        """
        df = system.get_stock_data("VCB", "1mo")
        
        if df is not None:  # Chỉ test nếu có kết nối
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_calculate_indicators(self, system, sample_data):
        """
        Test tính toán chỉ báo
        """
        df_with_indicators = system.calculate_indicators(sample_data)
        
        assert isinstance(df_with_indicators, pd.DataFrame)
        assert len(df_with_indicators) == len(sample_data)
        
        # Check some indicators exist
        expected_indicators = ['sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_generate_signals(self, system, sample_data):
        """
        Test tạo tín hiệu
        """
        # Calculate indicators first
        df_with_indicators = system.calculate_indicators(sample_data)
        df_with_signals = system.generate_signals(df_with_indicators)
        
        assert isinstance(df_with_signals, pd.DataFrame)
        assert 'signal_type' in df_with_signals.columns
        assert 'signal_score' in df_with_signals.columns
        assert 'signal_strength' in df_with_signals.columns
    
    def test_create_chart(self, system, sample_data):
        """
        Test tạo biểu đồ
        """
        df_with_indicators = system.calculate_indicators(sample_data)
        fig = system.create_chart(df_with_indicators, "TEST", ["rsi"])
        
        # Chart có thể None nếu plotly không có
        if fig is not None:
            assert hasattr(fig, 'data')
    
    def test_invalid_symbol(self, system):
        """
        Test với symbol không hợp lệ
        """
        df = system.get_stock_data("INVALID_SYMBOL", "1mo")
        assert df is None or len(df) == 0
    
    def test_empty_data(self, system):
        """
        Test với dữ liệu rỗng
        """
        empty_df = pd.DataFrame()
        result = system.calculate_indicators(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

# Integration tests
class TestSystemIntegration:
    """
    Test tích hợp các component
    """
    
    @pytest.fixture
    def system(self):
        return StockAnalysisSystem()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_analysis_workflow(self, system):
        """
        Test quy trình phân tích hoàn chỉnh
        """
        symbol = "VCB"
        period = "1mo"
        
        # 1. Get data
        df = system.get_stock_data(symbol, period)
        
        if df is None or len(df) == 0:
            pytest.skip("Không có dữ liệu để test")
        
        # 2. Calculate indicators
        df_indicators = system.calculate_indicators(df)
        assert len(df_indicators) == len(df)
        
        # 3. Generate signals
        df_signals = system.generate_signals(df_indicators)
        assert len(df_signals) == len(df)
        
        # 4. Create chart
        fig = system.create_chart(df_signals, symbol)
        # Chart creation might fail without plotly
        
        # Verify final data has all components
        assert 'close' in df_signals.columns
        assert 'rsi' in df_signals.columns
        assert 'signal_type' in df_signals.columns

# Performance tests
class TestSystemPerformance:
    """
    Test hiệu năng hệ thống
    """
    
    @pytest.fixture
    def large_dataset(self):
        """
        Tạo dataset lớn để test performance
        """
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        close_prices = 50000 + np.cumsum(np.random.randn(1000) * 500)
        high_prices = close_prices + np.random.rand(1000) * 1000
        low_prices = close_prices - np.random.rand(1000) * 1000
        open_prices = close_prices + np.random.randn(1000) * 200
        volumes = np.random.randint(100000, 1000000, 1000)
        
        return pd.DataFrame({
            'time': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
    
    def test_large_dataset_performance(self, large_dataset):
        """
        Test performance với dataset lớn
        """
        system = StockAnalysisSystem()
        
        start_time = datetime.now()
        df_indicators = system.calculate_indicators(large_dataset)
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert calculation_time < 10  # 10 seconds max
        assert len(df_indicators) == len(large_dataset)
    
    def test_memory_usage(self, large_dataset):
        """
        Test memory usage
        """
        system = StockAnalysisSystem()
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        df_indicators = system.calculate_indicators(large_dataset)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase

if __name__ == "__main__":
    """
    Run tests directly
    """
    pytest.main([__file__, "-v"])
