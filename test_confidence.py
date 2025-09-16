#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.utils.unified_config import UnifiedSignalAnalyzer, UnifiedConfig, TimeFrame
import pandas as pd
import numpy as np

def test_confidence_algorithm():
    print("🧪 Testing confidence algorithm...")
    
    analyzer = UnifiedSignalAnalyzer(TimeFrame.SHORT_TERM)

    # Test case 1: Trending upward data
    print("=" * 60)
    print("📈 Test case 1: Strong upward trend")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    upward_prices = [100 + i * 1.2 + np.random.normal(0, 0.5) for i in range(50)]  # Strong uptrend
    volumes = [1000000 + np.random.randint(-200000, 200000) for _ in range(50)]

    df_up = pd.DataFrame({
        'date': dates,
        'close': upward_prices,
        'volume': volumes
    })

    result_up = analyzer.analyze_comprehensive_signal(df_up)
    print(f"Signal: {result_up['signal']}")
    print(f"Confidence: {result_up['confidence']:.2%}")
    print(f"Reasons: {result_up['reasons']}")

    # Test case 2: Trending downward data  
    print("=" * 60)
    print("📉 Test case 2: Strong downward trend")
    downward_prices = [150 - i * 1.5 + np.random.normal(0, 0.8) for i in range(50)]  # Strong downtrend
    
    df_down = pd.DataFrame({
        'date': dates,
        'close': downward_prices,
        'volume': volumes
    })

    result_down = analyzer.analyze_comprehensive_signal(df_down)
    print(f"Signal: {result_down['signal']}")
    print(f"Confidence: {result_down['confidence']:.2%}")
    print(f"Reasons: {result_down['reasons']}")

    # Test case 3: Sideways market
    print("=" * 60)
    print("� Test case 3: Sideways market")
    sideways_prices = [120 + np.random.normal(0, 1) for _ in range(50)]  # Sideways
    
    df_sideways = pd.DataFrame({
        'date': dates,
        'close': sideways_prices,
        'volume': volumes
    })

    result_sideways = analyzer.analyze_comprehensive_signal(df_sideways)
    print(f"Signal: {result_sideways['signal']}")
    print(f"Confidence: {result_sideways['confidence']:.2%}")
    print(f"Reasons: {result_sideways['reasons']}")

    # Test case 4: High volatility
    print("=" * 60)
    print("🎢 Test case 4: High volatility market")
    volatile_prices = [120 + i * 0.1 + np.random.normal(0, 8) for i in range(50)]  # High volatility
    
    df_volatile = pd.DataFrame({
        'date': dates,
        'close': volatile_prices,
        'volume': volumes
    })

    result_volatile = analyzer.analyze_comprehensive_signal(df_volatile)
    print(f"Signal: {result_volatile['signal']}")
    print(f"Confidence: {result_volatile['confidence']:.2%}")
    print(f"Reasons: {result_volatile['reasons']}")
    
    print("=" * 60)
    print("✅ All test cases completed!")

if __name__ == "__main__":
    test_confidence_algorithm()