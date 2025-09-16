"""
📊 TRADING STRATEGIES - Các chiến lược giao dịch
==============================================

Module chứa các chiến lược giao dịch phổ biến để backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .backtesting_engine import BacktestingEngine, OrderType

@dataclass
class StrategySignal:
    """Tín hiệu từ chiến lược"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    quantity: int = 0
    price: float = 0.0
    reason: str = ""

class TradingStrategy(ABC):
    """
    Base class cho tất cả chiến lược giao dịch
    """
    
    def __init__(self, name: str):
        """
        Khởi tạo chiến lược
        
        Args:
            name: Tên chiến lược
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.parameters = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu giao dịch từ dữ liệu
        
        Args:
            data: DataFrame chứa dữ liệu OHLCV và indicators
            
        Returns:
            List các StrategySignal
        """
        pass
    
    def set_parameters(self, **kwargs):
        """Cập nhật tham số chiến lược"""
        self.parameters.update(kwargs)
        self.logger.info(f"📝 Cập nhật tham số {self.name}: {kwargs}")

class MovingAverageCrossoverStrategy(TradingStrategy):
    """
    Chiến lược cắt MA (Moving Average Crossover)
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """
        Khởi tạo MA Crossover Strategy
        
        Args:
            fast_period: Chu kỳ MA nhanh
            slow_period: Chu kỳ MA chậm
        """
        super().__init__("MA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu khi MA nhanh cắt MA chậm
        """
        signals = []
        
        # Tính MA nếu chưa có
        if f'sma_{self.fast_period}' not in data.columns:
            data[f'sma_{self.fast_period}'] = data['close'].rolling(self.fast_period).mean()
        
        if f'sma_{self.slow_period}' not in data.columns:
            data[f'sma_{self.slow_period}'] = data['close'].rolling(self.slow_period).mean()
        
        fast_ma = data[f'sma_{self.fast_period}']
        slow_ma = data[f'sma_{self.slow_period}']
        
        # Tìm crossover points
        for i in range(1, len(data)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            
            prev_fast = fast_ma.iloc[i-1]
            prev_slow = slow_ma.iloc[i-1]
            curr_fast = fast_ma.iloc[i]
            curr_slow = slow_ma.iloc[i]
            
            # Golden Cross (MA nhanh cắt lên MA chậm)
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.7,
                    price=data['close'].iloc[i],
                    reason=f"Golden Cross: MA{self.fast_period} cắt lên MA{self.slow_period}"
                ))
            
            # Death Cross (MA nhanh cắt xuống MA chậm)
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.7,
                    price=data['close'].iloc[i],
                    reason=f"Death Cross: MA{self.fast_period} cắt xuống MA{self.slow_period}"
                ))
        
        return signals

class RSIStrategy(TradingStrategy):
    """
    Chiến lược RSI (Relative Strength Index)
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Khởi tạo RSI Strategy
        
        Args:
            period: Chu kỳ RSI
            oversold: Ngưỡng quá bán
            overbought: Ngưỡng quá mua
        """
        super().__init__("RSI")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu dựa trên RSI oversold/overbought
        """
        signals = []
        
        # Tính RSI nếu chưa có
        if 'rsi' not in data.columns:
            data['rsi'] = self._calculate_rsi(data['close'], self.period)
        
        rsi = data['rsi']
        
        for i in range(1, len(data)):
            if pd.isna(rsi.iloc[i]):
                continue
            
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            
            # RSI thoát khỏi vùng oversold
            if prev_rsi <= self.oversold and current_rsi > self.oversold:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.8,
                    price=data['close'].iloc[i],
                    reason=f"RSI thoát oversold: {current_rsi:.1f}"
                ))
            
            # RSI vào vùng overbought
            elif prev_rsi < self.overbought and current_rsi >= self.overbought:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.8,
                    price=data['close'].iloc[i],
                    reason=f"RSI overbought: {current_rsi:.1f}"
                ))
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Tính RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class MACDStrategy(TradingStrategy):
    """
    Chiến lược MACD (Moving Average Convergence Divergence)
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Khởi tạo MACD Strategy
        
        Args:
            fast: Chu kỳ EMA nhanh
            slow: Chu kỳ EMA chậm  
            signal: Chu kỳ EMA signal line
        """
        super().__init__("MACD")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu dựa trên MACD crossover
        """
        signals = []
        
        # Tính MACD nếu chưa có
        if 'macd' not in data.columns:
            self._calculate_macd(data)
        
        macd = data['macd']
        macd_signal = data['macd_signal']
        
        for i in range(1, len(data)):
            if pd.isna(macd.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                continue
            
            prev_macd = macd.iloc[i-1]
            prev_signal = macd_signal.iloc[i-1]
            curr_macd = macd.iloc[i]
            curr_signal = macd_signal.iloc[i]
            
            # MACD cắt lên signal line
            if prev_macd <= prev_signal and curr_macd > curr_signal:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.75,
                    price=data['close'].iloc[i],
                    reason=f"MACD Bullish Crossover: {curr_macd:.3f} > {curr_signal:.3f}"
                ))
            
            # MACD cắt xuống signal line
            elif prev_macd >= prev_signal and curr_macd < curr_signal:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.75,
                    price=data['close'].iloc[i],
                    reason=f"MACD Bearish Crossover: {curr_macd:.3f} < {curr_signal:.3f}"
                ))
        
        return signals
    
    def _calculate_macd(self, data: pd.DataFrame):
        """Tính MACD"""
        ema_fast = data['close'].ewm(span=self.fast).mean()
        ema_slow = data['close'].ewm(span=self.slow).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.signal).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

class MeanReversionStrategy(TradingStrategy):
    """
    Chiến lược Mean Reversion (Hồi quy về trung bình)
    """
    
    def __init__(self, period: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        Khởi tạo Mean Reversion Strategy
        
        Args:
            period: Chu kỳ tính moving average
            entry_threshold: Ngưỡng vào lệnh (số lần std dev)
            exit_threshold: Ngưỡng thoát lệnh (số lần std dev)
        """
        super().__init__("Mean_Reversion")
        self.period = period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu dựa trên mean reversion
        """
        signals = []
        
        # Tính moving average và standard deviation
        data['ma'] = data['close'].rolling(window=self.period).mean()
        data['std'] = data['close'].rolling(window=self.period).std()
        data['z_score'] = (data['close'] - data['ma']) / data['std']
        
        for i in range(1, len(data)):
            if pd.isna(data['z_score'].iloc[i]):
                continue
            
            current_z = data['z_score'].iloc[i]
            prev_z = data['z_score'].iloc[i-1]
            current_price = data['close'].iloc[i]
            
            # Giá quá thấp so với trung bình -> BUY
            if prev_z <= -self.entry_threshold and current_z > -self.entry_threshold:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.7,
                    price=current_price,
                    reason=f"Mean Reversion BUY: Z-Score {current_z:.2f} từ oversold"
                ))
            
            # Giá quá cao so với trung bình -> SELL
            elif prev_z >= self.entry_threshold and current_z < self.entry_threshold:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.7,
                    price=current_price,
                    reason=f"Mean Reversion SELL: Z-Score {current_z:.2f} từ overbought"
                ))
            
            # Thoát lệnh khi giá về gần trung bình
            elif abs(prev_z) > self.exit_threshold and abs(current_z) <= self.exit_threshold:
                # Tạo tín hiệu HOLD để đóng position
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='HOLD',
                    confidence=0.5,
                    price=current_price,
                    reason=f"Mean Reversion EXIT: Z-Score {current_z:.2f} về trung bình"
                ))
        
        return signals

class MomentumStrategy(TradingStrategy):
    """
    Chiến lược Momentum (Theo xu hướng)
    """
    
    def __init__(self, period: int = 14, threshold: float = 0.02):
        """
        Khởi tạo Momentum Strategy
        
        Args:
            period: Chu kỳ tính momentum
            threshold: Ngưỡng momentum để tạo tín hiệu
        """
        super().__init__("Momentum")
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu dựa trên momentum
        """
        signals = []
        
        # Tính momentum
        data['momentum'] = data['close'].pct_change(periods=self.period)
        data['momentum_sma'] = data['momentum'].rolling(window=5).mean()
        
        for i in range(1, len(data)):
            if pd.isna(data['momentum'].iloc[i]):
                continue
            
            current_momentum = data['momentum'].iloc[i]
            prev_momentum = data['momentum'].iloc[i-1]
            current_price = data['close'].iloc[i]
            
            # Momentum tăng mạnh -> BUY
            if prev_momentum <= self.threshold and current_momentum > self.threshold:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.75,
                    price=current_price,
                    reason=f"Momentum BUY: {current_momentum:.2%} tăng mạnh"
                ))
            
            # Momentum giảm mạnh -> SELL
            elif prev_momentum >= -self.threshold and current_momentum < -self.threshold:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.75,
                    price=current_price,
                    reason=f"Momentum SELL: {current_momentum:.2%} giảm mạnh"
                ))
        
        return signals

class BollingerBandsStrategy(TradingStrategy):
    """
    Chiến lược Bollinger Bands
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Khởi tạo Bollinger Bands Strategy
        
        Args:
            period: Chu kỳ moving average
            std_dev: Số lần độ lệch chuẩn
        """
        super().__init__("Bollinger_Bands")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu dựa trên Bollinger Bands
        """
        signals = []
        
        # Tính Bollinger Bands nếu chưa có
        if 'bb_upper' not in data.columns:
            self._calculate_bollinger_bands(data)
        
        close = data['close']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        bb_middle = data['bb_middle']
        
        for i in range(1, len(data)):
            if pd.isna(bb_upper.iloc[i]) or pd.isna(bb_lower.iloc[i]):
                continue
            
            current_price = close.iloc[i]
            prev_price = close.iloc[i-1]
            
            # Giá chạm lower band và bật lên
            if prev_price <= bb_lower.iloc[i-1] and current_price > bb_lower.iloc[i]:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='BUY',
                    confidence=0.6,
                    price=current_price,
                    reason=f"BB Bounce từ Lower Band: {current_price:.1f}"
                ))
            
            # Giá chạm upper band và quay xuống
            elif prev_price >= bb_upper.iloc[i-1] and current_price < bb_upper.iloc[i]:
                signals.append(StrategySignal(
                    timestamp=data.index[i],
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data.columns else 'UNKNOWN',
                    signal_type='SELL',
                    confidence=0.6,
                    price=current_price,
                    reason=f"BB Reversal từ Upper Band: {current_price:.1f}"
                ))
        
        return signals
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame):
        """Tính Bollinger Bands"""
        data['bb_middle'] = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        data['bb_upper'] = data['bb_middle'] + (std * self.std_dev)
        data['bb_lower'] = data['bb_middle'] - (std * self.std_dev)

class MultiSignalStrategy(TradingStrategy):
    """
    Chiến lược kết hợp nhiều tín hiệu
    """
    
    def __init__(self, strategies: List[TradingStrategy], min_signals: int = 2):
        """
        Khởi tạo Multi-Signal Strategy
        
        Args:
            strategies: List các chiến lược con
            min_signals: Số tín hiệu tối thiểu để trade
        """
        super().__init__("Multi_Signal")
        self.strategies = strategies
        self.min_signals = min_signals
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """
        Tạo tín hiệu bằng cách kết hợp các chiến lược con
        """
        all_signals = []
        
        # Thu thập tín hiệu từ tất cả strategies
        for strategy in self.strategies:
            strategy_signals = strategy.generate_signals(data.copy())
            all_signals.extend(strategy_signals)
        
        # Group signals by timestamp và type
        signal_groups = {}
        for signal in all_signals:
            key = (signal.timestamp, signal.signal_type)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Tạo combined signals
        combined_signals = []
        for (timestamp, signal_type), signals in signal_groups.items():
            if len(signals) >= self.min_signals:
                # Tính confidence trung bình
                avg_confidence = np.mean([s.confidence for s in signals])
                
                # Tạo combined signal
                combined_signals.append(StrategySignal(
                    timestamp=timestamp,
                    symbol=signals[0].symbol,
                    signal_type=signal_type,
                    confidence=min(avg_confidence * 1.2, 1.0),  # Boost confidence
                    price=signals[0].price,
                    reason=f"Multi-signal ({len(signals)} strategies): " + 
                           ", ".join([s.reason.split(':')[0] for s in signals])
                ))
        
        return combined_signals

class StrategyOptimizer:
    """
    Optimizer để tìm tham số tốt nhất cho chiến lược
    """
    
    def __init__(self, engine: BacktestingEngine):
        """
        Khởi tạo Strategy Optimizer
        
        Args:
            engine: BacktestingEngine instance
        """
        self.engine = engine
        self.logger = logging.getLogger(__name__)
    
    def optimize_ma_crossover(self, data: pd.DataFrame, 
                            fast_range: Tuple[int, int] = (5, 20),
                            slow_range: Tuple[int, int] = (20, 50)) -> Dict[str, Any]:
        """
        Optimize MA Crossover strategy parameters
        
        Args:
            data: Dữ liệu để test
            fast_range: Range cho fast MA
            slow_range: Range cho slow MA
            
        Returns:
            Dict kết quả optimization
        """
        best_params = {}
        best_return = -np.inf
        results = []
        
        for fast in range(fast_range[0], fast_range[1] + 1, 2):
            for slow in range(slow_range[0], slow_range[1] + 1, 5):
                if fast >= slow:
                    continue
                
                # Reset engine
                self.engine.reset()
                
                # Test strategy
                strategy = MovingAverageCrossoverStrategy(fast, slow)
                signals = strategy.generate_signals(data.copy())
                
                # Execute signals
                for signal in signals:
                    if signal.signal_type == 'BUY':
                        self.engine.place_order(
                            signal.timestamp, signal.symbol,
                            OrderType.BUY, 1000, signal.price
                        )
                    elif signal.signal_type == 'SELL':
                        self.engine.place_order(
                            signal.timestamp, signal.symbol,
                            OrderType.SELL, 1000, signal.price
                        )
                
                # Calculate performance
                metrics = self.engine.calculate_performance_metrics()
                
                results.append({
                    'fast_ma': fast,
                    'slow_ma': slow,
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate
                })
                
                if metrics.total_return > best_return:
                    best_return = metrics.total_return
                    best_params = {'fast': fast, 'slow': slow}
        
        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': pd.DataFrame(results)
        }

# Test module
if __name__ == "__main__":
    """
    Test các strategies
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Trading Strategies...")
    
    # Tạo dữ liệu test
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'symbol': 'TEST'
    }, index=dates)
    
    test_data['high'] = test_data['open'] + np.random.uniform(0.5, 2, len(dates))
    test_data['low'] = test_data['open'] - np.random.uniform(0.5, 2, len(dates))
    test_data['close'] = test_data['low'] + (test_data['high'] - test_data['low']) * np.random.random(len(dates))
    
    # Test MA Crossover Strategy
    print("\n📊 Testing MA Crossover Strategy...")
    ma_strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=20)
    ma_signals = ma_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(ma_signals)} MA signals")
    
    # Test RSI Strategy
    print("\n📊 Testing RSI Strategy...")
    rsi_strategy = RSIStrategy(period=14)
    rsi_signals = rsi_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(rsi_signals)} RSI signals")
    
    # Test MACD Strategy
    print("\n📊 Testing MACD Strategy...")
    macd_strategy = MACDStrategy()
    macd_signals = macd_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(macd_signals)} MACD signals")
    
    # Test Mean Reversion Strategy
    print("\n📊 Testing Mean Reversion Strategy...")
    mean_reversion_strategy = MeanReversionStrategy(period=20)
    mean_reversion_signals = mean_reversion_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(mean_reversion_signals)} Mean Reversion signals")
    
    # Test Momentum Strategy
    print("\n📊 Testing Momentum Strategy...")
    momentum_strategy = MomentumStrategy(period=14)
    momentum_signals = momentum_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(momentum_signals)} Momentum signals")
    
    # Test Bollinger Bands Strategy
    print("\n📊 Testing Bollinger Bands Strategy...")
    bb_strategy = BollingerBandsStrategy(period=20)
    bb_signals = bb_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(bb_signals)} Bollinger Bands signals")
    
    # Test Multi-Signal Strategy
    print("\n📊 Testing Multi-Signal Strategy...")
    multi_strategy = MultiSignalStrategy([ma_strategy, rsi_strategy, macd_strategy, mean_reversion_strategy], min_signals=2)
    multi_signals = multi_strategy.generate_signals(test_data.copy())
    print(f"✅ Generated {len(multi_signals)} Multi-Signal signals")
    
    print("✅ All strategy tests completed!")
