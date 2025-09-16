"""
🔄 BACKTESTING ENGINE - Module kiểm tra lại hiệu quả chiến lược
=============================================================

Module này cung cấp framework để backtesting các chiế                # Kiểm tra tín                     # Tín                     # Tín hiệu mua
                    if action == 'buy' and current_position is None:
                        # Tính kích thước vị thế (20% vốn)
                        position_size = int((self.current_capital * 0.2) / current_price)
                        
                        if position_size > 0:
                            self.logger.info(f"💰 BUY signal: {position_size} shares at {current_price:.0f}")
                            # Vào lệnh
                            current_position = {
                                'quantity': position_size,
                                'entry_price': current_price,
                                'entry_date': date,
                                'action': 'buy'
                            }
                            
                            # Cập nhật vốn
                            self.current_capital -= position_size * current_price
                        else:
                            self.logger.warning(f"❌ Not enough capital for BUY")
                    
                    # Tín hiệu bán
                    elif action == 'sell' and current_position is not None:
                        self.logger.info(f"💸 SELL signal: {current_position['quantity']} shares at {current_price:.0f}")
                        # Đóng vị thế         if action == 'buy' and current_position is None:
                        self.logger.info(f"🔥 BUY signal at {date}: {current_price}")
                        # Tính kích thước vị thế (20% vốn)
                        position_size = int((self.current_capital * 0.2) / current_price)
                        
                        if position_size > 0:
                            # Vào lệnh
                            current_position = {
                                'quantity': position_size,
                                'entry_price': current_price,
                                'entry_date': date,
                                'action': 'buy'
                            }
                            
                            # Cập nhật vốn
                            self.current_capital -= position_size * current_price
                            self.logger.info(f"✅ Entered position: {position_size} shares @ {current_price}")
                    
                    # Tín hiệu bán
                    elif action == 'sell' and current_position is not None:
                        self.logger.info(f"🔥 SELL signal at {date}: {current_price}")            if i < len(signals):
                    # Xử lý signals (có thể là DataFrame hoặc list)
                    if hasattr(signals, 'iloc'):
                        signal = signals.iloc[i]
                        action = signal.get('action', 'hold')
                    else:
                        signal = signals[i]
                        # Xử lý StrategySignal object
                        if hasattr(signal, 'signal_type'):
                            action = signal.signal_type.lower()
                        elif hasattr(signal, 'action'):
                            action = signal.action
                        else:
                            action = 'hold'
                    
                    # Tín hiệu mua
                    if action == 'buy' and current_position is None:o dịch
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """Loại lệnh giao dịch"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Trạng thái lệnh"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    """Lệnh giao dịch"""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = ""
    commission: float = 0.0
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.order_type.value}"

@dataclass
class Position:
    """Vị thế giao dịch"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, new_price: float):
        """Cập nhật giá hiện tại và P&L"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity

@dataclass
class Trade:
    """Giao dịch hoàn chính (mua + bán)"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    commission: float
    return_pct: float
    duration_days: int
    
    def __post_init__(self):
        self.pnl = (self.exit_price - self.entry_price) * self.quantity - self.commission
        self.return_pct = (self.exit_price - self.entry_price) / self.entry_price * 100
        self.duration_days = (self.exit_time - self.entry_time).days

@dataclass
class PerformanceMetrics:
    """Metrics hiệu suất backtest"""
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0
    total_commission: float = 0.0

class BacktestingEngine:
    """
    Engine chính cho backtesting các chiến lược giao dịch
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,  # 100M VND
                 commission_rate: float = 0.0015,       # 0.15%
                 slippage: float = 0.001):              # 0.1%
        """
        Khởi tạo Backtesting Engine
        
        Args:
            initial_capital: Vốn ban đầu (VND)
            commission_rate: Tỷ lệ phí giao dịch
            slippage: Độ trượt giá
        """
        self.logger = logging.getLogger(__name__)
        
        # Cấu hình
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Trạng thái
        self.current_capital = initial_capital
        self.current_positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Lịch sử
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
        self.logger.info(f"🔄 BacktestingEngine khởi tạo với vốn {initial_capital:,.0f} VND")
    
    def reset(self):
        """Reset engine về trạng thái ban đầu"""
        self.current_capital = self.initial_capital
        self.current_positions.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        
        self.logger.info("🔄 Engine đã được reset")
    
    def backtest(self, strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Chạy backtest cho chiến lược với dữ liệu cho trước
        
        Args:
            strategy: Đối tượng chiến lược giao dịch
            data: DataFrame chứa dữ liệu giá
            
        Returns:
            Dict chứa kết quả backtest
        """
        try:
            # Khởi tạo kết quả
            results = {
                'trades': [],
                'portfolio_values': [],
                'drawdown': [],
                'performance_metrics': {}
            }
            
            # Theo dõi giá trị portfolio
            portfolio_values = []
            
            # Theo dõi vị thế
            current_position = None
            entry_signal = None
            
            # Tạo tín hiệu
            signals = strategy.generate_signals(data)
            
            # Debug: Kiểm tra signals
            self.logger.info(f"📊 Generated {len(signals)} signals from {len(data)} data points")
            if len(signals) > 0:
                self.logger.info(f"📊 First signal: {signals[0]}")
                # Đếm số tín hiệu mua/bán
                buy_signals = sum(1 for s in signals if hasattr(s, 'signal_type') and s.signal_type == 'BUY')
                sell_signals = sum(1 for s in signals if hasattr(s, 'signal_type') and s.signal_type == 'SELL')
                hold_signals = sum(1 for s in signals if hasattr(s, 'signal_type') and s.signal_type == 'HOLD')
                self.logger.info(f"📊 Buy: {buy_signals}, Sell: {sell_signals}, Hold: {hold_signals}")
            else:
                self.logger.warning("❌ No signals generated!")
            
            # Xử lý từng ngày
            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row['close']
                action = 'hold'  # Default action
                
                # Cập nhật giá trị portfolio
                portfolio_value = self.current_capital
                if current_position is not None:
                    portfolio_value += current_position['quantity'] * current_price
                
                portfolio_values.append({
                    'date': date,
                    'value': portfolio_value
                })
                
                # Kiểm tra tín hiệu giao dịch
                if i < len(signals):
                    # Xử lý signals (có thể là DataFrame hoặc list)
                    if hasattr(signals, 'iloc'):
                        signal = signals.iloc[i]
                        action = signal.get('action', 'hold')
                    else:
                        signal = signals[i]
                        # StrategySignal có thuộc tính signal_type
                        signal_type = getattr(signal, 'signal_type', 'HOLD')
                        action = signal_type.lower()  # 'BUY' -> 'buy', 'SELL' -> 'sell', 'HOLD' -> 'hold'
                        
                    # Debug action
                    if action != 'hold':
                        self.logger.info(f"📡 Day {i}: Signal {action} at price {current_price:.0f}")
                    
                    # Tín hiệu mua
                    if action == 'buy' and current_position is None:
                        # Tính kích thước vị thế (20% vốn)
                        position_size = int((self.current_capital * 0.2) / current_price)
                        
                        if position_size > 0:
                            # Vào lệnh
                            current_position = {
                                'quantity': position_size,
                                'entry_price': current_price,
                                'entry_date': date,
                                'action': 'buy'
                            }
                            
                            # Cập nhật vốn
                            self.current_capital -= position_size * current_price
                    
                    # Tín hiệu bán
                    elif action == 'sell' and current_position is not None:
                        # Đóng vị thế
                        exit_price = current_price
                        pnl = (exit_price - current_position['entry_price']) * current_position['quantity']
                        pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                        
                        # Tạo bản ghi giao dịch
                        duration_days = 0
                        try:
                            duration_days = (date - current_position['entry_date']).days
                        except:
                            duration_days = 1  # Default to 1 day if calculation fails
                        
                        trade = {
                            'entry_date': current_position['entry_date'],
                            'exit_date': date,
                            'action': current_position['action'],
                            'quantity': current_position['quantity'],
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration': duration_days,
                            'exit_reason': 'signal'
                        }
                        
                        results['trades'].append(trade)
                        
                        # Cập nhật vốn
                        self.current_capital += current_position['quantity'] * exit_price
                        current_position = None
                        entry_signal = None
            
            # Đóng vị thế cuối kỳ nếu có
            if current_position is not None:
                final_price = data.iloc[-1]['close']
                pnl = (final_price - current_position['entry_price']) * current_position['quantity']
                pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                
                # Tính duration an toàn
                duration_days = 0
                try:
                    duration_days = (data.index[-1] - current_position['entry_date']).days
                except:
                    duration_days = 1  # Default to 1 day if calculation fails
                
                trade = {
                    'entry_date': current_position['entry_date'],
                    'exit_date': data.index[-1],
                    'action': current_position['action'],
                    'quantity': current_position['quantity'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': final_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration': duration_days,
                    'exit_reason': 'end_of_data'
                }
                
                results['trades'].append(trade)
                self.current_capital += current_position['quantity'] * final_price
            
            results['portfolio_values'] = portfolio_values
            
            # Tính toán metrics hiệu suất
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                
                # Tổng lợi nhuận
                total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
                
                # Tỷ lệ thắng
                winning_trades = len(trades_df[trades_df['pnl'] > 0])
                total_trades = len(trades_df)
                win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                
                # Lợi nhuận trung bình
                avg_return = trades_df['pnl_pct'].mean()
                
                # Sharpe ratio đơn giản (giả sử risk-free rate = 5%)
                if trades_df['pnl_pct'].std() > 0:
                    sharpe_ratio = (avg_return - 5) / trades_df['pnl_pct'].std()
                else:
                    sharpe_ratio = 0.0
                
                # Max drawdown đơn giản
                portfolio_df = pd.DataFrame(portfolio_values)
                portfolio_df['peak'] = portfolio_df['value'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak'] * 100
                max_drawdown = portfolio_df['drawdown'].min()
                
                # Annual return (giả sử 252 ngày giao dịch/năm)
                days = len(data)
                if days > 0:
                    annual_return = ((self.current_capital / self.initial_capital) ** (252 / days) - 1) * 100
                else:
                    annual_return = 0.0
                
                results['performance_metrics'] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': abs(max_drawdown),
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'final_capital': self.current_capital
                }
            else:
                # Không có giao dịch nào
                results['performance_metrics'] = {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'final_capital': self.current_capital
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            return {
                'trades': [],
                'portfolio_values': [],
                'performance_metrics': {},
                'error': str(e)
            }
    
    def _calculate_portfolio_value(self, current_price: float, position: Optional[Dict] = None) -> float:
        """Tính giá trị portfolio hiện tại"""
        portfolio_value = self.current_capital
        
        if position is not None:
            position_value = position['quantity'] * current_price
            portfolio_value += position_value
        
        return portfolio_value
    
    def _calculate_position_size(self, price: float) -> int:
        """Tính kích thước vị thế (20% vốn)"""
        target_value = self.current_capital * 0.2
        return int(target_value / price)

class Portfolio:
    """
    Portfolio management for backtesting
    """
    
    def __init__(self, 
                 initial_capital: float,
                 position_size_method: str = 'fixed_percentage',
                 position_size_value: float = 0.1,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.1,
                 max_drawdown_pct: float = 0.2):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
            position_size_method: Method for position sizing
            position_size_value: Value for position sizing
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_drawdown_pct: Maximum drawdown before stopping
        """
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.position_size_method = position_size_method
        self.position_size_value = position_size_value
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        self.positions = {}
        self.trades = []
    
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate position size based on method
        
        Args:
            price: Current stock price
            
        Returns:
            Number of shares to buy
        """
        if self.position_size_method == 'fixed_percentage':
            # Fixed percentage of portfolio
            target_value = self.current_cash * self.position_size_value
            return int(target_value / price)
        
        elif self.position_size_method == 'kelly_criterion':
            # Simplified Kelly criterion (using position_size_value as Kelly %)
            target_value = self.current_cash * min(self.position_size_value, 0.25)  # Cap at 25%
            return int(target_value / price)
        
        elif self.position_size_method == 'risk_parity':
            # Risk parity approach
            target_value = self.current_cash * self.position_size_value
            return int(target_value / price)
        
        else:
            # Default to fixed percentage
            target_value = self.current_cash * 0.1
            return int(target_value / price)
    
    def calculate_portfolio_value(self, current_price: float, position: Optional[Dict] = None) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_price: Current stock price
            position: Current position dict
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.current_cash
        
        if position is not None:
            position_value = position['quantity'] * current_price
            portfolio_value += position_value
        
        return portfolio_value

if __name__ == "__main__":
    print("🔄 Testing BacktestingEngine...")
    
    # Tạo dữ liệu test
    import pandas as pd
    from datetime import datetime, timedelta
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'open': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print("✅ Test data created")
    
    # Test engine
    engine = BacktestingEngine(initial_capital=100_000_000)
    print("✅ BacktestingEngine created successfully!")
