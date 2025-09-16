"""
🔄 BACKTESTING ENGINE - Module kiểm tra lại hiệu quả chiến lược
=============================================================

Module này cung cấp framework để backtesting các chiến lược giao dịch
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
    """Giao dịch hoàn chình (mua + bán)"""
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
    
    def place_order(self, 
                   timestamp: datetime,
                   symbol: str,
                   order_type: OrderType,
                   quantity: int,
                   price: float) -> Order:
        """
        Đặt lệnh giao dịch
        
        Args:
            timestamp: Thời gian đặt lệnh
            symbol: Mã cổ phiếu
            order_type: Loại lệnh (BUY/SELL)
            quantity: Số lượng
            price: Giá
            
        Returns:
            Order object
        """
        # Tính commission
        commission = abs(quantity * price * self.commission_rate)
        
        # Tạo order
        order = Order(
            timestamp=timestamp,
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            commission=commission
        )
        
        # Kiểm tra đủ tiền/cổ phiếu
        if order_type == OrderType.BUY:
            required_capital = quantity * price + commission
            if required_capital > self.current_capital:
                self.logger.warning(f"❌ Không đủ vốn để mua {symbol}")
                order.status = OrderStatus.CANCELLED
                return order
        
        elif order_type == OrderType.SELL:
            current_quantity = self.current_positions.get(symbol, Position(symbol, 0, 0, timestamp)).quantity
            if quantity > current_quantity:
                self.logger.warning(f"❌ Không đủ cổ phiếu {symbol} để bán")
                order.status = OrderStatus.CANCELLED
                return order
        
        self.pending_orders.append(order)
        self.logger.info(f"📝 Đặt lệnh {order_type.value} {quantity} {symbol} @ {price:,.0f}")
        
        return order
    
    def execute_pending_orders(self, current_data: pd.Series):
        """
        Thực hiện các lệnh đang chờ
        
        Args:
            current_data: Dữ liệu giá hiện tại (OHLCV)
        """
        timestamp = current_data.name if hasattr(current_data, 'name') else datetime.now()
        
        executed_orders = []
        
        for order in self.pending_orders:
            if order.status != OrderStatus.PENDING:
                continue
            
            # Áp dụng slippage
            execution_price = order.price
            if order.order_type == OrderType.BUY:
                execution_price = order.price * (1 + self.slippage)
            else:
                execution_price = order.price * (1 - self.slippage)
            
            # Kiểm tra có thể thực hiện với giá hiện tại không
            can_execute = False
            if order.order_type == OrderType.BUY:
                # Có thể mua nếu giá ask <= execution_price
                can_execute = current_data.get('low', current_data.get('close', 0)) <= execution_price
            else:
                # Có thể bán nếu giá bid >= execution_price  
                can_execute = current_data.get('high', current_data.get('close', 0)) >= execution_price
            
            if can_execute:
                # Thực hiện lệnh
                self._execute_order(order, execution_price, timestamp)
                executed_orders.append(order)
        
        # Xóa các lệnh đã thực hiện
        for order in executed_orders:
            self.pending_orders.remove(order)
    
    def _execute_order(self, order: Order, execution_price: float, timestamp: datetime):
        """
        Thực hiện lệnh giao dịch
        
        Args:
            order: Lệnh cần thực hiện
            execution_price: Giá thực hiện
            timestamp: Thời gian thực hiện
        """
        order.price = execution_price
        order.status = OrderStatus.FILLED
        self.filled_orders.append(order)
        
        if order.order_type == OrderType.BUY:
            self._execute_buy(order, timestamp)
        else:
            self._execute_sell(order, timestamp)
        
        self.logger.info(f"✅ Thực hiện {order.order_type.value} {order.quantity} {order.symbol} @ {execution_price:,.0f}")
    
    def _execute_buy(self, order: Order, timestamp: datetime):
        """Thực hiện lệnh mua"""
        total_cost = order.quantity * order.price + order.commission
        self.current_capital -= total_cost
        
        # Cập nhật position
        if order.symbol in self.current_positions:
            # Trung bình giá nếu đã có position
            current_pos = self.current_positions[order.symbol]
            total_quantity = current_pos.quantity + order.quantity
            avg_price = (current_pos.entry_price * current_pos.quantity + 
                        order.price * order.quantity) / total_quantity
            
            current_pos.quantity = total_quantity
            current_pos.entry_price = avg_price
        else:
            # Tạo position mới
            self.current_positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=order.price,
                entry_time=timestamp
            )
    
    def _execute_sell(self, order: Order, timestamp: datetime):
        """Thực hiện lệnh bán"""
        total_received = order.quantity * order.price - order.commission
        self.current_capital += total_received
        
        # Cập nhật position
        if order.symbol in self.current_positions:
            current_pos = self.current_positions[order.symbol]
            
            # Tạo trade record
            trade = Trade(
                symbol=order.symbol,
                entry_time=current_pos.entry_time,
                exit_time=timestamp,
                entry_price=current_pos.entry_price,
                exit_price=order.price,
                quantity=order.quantity,
                pnl=0,  # Sẽ được tính trong __post_init__
                commission=order.commission,
                return_pct=0,  # Sẽ được tính trong __post_init__
                duration_days=0  # Sẽ được tính trong __post_init__
            )
            self.trades.append(trade)
            
            # Giảm position
            current_pos.quantity -= order.quantity
            
            # Xóa position nếu hết cổ phiếu
            if current_pos.quantity <= 0:
                del self.current_positions[order.symbol]
    
    def update_positions(self, market_data: Dict[str, pd.Series]):
        """
        Cập nhật giá trị positions với dữ liệu thị trường
        
        Args:
            market_data: Dictionary {symbol: price_data}
        """
        for symbol, position in self.current_positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('close', position.current_price)
                position.update_price(current_price)
    
    def get_portfolio_value(self) -> float:
        """
        Tính tổng giá trị portfolio
        
        Returns:
            Tổng giá trị portfolio (tiền mặt + cổ phiếu)
        """
        portfolio_value = self.current_capital
        
        for position in self.current_positions.values():
            portfolio_value += position.quantity * position.current_price
        
        return portfolio_value
    
    def record_equity(self, timestamp: datetime):
        """
        Ghi lại giá trị equity tại thời điểm
        
        Args:
            timestamp: Thời gian
        """
        portfolio_value = self.get_portfolio_value()
        self.equity_curve.append((timestamp, portfolio_value))
        
        # Tính daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2][1]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Tính toán các metrics hiệu suất
        
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if not self.equity_curve:
            return metrics
        
        # Basic metrics
        final_value = self.equity_curve[-1][1]
        metrics.total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        if days > 0:
            metrics.annual_return = ((final_value / self.initial_capital) ** (365.25 / days) - 1) * 100
        
        # Volatility and Sharpe
        if self.daily_returns:
            daily_std = np.std(self.daily_returns)
            metrics.volatility = daily_std * np.sqrt(252) * 100  # Annualized
            
            avg_daily_return = np.mean(self.daily_returns)
            if daily_std > 0:
                metrics.sharpe_ratio = avg_daily_return / daily_std * np.sqrt(252)
        
        # Drawdown
        equity_values = [eq[1] for eq in self.equity_curve]
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        metrics.max_drawdown = np.max(drawdown) * 100
        
        # Trade statistics
        metrics.total_trades = len(self.trades)
        
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.win_rate = len(winning_trades) / len(self.trades) * 100
            
            if winning_trades:
                metrics.avg_win = np.mean([t.pnl for t in winning_trades])
            
            if losing_trades:
                metrics.avg_loss = np.mean([t.pnl for t in losing_trades])
            
            # Profit factor
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            
            if total_loss > 0:
                metrics.profit_factor = total_profit / total_loss
            
            # Average trade duration
            metrics.avg_trade_duration = np.mean([t.duration_days for t in self.trades])
            
            # Total commission
            metrics.total_commission = sum(order.commission for order in self.filled_orders)
        
        return metrics
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """
        Lấy equity curve dưới dạng DataFrame
        
        Returns:
            DataFrame với columns [timestamp, equity]
        """
        if not self.equity_curve:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df['returns'] = df['equity'].pct_change()
        df['cumulative_returns'] = (df['equity'] / self.initial_capital - 1) * 100
        
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Lấy danh sách trades dưới dạng DataFrame
        
        Returns:
            DataFrame chứa thông tin các trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'Symbol': trade.symbol,
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'Quantity': trade.quantity,
                'P&L': trade.pnl,
                'Return %': trade.return_pct,
                'Duration (days)': trade.duration_days,
                'Commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)
    
    def export_results(self, filename: str = None) -> str:
        """
        Xuất kết quả backtest ra file Excel
        
        Args:
            filename: Tên file xuất (nếu None sẽ tự tạo)
            
        Returns:
            Đường dẫn file đã xuất
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Performance metrics
                metrics = self.calculate_performance_metrics()
                metrics_df = pd.DataFrame([{
                    'Metric': 'Total Return (%)',
                    'Value': f"{metrics.total_return:.2f}%"
                }, {
                    'Metric': 'Annual Return (%)',
                    'Value': f"{metrics.annual_return:.2f}%"
                }, {
                    'Metric': 'Volatility (%)',
                    'Value': f"{metrics.volatility:.2f}%"
                }, {
                    'Metric': 'Sharpe Ratio',
                    'Value': f"{metrics.sharpe_ratio:.2f}"
                }, {
                    'Metric': 'Max Drawdown (%)',
                    'Value': f"{metrics.max_drawdown:.2f}%"
                }, {
                    'Metric': 'Win Rate (%)',
                    'Value': f"{metrics.win_rate:.2f}%"
                }, {
                    'Metric': 'Profit Factor',
                    'Value': f"{metrics.profit_factor:.2f}"
                }, {
                    'Metric': 'Total Trades',
                    'Value': metrics.total_trades
                }, {
                    'Metric': 'Avg Trade Duration (days)',
                    'Value': f"{metrics.avg_trade_duration:.1f}"
                }, {
                    'Metric': 'Total Commission',
                    'Value': f"{metrics.total_commission:,.0f}"
                }])
                
                metrics_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Equity curve
                equity_df = self.get_equity_dataframe()
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
                
                # Trades
                trades_df = self.get_trades_dataframe()
                if not trades_df.empty:
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            self.logger.info(f"✅ Đã xuất kết quả backtest: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi xuất file: {e}")
            return ""

# Test module
if __name__ == "__main__":
    """
    Test BacktestingEngine
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing BacktestingEngine...")
    
    # Tạo engine
    engine = BacktestingEngine(
        initial_capital=100_000_000,  # 100M VND
        commission_rate=0.0015
    )
    
    # Tạo dữ liệu test
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 0,
        'low': 0, 
        'close': 0,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    test_data['high'] = test_data['open'] + np.random.uniform(0.5, 3, len(dates))
    test_data['low'] = test_data['open'] - np.random.uniform(0.5, 3, len(dates))
    test_data['close'] = test_data['low'] + (test_data['high'] - test_data['low']) * np.random.random(len(dates))
    
    # Test đặt lệnh và thực hiện
    for i, (date, row) in enumerate(test_data.head(50).iterrows()):
        # Cập nhật positions
        engine.update_positions({'TEST': row})
        
        # Đặt lệnh mua ở ngày đầu
        if i == 0:
            engine.place_order(date, 'TEST', OrderType.BUY, 1000, row['close'])
        
        # Đặt lệnh bán ở ngày 30
        elif i == 30:
            engine.place_order(date, 'TEST', OrderType.SELL, 1000, row['close'])
        
        # Thực hiện lệnh
        engine.execute_pending_orders(row)
        
        # Ghi equity
        engine.record_equity(date)
    
    # Tính performance
    metrics = engine.calculate_performance_metrics()
    print(f"✅ Total Return: {metrics.total_return:.2f}%")
    print(f"✅ Total Trades: {metrics.total_trades}")
    print(f"✅ Win Rate: {metrics.win_rate:.2f}%")
    
    # Export results
    filename = engine.export_results()
    print(f"✅ Exported: {filename}")
    
    print("✅ Test completed!")


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
    """
    Simplified backtest engine for strategy testing
    """
    
    def __init__(self, portfolio: 'Portfolio'):
        """
        Initialize backtest engine
        
        Args:
            portfolio: Portfolio object for position management
        """
        self.portfolio = portfolio
        self.logger = logging.getLogger(__name__)
    
    def backtest(self, strategy, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on strategy with given data
        
        Args:
            strategy: Trading strategy object
            data: Price data DataFrame
            
        Returns:
            Dict containing backtest results
        """
        try:
            # Initialize results
            results = {
                'trades': [],
                'portfolio_values': [],
                'drawdown': [],
                'performance_metrics': {}
            }
            
            # Initialize portfolio value tracking
            portfolio_values = []
            
            # Track positions
            current_position = None
            entry_signal = None
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Process each day
            for i, (date, row) in enumerate(data.iterrows()):
                current_price = row['close']
                
                # Update portfolio value
                portfolio_value = self.portfolio.calculate_portfolio_value(current_price, current_position)
                portfolio_values.append({
                    'date': date,
                    'value': portfolio_value
                })
                
                # Check for trading signals
                if i < len(signals):
                    signal = signals.iloc[i]
                    
                    # Entry signal
                    if signal['action'] == 'buy' and current_position is None:
                        # Calculate position size
                        position_size = self.portfolio.calculate_position_size(current_price)
                        
                        if position_size > 0:
                            # Enter position
                            current_position = {
                                'quantity': position_size,
                                'entry_price': current_price,
                                'entry_date': date,
                                'action': 'buy'
                            }
                            entry_signal = signal
                            
                            # Update portfolio
                            self.portfolio.current_cash -= position_size * current_price
                    
                    # Exit signal
                    elif signal['action'] == 'sell' and current_position is not None:
                        # Close position
                        exit_price = current_price
                        pnl = (exit_price - current_position['entry_price']) * current_position['quantity']
                        pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                        
                        # Create trade record
                        trade = {
                            'entry_date': current_position['entry_date'],
                            'exit_date': date,
                            'action': current_position['action'],
                            'quantity': current_position['quantity'],
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration': (date - current_position['entry_date']).days,
                            'exit_reason': 'signal'
                        }
                        
                        results['trades'].append(trade)
                        
                        # Update portfolio
                        self.portfolio.current_cash += current_position['quantity'] * exit_price
                        current_position = None
                        entry_signal = None
                
                # Check risk management
                if current_position is not None:
                    # Stop loss
                    if self.portfolio.stop_loss_pct > 0:
                        loss_pct = (current_position['entry_price'] - current_price) / current_position['entry_price']
                        if loss_pct >= self.portfolio.stop_loss_pct:
                            # Stop loss triggered
                            exit_price = current_price
                            pnl = (exit_price - current_position['entry_price']) * current_position['quantity']
                            pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                            
                            trade = {
                                'entry_date': current_position['entry_date'],
                                'exit_date': date,
                                'action': current_position['action'],
                                'quantity': current_position['quantity'],
                                'entry_price': current_position['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'duration': (date - current_position['entry_date']).days,
                                'exit_reason': 'stop_loss'
                            }
                            
                            results['trades'].append(trade)
                            self.portfolio.current_cash += current_position['quantity'] * exit_price
                            current_position = None
                    
                    # Take profit
                    if self.portfolio.take_profit_pct > 0:
                        profit_pct = (current_price - current_position['entry_price']) / current_position['entry_price']
                        if profit_pct >= self.portfolio.take_profit_pct:
                            # Take profit triggered
                            exit_price = current_price
                            pnl = (exit_price - current_position['entry_price']) * current_position['quantity']
                            pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                            
                            trade = {
                                'entry_date': current_position['entry_date'],
                                'exit_date': date,
                                'action': current_position['action'],
                                'quantity': current_position['quantity'],
                                'entry_price': current_position['entry_price'],
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'pnl_pct': pnl_pct,
                                'duration': (date - current_position['entry_date']).days,
                                'exit_reason': 'take_profit'
                            }
                            
                            results['trades'].append(trade)
                            self.portfolio.current_cash += current_position['quantity'] * exit_price
                            current_position = None
            
            # Close any remaining position
            if current_position is not None:
                final_price = data['close'].iloc[-1]
                final_date = data.index[-1]
                
                pnl = (final_price - current_position['entry_price']) * current_position['quantity']
                pnl_pct = pnl / (current_position['entry_price'] * current_position['quantity'])
                
                trade = {
                    'entry_date': current_position['entry_date'],
                    'exit_date': final_date,
                    'action': current_position['action'],
                    'quantity': current_position['quantity'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': final_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration': (final_date - current_position['entry_date']).days,
                    'exit_reason': 'end_of_data'
                }
                
                results['trades'].append(trade)
                self.portfolio.current_cash += current_position['quantity'] * final_price
            
            # Calculate performance metrics
            results['portfolio_values'] = portfolio_values
            results['performance_metrics'] = self._calculate_performance_metrics(results['trades'], portfolio_values)
            results['drawdown'] = self._calculate_drawdown(portfolio_values)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Backtest error: {e}")
            return {'trades': [], 'portfolio_values': [], 'drawdown': [], 'performance_metrics': {}}
    
    def _calculate_performance_metrics(self, trades: List[Dict], portfolio_values: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        if not portfolio_values:
            return {}
        
        # Basic metrics
        initial_value = portfolio_values[0]['value']
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        values = [pv['value'] for pv in portfolio_values]
        daily_returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Annual return
        num_days = len(portfolio_values)
        annual_return = (1 + total_return) ** (252 / num_days) - 1
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = initial_value
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
        total_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _calculate_drawdown(self, portfolio_values: List[Dict]) -> List[Dict]:
        """Calculate drawdown series"""
        drawdown_data = []
        peak = portfolio_values[0]['value']
        
        for pv in portfolio_values:
            value = pv['value']
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            drawdown_data.append({
                'date': pv['date'],
                'drawdown': -drawdown  # Negative for display
            })
        
        return drawdown_data


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
