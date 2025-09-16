"""
🔄 BACKTESTING SYSTEM - Hệ thống backtest chiến lược
================================================

Module này thực hiện backtest các chiến lược giao dịch
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """
    Loại lệnh giao dịch
    """
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """
    Trạng thái lệnh
    """
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    """
    Lệnh giao dịch
    """
    symbol: str
    order_type: OrderType
    quantity: int
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    executed_price: Optional[float] = None
    executed_time: Optional[datetime] = None

@dataclass
class Position:
    """
    Vị thế giao dịch
    """
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class BacktestResult:
    """
    Kết quả backtest
    """
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame

class BacktestEngine:
    """
    Engine thực hiện backtest
    """
    
    def __init__(self, initial_capital: float = 100000000):  # 100M VND
        """
        Khởi tạo BacktestEngine
        
        Args:
            initial_capital: Vốn ban đầu (VND)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔄 BacktestEngine đã được khởi tạo với vốn: {initial_capital:,.0f} VND")
    
    def reset(self):
        """
        Reset engine về trạng thái ban đầu
        """
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.logger.info("🔄 Engine đã được reset")
    
    def run_backtest(self, df: pd.DataFrame, strategy_func: callable,
                    commission: float = 0.0015) -> BacktestResult:
        """
        Chạy backtest với chiến lược
        
        Args:
            df: DataFrame chứa dữ liệu và tín hiệu
            strategy_func: Hàm chiến lược trả về tín hiệu
            commission: Phí giao dịch (%)
            
        Returns:
            BacktestResult object
        """
        self.reset()
        self.commission = commission
        
        if len(df) == 0:
            self.logger.error("❌ Không có dữ liệu để backtest")
            return None
        
        symbol = 'STOCK'  # Default symbol
        
        # Iterate through each day
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_date = current_row.get('time', datetime.now())
            current_price = current_row['close']
            
            # Update portfolio value
            portfolio_value = self.calculate_portfolio_value(current_price, symbol)
            self.equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'stock_value': portfolio_value - self.current_capital
            })
            
            # Get strategy signal
            signal = strategy_func(df.iloc[:i+1], i)
            
            # Execute trades based on signal
            if signal == 'BUY':
                self.buy_stock(symbol, current_price, current_date)
            elif signal == 'SELL':
                self.sell_stock(symbol, current_price, current_date)
        
        # Calculate final results
        return self.calculate_results(df, symbol)
    
    def buy_stock(self, symbol: str, price: float, timestamp: datetime,
                 amount: Optional[float] = None):
        """
        Mua cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            price: Giá mua
            timestamp: Thời gian
            amount: Số tiền mua (None = mua hết tiền có)
        """
        if amount is None:
            amount = self.current_capital * 0.95  # Để lại 5% cash
        
        if amount > self.current_capital:
            self.logger.warning(f"⚠️ Không đủ tiền để mua {symbol}")
            return
        
        quantity = int(amount / (price * (1 + self.commission)))
        if quantity <= 0:
            return
        
        total_cost = quantity * price * (1 + self.commission)
        
        # Create order
        order = Order(
            symbol=symbol,
            order_type=OrderType.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            status=OrderStatus.EXECUTED,
            executed_price=price,
            executed_time=timestamp
        )
        self.orders.append(order)
        
        # Update position
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            total_quantity = current_pos.quantity + quantity
            total_value = (current_pos.quantity * current_pos.avg_price) + (quantity * price)
            new_avg_price = total_value / total_quantity
            
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_price=new_avg_price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=current_pos.realized_pnl
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
        # Update capital
        self.current_capital -= total_cost
        
        self.logger.debug(f"✅ Mua {quantity} cổ phiếu {symbol} @ {price:,.0f}")
    
    def sell_stock(self, symbol: str, price: float, timestamp: datetime,
                  quantity: Optional[int] = None):
        """
        Bán cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            price: Giá bán
            timestamp: Thời gian
            quantity: Số lượng bán (None = bán hết)
        """
        if symbol not in self.positions or self.positions[symbol].quantity <= 0:
            self.logger.warning(f"⚠️ Không có cổ phiếu {symbol} để bán")
            return
        
        current_pos = self.positions[symbol]
        
        if quantity is None or quantity > current_pos.quantity:
            quantity = current_pos.quantity
        
        total_revenue = quantity * price * (1 - self.commission)
        cost_basis = quantity * current_pos.avg_price
        realized_pnl = total_revenue - cost_basis
        
        # Create order
        order = Order(
            symbol=symbol,
            order_type=OrderType.SELL,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            status=OrderStatus.EXECUTED,
            executed_price=price,
            executed_time=timestamp
        )
        self.orders.append(order)
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': None,  # Would need to track this
            'exit_date': timestamp,
            'entry_price': current_pos.avg_price,
            'exit_price': price,
            'quantity': quantity,
            'pnl': realized_pnl,
            'return_pct': (price / current_pos.avg_price - 1) * 100
        }
        self.trades.append(trade)
        
        # Update position
        remaining_quantity = current_pos.quantity - quantity
        if remaining_quantity > 0:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                avg_price=current_pos.avg_price,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=current_pos.realized_pnl + realized_pnl
            )
        else:
            del self.positions[symbol]
        
        # Update capital
        self.current_capital += total_revenue
        
        self.logger.debug(f"✅ Bán {quantity} cổ phiếu {symbol} @ {price:,.0f}, PnL: {realized_pnl:,.0f}")
    
    def calculate_portfolio_value(self, current_price: float, symbol: str) -> float:
        """
        Tính giá trị danh mục
        
        Args:
            current_price: Giá hiện tại
            symbol: Mã cổ phiếu
            
        Returns:
            Tổng giá trị danh mục
        """
        total_value = self.current_capital
        
        if symbol in self.positions:
            position = self.positions[symbol]
            stock_value = position.quantity * current_price
            total_value += stock_value
        
        return total_value
    
    def calculate_results(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Tính toán kết quả backtest
        
        Args:
            df: DataFrame dữ liệu
            symbol: Mã cổ phiếu
            
        Returns:
            BacktestResult object
        """
        if len(self.equity_curve) == 0:
            self.logger.error("❌ Không có dữ liệu equity curve")
            return None
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate returns
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Calculate annualized return
        days = len(equity_df)
        years = days / 365.25
        annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate max drawdown
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_df) > 1:
            returns = equity_df['portfolio_value'].pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Trade statistics
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            total_trades = len(trades_df)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
            else:
                profit_factor = float('inf') if avg_win > 0 else 0
        else:
            winning_trades = losing_trades = total_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_df=trades_df,
            equity_curve=equity_df
        )
        
        self.logger.info(f"✅ Backtest hoàn thành: Return {total_return:.2f}%, Max DD {max_drawdown:.2f}%")
        
        return result
    
    def print_results(self, result: BacktestResult):
        """
        In kết quả backtest
        
        Args:
            result: BacktestResult object
        """
        print("="*60)
        print("📊 KẾT QUẢ BACKTEST")
        print("="*60)
        print(f"💰 Vốn ban đầu: {self.initial_capital:,.0f} VND")
        print(f"💰 Vốn cuối kỳ: {result.equity_curve['portfolio_value'].iloc[-1]:,.0f} VND")
        print(f"📈 Tổng lợi nhuận: {result.total_return:.2f}%")
        print(f"📈 Lợi nhuận hàng năm: {result.annualized_return:.2f}%")
        print(f"📉 Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"🎯 Tỷ lệ thắng: {result.win_rate:.1f}%")
        print(f"📊 Profit Factor: {result.profit_factor:.2f}")
        print(f"🔄 Tổng số giao dịch: {result.total_trades}")
        print(f"✅ Giao dịch thắng: {result.winning_trades}")
        print(f"❌ Giao dịch thua: {result.losing_trades}")
        print(f"💚 Lãi trung bình: {result.avg_win:,.0f} VND")
        print(f"💔 Lỗ trung bình: {result.avg_loss:,.0f} VND")
        print("="*60)

class StrategyLibrary:
    """
    Thư viện các chiến lược giao dịch
    """
    
    @staticmethod
    def golden_cross_strategy(df: pd.DataFrame, index: int) -> str:
        """
        Chiến lược Golden Cross (SMA 50 > SMA 200)
        
        Args:
            df: DataFrame dữ liệu
            index: Chỉ số hiện tại
            
        Returns:
            Tín hiệu: 'BUY', 'SELL', hoặc 'HOLD'
        """
        if index < 200:  # Chưa đủ dữ liệu
            return 'HOLD'
        
        current = df.iloc[index]
        previous = df.iloc[index-1]
        
        # Golden Cross: SMA 50 vượt lên SMA 200
        if ('sma_50' in df.columns and 'sma_200' in df.columns):
            if (current['sma_50'] > current['sma_200'] and 
                previous['sma_50'] <= previous['sma_200']):
                return 'BUY'
            
            # Death Cross: SMA 50 xuống dưới SMA 200
            elif (current['sma_50'] < current['sma_200'] and 
                  previous['sma_50'] >= previous['sma_200']):
                return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def rsi_strategy(df: pd.DataFrame, index: int) -> str:
        """
        Chiến lược RSI
        
        Args:
            df: DataFrame dữ liệu
            index: Chỉ số hiện tại
            
        Returns:
            Tín hiệu: 'BUY', 'SELL', hoặc 'HOLD'
        """
        if index < 14 or 'rsi' not in df.columns:
            return 'HOLD'
        
        current_rsi = df.iloc[index]['rsi']
        previous_rsi = df.iloc[index-1]['rsi']
        
        # Mua khi RSI vượt lên 30 (thoát vùng oversold)
        if current_rsi > 30 and previous_rsi <= 30:
            return 'BUY'
        
        # Bán khi RSI xuống dưới 70 (thoát vùng overbought)
        elif current_rsi < 70 and previous_rsi >= 70:
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def macd_strategy(df: pd.DataFrame, index: int) -> str:
        """
        Chiến lược MACD
        
        Args:
            df: DataFrame dữ liệu
            index: Chỉ số hiện tại
            
        Returns:
            Tín hiệu: 'BUY', 'SELL', hoặc 'HOLD'
        """
        if (index < 26 or 'macd' not in df.columns or 
            'macd_signal' not in df.columns):
            return 'HOLD'
        
        current = df.iloc[index]
        previous = df.iloc[index-1]
        
        # MACD vượt lên Signal line
        if (current['macd'] > current['macd_signal'] and 
            previous['macd'] <= previous['macd_signal']):
            return 'BUY'
        
        # MACD xuống dưới Signal line
        elif (current['macd'] < current['macd_signal'] and 
              previous['macd'] >= previous['macd_signal']):
            return 'SELL'
        
        return 'HOLD'
    
    @staticmethod
    def bollinger_bands_strategy(df: pd.DataFrame, index: int) -> str:
        """
        Chiến lược Bollinger Bands
        
        Args:
            df: DataFrame dữ liệu
            index: Chỉ số hiện tại
            
        Returns:
            Tín hiệu: 'BUY', 'SELL', hoặc 'HOLD'
        """
        if (index < 20 or not all(col in df.columns for col in 
                                 ['bb_upper', 'bb_lower', 'close'])):
            return 'HOLD'
        
        current = df.iloc[index]
        previous = df.iloc[index-1]
        
        # Mua khi giá chạm lower band rồi hồi phục
        if (previous['close'] <= previous['bb_lower'] and 
            current['close'] > current['bb_lower']):
            return 'BUY'
        
        # Bán khi giá chạm upper band
        elif current['close'] >= current['bb_upper']:
            return 'SELL'
        
        return 'HOLD'

# Test module
if __name__ == "__main__":
    """
    Test BacktestEngine
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=252, freq='D')  # 1 year
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 50000 + np.cumsum(np.random.randn(252) * 500)  # VND prices
    high_prices = close_prices + np.random.rand(252) * 1000
    low_prices = close_prices - np.random.rand(252) * 1000
    open_prices = close_prices + np.random.randn(252) * 200
    volumes = np.random.randint(100000, 1000000, 252)
    
    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Add indicators
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['rsi'] = 50 + np.random.randn(252) * 20
    
    print("🧪 Testing BacktestEngine...")
    
    # Test Golden Cross strategy
    engine = BacktestEngine(initial_capital=100_000_000)  # 100M VND
    result = engine.run_backtest(df, StrategyLibrary.golden_cross_strategy)
    
    if result:
        engine.print_results(result)
        print(f"✅ Equity curve points: {len(result.equity_curve)}")
        print(f"✅ Total trades: {len(result.trades_df)}")
    
    print("✅ Test completed!")
