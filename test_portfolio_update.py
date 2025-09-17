#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.utils.trading_portfolio_manager import TradingPortfolioManager

def test_portfolio_update():
    print("🧪 Testing portfolio statistics update...")
    
    # Khởi tạo portfolio manager
    manager = TradingPortfolioManager()
    
    # Lấy danh sách portfolios hiện tại
    portfolios = manager.get_portfolios()
    
    print(f"📊 Found {len(portfolios)} portfolios")
    
    for portfolio in portfolios:
        print("\n" + "="*50)
        print(f"📁 Portfolio: {portfolio['name']}")
        print(f"🆔 ID: {portfolio['id']}")
        print(f"💰 Initial Cash: {portfolio['initial_cash']:,.0f} VNĐ")
        print(f"📈 Total Invested: {portfolio.get('total_invested', 0):,.0f} VNĐ")
        print(f"💵 Current Cash: {portfolio.get('current_cash', 0):,.0f} VNĐ")
        print(f"💎 Total Value: {portfolio.get('total_value', 0):,.0f} VNĐ")
        print(f"📊 Profit/Loss: {portfolio.get('total_profit_loss', 0):,.0f} VNĐ")
        print(f"📅 Last Updated: {portfolio.get('last_updated', 'N/A')}")
    
    # Test refresh
    print("\n" + "="*60)
    print("🔄 Testing refresh all portfolios...")
    
    success = manager.refresh_all_portfolios()
    print(f"✅ Refresh result: {success}")
    
    # Lấy lại danh sách sau khi refresh
    updated_portfolios = manager.get_portfolios()
    
    print("\n📊 Updated portfolio stats:")
    for portfolio in updated_portfolios:
        print(f"📁 {portfolio['name']}: Invested={portfolio.get('total_invested', 0):,.0f}, Value={portfolio.get('total_value', 0):,.0f}, P/L={portfolio.get('total_profit_loss', 0):,.0f}")

if __name__ == "__main__":
    test_portfolio_update()