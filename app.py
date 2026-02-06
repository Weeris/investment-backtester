import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random

# Currency conversion rates (approximate)
CURRENCY_RATES = {
    'THB': 30,  # 1 USD = 30 THB
    'USD': 1,   # 1 USD = 1 USD
    'HKD': 7.8  # 1 USD = 7.8 HKD
}

# Language dictionaries
LANGUAGES = {
    'th': {
        'title': '📈 ระบบย้อนกลับการลงทุน (หน่วยเงินบาท) - หลายภาษา',
        'subtitle': 'ระบบย้อนกลับการลงทุนที่ใช้หน่วยเงินไทย โดยใช้กลยุทธ์ EMA และ RSI ที่คำนวณจากราคาปิดและดำเนินการซื้อขายที่ราคาเปิด',
        'currency_label': 'เลือกสกุลเงิน:',
        'symbol_label': 'สัญลักษณ์หุ้น/สินทรัพย์',
        'start_date_label': 'วันเริ่มต้น',
        'end_date_label': 'วันสิ้นสุด',
        'capital_label': 'ทุนเริ่มต้น',
        'strategy_label': 'กลยุทธ์การเทรด',
        'super_trend': 'SuperTrend',
        'buy_and_hold': 'ถือยาว (Buy and Hold)',
        'ema_settings': 'การตั้งค่า EMA',
        'rsi_settings': 'การตั้งค่า RSI',
        'position_size': 'ขนาดตำแหน่ง',
        'risk_management': 'การจัดการความเสี่ยง',
        'stop_loss': 'หยุดขาดทุน (%)',
        'take_profit': 'ทำกำไร (%)',
        'fast_ema': 'EMA เร็ว',
        'slow_ema': 'EMA ช้า',
        'buy_threshold': 'เกณฑ์ซื้อ RSI',
        'sell_threshold': 'เกณฑ์ขาย RSI',
        'size_percent': 'ขนาดตำแหน่ง (%)',
        'run_backtest': 'เริ่มการย้อนกลับ',
        'symbol': 'สัญลักษณ์',
        'capital': 'ทุนเริ่มต้น',
        'start': 'วันเริ่มต้น',
        'end': 'วันสิ้นสุด',
        'strategy': 'กลยุทธ์',
        'results': 'ผลลัพธ์การเทรด',
        'total_trades': 'จำนวนเทรดทั้งหมด',
        'win_rate': 'อัตราชนะ',
        'final_value': 'มูลค่าสุดท้าย',
        'total_return': 'ผลตอบแทนรวม',
        'trade_log': 'บันทึกการเทรด (คู่ซื้อ/ขาย)',
        'buy_date': 'วันที่ซื้อ',
        'buy_price': 'ราคาซื้อ',
        'sell_date': 'วันที่ขาย',
        'sell_price': 'ราคาขาย',
        'shares': 'จำนวนหุ้น',
        'profit': 'กำไร',
        'profit_pct': 'กำไร %',
        'holding_period': 'ถือครอง (วัน)',
        'indices': 'ดัชนี',
        'us_stocks': 'หุ้น US',
        'intl_stocks': 'หุ้นต่างประเทศ',
        'thai_stocks': 'หุ้นไทย',
        'other_assets': 'สินทรัพย์อื่นๆ',
        'price_chart': 'ราคา & EMAs',
        'rsi_chart': 'RSI',
        'portfolio_chart': 'มูลค่าพอร์ต ({})',
        'sell_level': 'ระดับขาย',
        'buy_level': 'ระดับซื้อ',
        'no_completed_trades': 'ไม่มีการเทรดที่เสร็จสมบูรณ์ (ซื้อ + ขาย)',
        'no_trades_found': 'ไม่มีคู่ซื้อ/ขายที่แสดง',
        'backtest_complete': 'การย้อนกลับเสร็จสมบูณ์!',
        'signals_found': 'พบสัญญาณซื้อ {} ครั้ง และสัญญาณขาย {} ครั้ง',
        'no_signals': 'ไม่พบสัญญาณการซื้อหรือขายสำหรับกลยุทธ์ {} บนสินทรัพย์ {}',
        'try_different_params': 'อาจเป็นเพราะช่วงเวลาที่เลือกไม่มีการเคลื่อนไหวที่เหมาะสม หรือพารามิเตอร์ที่ตั้งไว้ไม่เหมาะสม'
    },
    'en': {
        'title': '📈 Investment Backtesting Platform - Multilingual',
        'subtitle': 'Backtesting platform using various currencies with EMA and RSI strategies calculating from closing prices and executing trades at opening prices',
        'currency_label': 'Select Currency:',
        'symbol_label': 'Stock/Asset Symbol',
        'start_date_label': 'Start Date',
        'end_date_label': 'End Date',
        'capital_label': 'Initial Capital',
        'strategy_label': 'Trading Strategy',
        'super_trend': 'SuperTrend',
        'buy_and_hold': 'Buy and Hold',
        'ema_settings': 'EMA Settings',
        'rsi_settings': 'RSI Settings',
        'position_size': 'Position Size',
        'risk_management': 'Risk Management',
        'stop_loss': 'Stop Loss (%)',
        'take_profit': 'Take Profit (%)',
        'fast_ema': 'Fast EMA',
        'slow_ema': 'Slow EMA',
        'buy_threshold': 'RSI Buy Threshold',
        'sell_threshold': 'RSI Sell Threshold',
        'size_percent': 'Position Size (%)',
        'run_backtest': 'Run Backtest',
        'symbol': 'Symbol',
        'capital': 'Initial Capital',
        'start': 'Start Date',
        'end': 'End Date',
        'strategy': 'Strategy',
        'results': 'Trading Results',
        'total_trades': 'Total Trades',
        'win_rate': 'Win Rate',
        'final_value': 'Final Value',
        'total_return': 'Total Return',
        'trade_log': 'Trade Log (Buy/Sell Pairs)',
        'buy_date': 'Buy Date',
        'buy_price': 'Buy Price',
        'sell_date': 'Sell Date',
        'sell_price': 'Sell Price',
        'shares': 'Shares',
        'profit': 'Profit',
        'profit_pct': 'Profit %',
        'holding_period': 'Holding (days)',
        'indices': 'Indices',
        'us_stocks': 'US Stocks',
        'intl_stocks': 'International Stocks',
        'thai_stocks': 'Thai Stocks',
        'other_assets': 'Other Assets',
        'price_chart': 'Price & EMAs',
        'rsi_chart': 'RSI',
        'portfolio_chart': 'Portfolio Value ({})',
        'sell_level': 'Sell Level',
        'buy_level': 'Buy Level',
        'no_completed_trades': 'No completed trades (buy + sell)',
        'no_trades_found': 'No buy/sell pairs to display',
        'backtest_complete': 'Backtest completed!',
        'signals_found': 'Found {} buy signals and {} sell signals',
        'no_signals': 'No buy or sell signals found for strategy {} on asset {}',
        'try_different_params': 'This may be because the selected time period has no suitable movements or parameters are not appropriate'
    },
    'zh': {
        'title': '📈 投资回测平台 - 多语言',
        'subtitle': '使用多种货币的回测平台，采用EMA和RSI策略，从收盘价计算并以开盘价执行交易',
        'currency_label': '选择货币:',
        'symbol_label': '股票/资产代码',
        'start_date_label': '开始日期',
        'end_date_label': '结束日期',
        'capital_label': '初始资本',
        'strategy_label': '交易策略',
        'super_trend': 'SuperTrend',
        'buy_and_hold': '买入并持有',
        'ema_settings': 'EMA设置',
        'rsi_settings': 'RSI设置',
        'position_size': '仓位大小',
        'risk_management': '风险管理',
        'stop_loss': '止损 (%)',
        'take_profit': '止盈 (%)',
        'fast_ema': '快速EMA',
        'slow_ema': '慢速EMA',
        'buy_threshold': 'RSI买入阈值',
        'sell_threshold': 'RSI卖出阈值',
        'size_percent': '仓位大小 (%)',
        'run_backtest': '运行回测',
        'symbol': '代码',
        'capital': '初始资本',
        'start': '开始日期',
        'end': '结束日期',
        'strategy': '策略',
        'results': '交易结果',
        'total_trades': '总交易数',
        'win_rate': '胜率',
        'final_value': '最终价值',
        'total_return': '总回报',
        'trade_log': '交易记录 (买卖对)',
        'buy_date': '买入日期',
        'buy_price': '买入价格',
        'sell_date': '卖出日期',
        'sell_price': '卖出价格',
        'shares': '股数',
        'profit': '利润',
        'profit_pct': '利润率',
        'holding_period': '持有期 (天)',
        'indices': '指数',
        'us_stocks': '美国股票',
        'intl_stocks': '国际股票',
        'thai_stocks': '泰国股票',
        'other_assets': '其他资产',
        'price_chart': '价格 & EMA',
        'rsi_chart': 'RSI',
        'portfolio_chart': '投资组合价值 ({})',
        'sell_level': '卖出水平',
        'buy_level': '买入水平',
        'no_completed_trades': '无完成交易 (买入 + 卖出)',
        'no_trades_found': '无买卖对显示',
        'backtest_complete': '回测完成!',
        'signals_found': '发现 {} 个买入信号和 {} 个卖出信号',
        'no_signals': '在资产 {} 上未找到策略 {} 的买入或卖出信号',
        'try_different_params': '这可能是因为所选时间段内没有合适的走势，或者参数设置不当'
    }
}

# Initialize session state for language and currency
if 'language' not in st.session_state:
    st.session_state.language = 'th'  # Default to Thai
if 'currency' not in st.session_state:
    st.session_state.currency = 'THB'  # Default to THB

# Technical indicators
def calculate_ema(data, window):
    """Calculate Exponential Moving Average using closing prices"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index using closing prices"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(data, window=14):
    """Calculate Average True Range for volatility using closing prices"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_supertrend(data, atr_multiplier=3, atr_window=10):
    """Calculate SuperTrend indicator"""
    df = data.copy()
    
    # Calculate ATR
    atr = calculate_atr(df, atr_window)
    
    # Calculate Basic Upper and Lower Bands
    df['Basic_Upper_Band'] = (df['High'] + df['Low']) / 2 + atr_multiplier * atr
    df['Basic_Lower_Band'] = (df['High'] + df['Low']) / 2 - atr_multiplier * atr
    
    # Initialize Final Upper and Lower Bands
    df['Final_Upper_Band'] = df['Basic_Upper_Band'].copy()
    df['Final_Lower_Band'] = df['Basic_Lower_Band'].copy()
    
    # Calculate SuperTrend
    df['SuperTrend'] = np.nan
    
    # Initialize the first SuperTrend value
    df['SuperTrend'].iloc[0] = df['Final_Upper_Band'].iloc[0] if df['Close'].iloc[0] <= df['Final_Upper_Band'].iloc[0] else df['Final_Lower_Band'].iloc[0]
    
    for i in range(1, len(df)):
        # Update Final Upper Band
        df['Final_Upper_Band'].iloc[i] = min(df['Basic_Upper_Band'].iloc[i], df['Final_Upper_Band'].iloc[i-1])
        if df['Close'].iloc[i-1] > df['Final_Upper_Band'].iloc[i-1]:
            df['Final_Upper_Band'].iloc[i] = df['Basic_Upper_Band'].iloc[i]
        
        # Update Final Lower Band
        df['Final_Lower_Band'].iloc[i] = max(df['Basic_Lower_Band'].iloc[i], df['Final_Lower_Band'].iloc[i-1])
        if df['Close'].iloc[i-1] < df['Final_Lower_Band'].iloc[i-1]:
            df['Final_Lower_Band'].iloc[i] = df['Basic_Lower_Band'].iloc[i]
        
        # Determine SuperTrend value
        if pd.isna(df['SuperTrend'].iloc[i-1]):
            # For the first iteration after initialization
            df['SuperTrend'].iloc[i] = df['Final_Upper_Band'].iloc[i] if df['Close'].iloc[i] <= df['Final_Upper_Band'].iloc[i] else df['Final_Lower_Band'].iloc[i]
        elif df['SuperTrend'].iloc[i-1] == df['Final_Upper_Band'].iloc[i-1]:
            # Previous SuperTrend was upper band
            if df['Close'].iloc[i] <= df['Final_Upper_Band'].iloc[i]:
                df['SuperTrend'].iloc[i] = df['Final_Upper_Band'].iloc[i]
            else:
                df['SuperTrend'].iloc[i] = df['Final_Lower_Band'].iloc[i]
        else:
            # Previous SuperTrend was lower band
            if df['Close'].iloc[i] >= df['Final_Lower_Band'].iloc[i]:
                df['SuperTrend'].iloc[i] = df['Final_Lower_Band'].iloc[i]
            else:
                df['SuperTrend'].iloc[i] = df['Final_Upper_Band'].iloc[i]
    
    return df['SuperTrend']

def calculate_chaloke_cdc(data, atr_multiplier=1.5, pivot_lookback=5):
    """
    Calculate ChalokeDotCom CDC indicator
    Based on Chaloke's methodology for Thai stock market analysis
    """
    df = data.copy()
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(df, window=14)
    
    # Calculate pivot points for support/resistance levels
    df['High_Last_N'] = df['High'].rolling(window=pivot_lookback).max()
    df['Low_Last_N'] = df['Low'].rolling(window=pivot_lookback).min()
    
    # Calculate CDC trend lines
    df['CDC_Middle_Line'] = (df['High_Last_N'] + df['Low_Last_N']) / 2
    
    # Calculate support and resistance levels with ATR adjustment
    df['CDC_Support'] = df['Low_Last_N'] - (atr * atr_multiplier)
    df['CDC_Resistance'] = df['High_Last_N'] + (atr * atr_multiplier)
    
    # Calculate bullish and bearish signal conditions
    df['Price_Above_Middle'] = df['Close'] > df['CDC_Middle_Line']
    df['Price_Below_Middle'] = df['Close'] < df['CDC_Middle_Line']
    
    # Previous conditions for crossover detection
    df['Prev_Above_Middle'] = df['Price_Above_Middle'].shift(1)
    df['Prev_Below_Middle'] = df['Price_Below_Middle'].shift(1)
    
    # Bullish signal: price crosses above middle line
    df['CDC_Bullish_Signal'] = (df['Prev_Below_Middle']) & (df['Price_Above_Middle'])
    
    # Bearish signal: price crosses below middle line
    df['CDC_Bearish_Signal'] = (df['Prev_Above_Middle']) & (df['Price_Below_Middle'])
    
    # Calculate signal strength based on distance from middle line and ATR
    df['Distance_From_Middle'] = abs(df['Close'] - df['CDC_Middle_Line'])
    df['Signal_Strength'] = df['Distance_From_Middle'] / atr
    
    # Normalize signal strength
    df['CDC_Signal_Strength'] = np.where(
        df['CDC_Bullish_Signal'] | df['CDC_Bearish_Signal'],
        np.minimum(df['Signal_Strength'], 2.0),  # Cap at 2.0 for normalization
        0
    )
    
    # Determine overall trend
    df['CDC_Trend'] = np.where(
        df['Close'] > df['CDC_Middle_Line'],
        'Bullish',
        np.where(df['Close'] < df['CDC_Middle_Line'], 'Bearish', 'Neutral')
    )
    
    return (
        df['CDC_Bullish_Signal'],
        df['CDC_Bearish_Signal'],
        df['CDC_Trend'],
        df['CDC_Support'],
        df['CDC_Resistance'],
        df['CDC_Signal_Strength']
    )

class MultiCurrencyBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000, currency='THB'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.currency = currency
        self.data = None
        self.positions = []
        self.trades = []

    def load_data_with_delay(self):
        """Load historical data with delay to avoid rate limits"""
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        ticker = yf.Ticker(self.symbol)
        try:
            self.data = ticker.history(start=self.start_date, end=self.end_date, interval="1d")
            return not self.data.empty
        except Exception as e:
            # Some Thai stocks might need different approach
            if "BK" in self.symbol or self.symbol in ["^STI", "^SET50", "^SET100"]:
                # Try with different parameters for Thai market
                try:
                    self.data = ticker.history(start=self.start_date, end=self.end_date, interval="1d", auto_adjust=True)
                    return not self.data.empty
                except:
                    # If still fails, try a broader date range
                    try:
                        adjusted_start = self.start_date - timedelta(days=30)  # Try 30 days earlier
                        self.data = ticker.history(start=adjusted_start, end=self.end_date, interval="1d", auto_adjust=True)
                        # Filter back to original date range
                        self.data = self.data[self.data.index.date >= self.start_date]
                        return not self.data.empty
                    except:
                        return False
            return False

    def add_indicators(self, ema_fast_window=12, ema_slow_window=26, rsi_window=14, supertrend_multiplier=3, supertrend_window=10):
        """Add technical indicators using closing prices"""
        self.data['EMA_Fast'] = calculate_ema(self.data['Close'], ema_fast_window)
        self.data['EMA_Slow'] = calculate_ema(self.data['Close'], ema_slow_window)
        self.data['RSI'] = calculate_rsi(self.data['Close'], rsi_window)
        self.data['ATR'] = calculate_atr(self.data, 14)
        self.data['SuperTrend'] = calculate_supertrend(self.data, supertrend_multiplier, supertrend_window)

    def generate_signals_by_strategy(self, strategy_type, ema_slow_window=26, rsi_buy_threshold=30, rsi_sell_threshold=70, 
                                    cdc_atr_multiplier=1.5, cdc_pivot_lookback=5):
        """Generate buy/sell signals based on selected strategy using closing prices"""
        buy_signals = []
        sell_signals = []
        
        # Ensure we have enough data points for calculations
        rsi_window = 14  # Standard RSI window
        max_window = max(ema_slow_window, rsi_window, cdc_pivot_lookback)
        
        for i in range(max_window, len(self.data)):  # Using max window for safety
            current_row = self.data.iloc[i]
            prev_row = self.data.iloc[i-1]
            
            buy_signal = False
            sell_signal = False
            
            if strategy_type == "EMA Crossover":
                # Buy when fast EMA crosses above slow EMA (using closing prices for indicator)
                if (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] > current_row['EMA_Slow']):
                    buy_signal = True
                # Sell when fast EMA crosses below slow EMA (using closing prices for indicator)
                elif (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] < current_row['EMA_Slow']):
                    sell_signal = True
            
            elif strategy_type == "RSI Oversold/Oversold":
                # Buy when RSI is below threshold (using closing prices for indicator)
                if not pd.isna(prev_row['RSI']) and not pd.isna(current_row['RSI']):
                    if prev_row['RSI'] <= rsi_buy_threshold and current_row['RSI'] > rsi_buy_threshold:
                        buy_signal = True
                    # Sell when RSI is above threshold (using closing prices for indicator)
                    elif prev_row['RSI'] >= rsi_sell_threshold and current_row['RSI'] < rsi_sell_threshold:
                        sell_signal = True
            
            elif strategy_type == "Buy and Hold":
                # Buy and Hold strategy - Buy on first day, sell on last day
                # Buy on the first available day
                if i == max_window:  # First day where we have data for all indicators
                    buy_signal = True
                # Sell on the last day
                elif i == len(self.data) - 1:
                    sell_signal = True
            
            elif strategy_type == "SuperTrend":
                # SuperTrend strategy - Buy when price closes above SuperTrend, Sell when below
                if not pd.isna(prev_row["SuperTrend"]) and not pd.isna(current_row["SuperTrend"]):
                    # Buy when price moves above SuperTrend (uptrend)
                    if prev_row["Close"] <= prev_row["SuperTrend"] and current_row["Close"] > current_row["SuperTrend"]:
                        buy_signal = True
                    # Sell when price moves below SuperTrend (downtrend)
                    elif prev_row["Close"] >= prev_row["SuperTrend"] and current_row["Close"] < current_row["SuperTrend"]:
                        sell_signal = True

            elif strategy_type == "Chaloke CDC":
                # Calculate CDC indicators for the current data up to this point
                df_subset = self.data.iloc[:i+1].copy()  # Include current and previous data
                
                if len(df_subset) > cdc_pivot_lookback:  # Ensure we have enough data for CDC calculation
                    (bullish_signals, bearish_signals, cdc_trend, 
                     support_levels, resistance_levels, signal_strength) = calculate_chaloke_cdc(
                         df_subset, atr_multiplier=cdc_atr_multiplier, pivot_lookback=cdc_pivot_lookback
                     )
                     
                    # Get the current CDC signals
                    current_bullish = bullish_signals.iloc[-1] if len(bullish_signals) > 0 else False
                    current_bearish = bearish_signals.iloc[-1] if len(bearish_signals) > 0 else False
                    
                    if current_bullish:
                        buy_signal = True
                    elif current_bearish:
                        sell_signal = True

            elif strategy_type == "Combined":
                # Combined strategy using both EMA and RSI (using closing prices for indicators)
                # Buy when EMA bullish AND RSI bullish
                ema_bullish = (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] > current_row['EMA_Slow'])
                rsi_bullish = not pd.isna(prev_row['RSI']) and not pd.isna(current_row['RSI']) and \
                              prev_row['RSI'] <= rsi_buy_threshold and current_row['RSI'] > rsi_buy_threshold
                
                # Sell when EMA bearish AND RSI bearish
                ema_bearish = (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] < current_row['EMA_Slow'])
                rsi_bearish = not pd.isna(prev_row['RSI']) and not pd.isna(current_row['RSI']) and \
                              prev_row['RSI'] >= rsi_sell_threshold and current_row['RSI'] < rsi_sell_threshold
                
                if ema_bullish and rsi_bullish:
                    buy_signal = True
                elif ema_bearish and rsi_bearish:
                    sell_signal = True
            
            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)
        
        # Fill with False for initial data points where we can't calculate signals
        for _ in range(len(self.data) - len(buy_signals)):
            buy_signals.insert(0, False)
            sell_signals.insert(0, False)
        
        return buy_signals, sell_signals

    def run_backtest_by_strategy(self, strategy_type, position_size_pct=0.1, stop_loss_pct=None, take_profit_pct=None, 
                                rsi_buy_threshold=30, rsi_sell_threshold=70, ema_fast_window=12, ema_slow_window=26,
                                cdc_atr_multiplier=1.5, cdc_pivot_lookback=5):
        """Run backtest with predefined strategy using closing prices for indicators and opening prices for transactions"""
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        in_position = False
        entry_price = 0
        trade_start_date = None
        
        # Generate signals based on strategy (using closing prices for indicators)
        buy_signals, sell_signals = self.generate_signals_by_strategy(
            strategy_type, ema_slow_window=ema_slow_window, rsi_buy_threshold=rsi_buy_threshold, rsi_sell_threshold=rsi_sell_threshold,
            cdc_atr_multiplier=cdc_atr_multiplier, cdc_pivot_lookback=cdc_pivot_lookback
        )
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            # Use OPENING price for actual buy/sell transactions
            current_open_price = row['Open']
            current_close_price = row['Close']
            
            # Check if we have valid signals for this day
            if i < len(buy_signals):
                should_buy = buy_signals[i] and not in_position and cash > 0
                should_sell = sell_signals[i] and in_position and shares > 0
            else:
                should_buy = False
                should_sell = False
            
            # Apply stop loss and take profit if in position (based on current open price)
            if in_position:
                current_profit_pct = (current_open_price - entry_price) / entry_price
                
                # Stop loss
                if stop_loss_pct and current_profit_pct <= -stop_loss_pct/100:
                    should_sell = True
                
                # Take profit
                if take_profit_pct and current_profit_pct >= take_profit_pct/100:
                    should_sell = True
            
            # Execute buy using OPENING price
            if should_buy:
                shares_to_buy = int((cash * position_size_pct) / current_open_price)
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cost = shares_to_buy * current_open_price  # Use opening price for transaction
                    cash -= cost
                    in_position = True
                    entry_price = current_open_price  # Use opening price as entry price
                    trade_start_date = date
                    
                    self.trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': current_open_price,  # Opening price used for transaction
                        'shares': shares_to_buy,
                        'amount': cost,
                        'portfolio_value': cash + shares * current_close_price  # Use closing price for valuation
                    })
            
            # Execute sell using OPENING price
            elif should_sell:
                sale_amount = shares * current_open_price  # Use opening price for transaction
                cash += sale_amount
                
                profit = sale_amount - (shares * entry_price)
                profit_pct = ((current_open_price - entry_price) / entry_price) * 100
                
                self.trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': current_open_price,  # Opening price used for transaction
                    'shares': shares,
                    'amount': sale_amount,
                    'portfolio_value': cash,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_period': (date - trade_start_date).days if trade_start_date else 0
                })
                
                in_position = False
                shares = 0
            
            # Track portfolio value using closing price
            portfolio_value = cash + shares * current_close_price  # Use closing price for portfolio value
            portfolio_values.append(portfolio_value)
        
        self.data['Portfolio_Value'] = portfolio_values
        return self.trades

def main():
    # Get current language texts
    texts = LANGUAGES[st.session_state.language]
    
    st.set_page_config(page_title=texts['title'], layout="wide")
    st.title(texts['title'])
    st.markdown(texts['subtitle'])

    # Language and currency selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        selected_lang = st.selectbox(
            "🌐 เลือกภาษา / Choose Language / 选择语言",
            options=['th', 'en', 'zh'],
            format_func=lambda x: {'th': 'ไทย', 'en': 'English', 'zh': '中文'}[x],
            index=['th', 'en', 'zh'].index(st.session_state.language)
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
    
    with col3:
        selected_currency = st.selectbox(
            texts['currency_label'],
            options=['THB', 'USD', 'HKD'],
            format_func=lambda x: {'THB': 'THB (฿)', 'USD': 'USD ($)', 'HKD': 'HKD (HK$)'}[x],
            index=['THB', 'USD', 'HKD'].index(st.session_state.currency)
        )
        
        if selected_currency != st.session_state.currency:
            st.session_state.currency = selected_currency
            st.rerun()

    # Update texts after language change
    texts = LANGUAGES[st.session_state.language]
    
    # Define symbol groups with translations
    symbol_groups = {}
    if st.session_state.language == 'th':
        symbol_groups = {
            "หุ้น US": [
                ("AAPL", "Apple"),
                ("MSFT", "Microsoft"),
                ("GOOGL", "Google (Alphabet)"),
                ("AMZN", "Amazon"),
                ("META", "Meta Platforms (Facebook)"),
                ("NVDA", "NVIDIA"),
                ("TSLA", "Tesla"),
                ("JPM", "JPMorgan Chase"),
                ("JNJ", "Johnson & Johnson"),
                ("V", "Visa"),
                ("PG", "Procter & Gamble")
            ],
            "หุ้นเทคโนโลยีโลก": [
                ("TSM", "TSMC (ไต้หวัน)"),
                ("SONY", "Sony (ญี่ปุ่น)"),
                ("TM", "Toyota (ญี่ปุ่น)"),
                ("SAP", "SAP (เยอรมนี)"),
                ("ASML", "ASML (เนเธอร์แลนด์)"),
                ("NVO", "Novo Nordisk (เดนมาร์ก)"),
                ("NESN", "Nestlé (สวิตเซอร์แลนด์)")
            ],
            "หุ้นเอเชีย": [
                ("7203.T", "Toyota (ญี่ปุ่น)"),
                ("9984.T", "SoftBank (ญี่ปุ่น)"),
                ("005930.KS", "Samsung Electronics (เกาหลีใต้)"),
                ("6502.T", "Hitachi (ญี่ปุ่น)"),
                ("4689.T", "Fanuc (ญี่ปุ่น)"),
                ("BABA", "Alibaba (จีน)"),
                ("JD", "JD.com (จีน)")
            ],
            "หุ้นยุโรป": [
                ("NOKIA.HE", "Nokia (ฟินแลนด์)"),
                ("BMW.DE", "BMW (เยอรมนี)"),
                ("SIE.DE", "Siemens (เยอรมนี)"),
                ("AIR.PA", "Airbus (ฝรั่งเศส)"),
                ("SAN.PA", "Sanofi (ฝรั่งเศส)"),
                ("BP.L", "BP (อังกฤษ)"),
                ("RIO.L", "Rio Tinto (อังกฤษ)")
            ]
        }
    elif st.session_state.language == 'en':
        symbol_groups = {
            "US Stocks": [
                ("AAPL", "Apple"),
                ("MSFT", "Microsoft"),
                ("GOOGL", "Google (Alphabet)"),
                ("AMZN", "Amazon"),
                ("META", "Meta Platforms (Facebook)"),
                ("NVDA", "NVIDIA"),
                ("TSLA", "Tesla"),
                ("JPM", "JPMorgan Chase"),
                ("JNJ", "Johnson & Johnson"),
                ("V", "Visa"),
                ("PG", "Procter & Gamble")
            ],
            "Global Tech Stocks": [
                ("TSM", "TSMC (Taiwan)"),
                ("SONY", "Sony (Japan)"),
                ("TM", "Toyota (Japan)"),
                ("SAP", "SAP (Germany)"),
                ("ASML", "ASML (Netherlands)"),
                ("NVO", "Novo Nordisk (Denmark)"),
                ("NESN", "Nestlé (Switzerland)")
            ],
            "Asian Stocks": [
                ("7203.T", "Toyota (Japan)"),
                ("9984.T", "SoftBank (Japan)"),
                ("005930.KS", "Samsung Electronics (South Korea)"),
                ("6502.T", "Hitachi (Japan)"),
                ("4689.T", "Fanuc (Japan)"),
                ("BABA", "Alibaba (China)"),
                ("JD", "JD.com (China)")
            ],
            "European Stocks": [
                ("NOKIA.HE", "Nokia (Finland)"),
                ("BMW.DE", "BMW (Germany)"),
                ("SIE.DE", "Siemens (Germany)"),
                ("AIR.PA", "Airbus (France)"),
                ("SAN.PA", "Sanofi (France)"),
                ("BP.L", "BP (UK)"),
                ("RIO.L", "Rio Tinto (UK)")
            ]
        }
    else:  # zh
        symbol_groups = {
            "美股": [
                ("AAPL", "苹果"),
                ("MSFT", "微软"),
                ("GOOGL", "谷歌 (Alphabet)"),
                ("AMZN", "亚马逊"),
                ("META", "Meta平台 (Facebook)"),
                ("NVDA", "英伟达"),
                ("TSLA", "特斯拉"),
                ("JPM", "摩根大通"),
                ("JNJ", "强生公司"),
                ("V", "Visa"),
                ("PG", "宝洁公司")
            ],
            "全球科技股": [
                ("TSM", "台积电 (台湾)"),
                ("SONY", "索尼 (日本)"),
                ("TM", "丰田 (日本)"),
                ("SAP", "SAP (德国)"),
                ("ASML", "ASML (荷兰)"),
                ("NVO", "诺和诺德 (丹麦)"),
                ("NESN", "雀巢 (瑞士)")
            ],
            "亚洲股票": [
                ("7203.T", "丰田 (日本)"),
                ("9984.T", "软银 (日本)"),
                ("005930.KS", "三星电子 (韩国)"),
                ("6502.T", "日立 (日本)"),
                ("4689.T", "发那科 (日本)"),
                ("BABA", "阿里巴巴 (中国)"),
                ("JD", "京东 (中国)")
            ],
            "欧洲股票": [
                ("NOKIA.HE", "诺基亚 (芬兰)"),
                ("BMW.DE", "宝马 (德国)"),
                ("SIE.DE", "西门子 (德国)"),
                ("AIR.PA", "空中客车 (法国)"),
                ("SAN.PA", "赛诺菲 (法国)"),
                ("BP.L", "英国石油 (英国)"),
                ("RIO.L", "力拓 (英国)")
            ]
        }

    # Flatten all symbols with descriptions
    all_symbols = {}
    for category, symbols in symbol_groups.items():
        for symbol, name in symbols:
            all_symbols[f"{name} ({symbol})"] = symbol

    # Sidebar for inputs
    st.sidebar.header(texts['strategy_label'])

    # Symbol selection with dropdown
    symbol_option = st.sidebar.selectbox(
        texts['symbol_label'],
        options=list(all_symbols.keys()),
        format_func=lambda x: x
    )
    symbol = all_symbols[symbol_option]

    # Date range with new defaults
    col1, col2 = st.sidebar.columns(2)
    # Default start date to 2017/01/01
    start_date = col1.date_input(texts['start_date_label'], value=datetime(2017, 1, 1))
    # Default end date to last business day before today
    last_business_day = datetime.now() - timedelta(days=1)
    if last_business_day.weekday() >= 5:  # Weekend
        # Go back to Friday
        days_back = last_business_day.weekday() - 4
        last_business_day = last_business_day - timedelta(days=days_back)
    end_date = col2.date_input(texts['end_date_label'], value=last_business_day.date())

    # Initial capital (converted to selected currency)
    initial_capital_usd = st.sidebar.number_input(texts['capital_label'], value=10000, min_value=100, step=100)
    initial_capital_converted = initial_capital_usd * CURRENCY_RATES[st.session_state.currency]

    # Strategy selection
    st.sidebar.subheader(texts['strategy_label'])
    strategy_type = st.sidebar.selectbox(
        texts['strategy_label'],
        options=[
            "EMA Crossover",
            "SuperTrend",
            "Chaloke CDC",
            "Buy and Hold"
        ],
        index=0
    )

    # Initialize all strategy parameters with defaults first
    ema_fast = 12
    ema_slow = 26
    cdc_atr_multiplier = 1.5  # Default CDC values
    cdc_pivot_lookback = 5
    rsi_buy_threshold = 30
    rsi_sell_threshold = 70
    
    # Show indicators based on selected strategy
    if strategy_type in ["EMA Crossover"]:
        st.sidebar.subheader(texts['ema_settings'])
        ema_fast = st.sidebar.slider(texts['fast_ema'], 5, 50, 12)
        ema_slow = st.sidebar.slider(texts['slow_ema'], 5, 50, 26)
    elif strategy_type in ["Chaloke CDC"]:
        st.sidebar.subheader("CDC Settings")
        cdc_atr_multiplier = st.sidebar.slider("ATR Multiplier", 0.5, 3.0, 1.5, 0.1)
        cdc_pivot_lookback = st.sidebar.slider("Pivot Lookback", 2, 20, 5)
        ema_fast = 12  # Default values for EMA (needed for other calculations)
        ema_slow = 26
    else:
        pass  # Use default values already initialized above

    # RSI settings are only needed for RSI strategy (which was removed)
    if strategy_type in []:
        st.sidebar.subheader(texts['rsi_settings'])
        rsi_buy_threshold = st.sidebar.slider(texts['buy_threshold'], 10, 50, 30)
        rsi_sell_threshold = st.sidebar.slider(texts['sell_threshold'], 50, 90, 70)
    else:
        rsi_buy_threshold = 30  # Default values
        rsi_sell_threshold = 70  # Default values
        rsi_sell_threshold = 70

    # Position sizing
    st.sidebar.subheader(texts['position_size'])
    position_size = st.sidebar.slider(texts['size_percent'], 1, 100, 10) / 100

    # Risk management
    st.sidebar.subheader(texts['risk_management'])
    stop_loss = st.sidebar.slider(texts['stop_loss'], 0, 20, 0)  # 0 means disabled
    take_profit = st.sidebar.slider(texts['take_profit'], 0, 30, 0)  # 0 means disabled

    # Initialize session state
    if 'backtester' not in st.session_state:
        st.session_state.backtester = None
    if 'trades' not in st.session_state:
        st.session_state.trades = None

    # Run backtest button
    if st.sidebar.button(texts['run_backtest']):
        with st.spinner(f"{texts['run_backtest']} (may take a moment due to API rate limits)..."):
            try:
                backtester = MultiCurrencyBacktester(symbol, start_date, end_date, initial_capital_usd, st.session_state.currency)

                if backtester.load_data_with_delay():
                    backtester.add_indicators(ema_fast, ema_slow, 14, 3, 10)  # Always calculate SuperTrend with default parameters

                    # Generate signals to check if there are any
                    buy_signals, sell_signals = backtester.generate_signals_by_strategy(
                        strategy_type, ema_slow_window=ema_slow, rsi_buy_threshold=rsi_buy_threshold, rsi_sell_threshold=rsi_sell_threshold,
                        cdc_atr_multiplier=cdc_atr_multiplier, cdc_pivot_lookback=cdc_pivot_lookback
                    )

                    # Count signals
                    buy_count = sum(1 for signal in buy_signals if signal)
                    sell_count = sum(1 for signal in sell_signals if signal)

                    if buy_count == 0 and sell_count == 0:
                        st.warning(texts['no_signals'].format(strategy_type, symbol))
                        st.info(texts['try_different_params'])

                    # Convert percentage to decimal for stop loss and take profit
                    sl_pct = stop_loss if stop_loss > 0 else None
                    tp_pct = take_profit if take_profit > 0 else None

                    trades = backtester.run_backtest_by_strategy(
                        strategy_type,
                        position_size,
                        sl_pct,
                        tp_pct,
                        rsi_buy_threshold,
                        rsi_sell_threshold,
                        ema_fast,
                        ema_slow,
                        cdc_atr_multiplier,
                        cdc_pivot_lookback
                    )

                    st.session_state.backtester = backtester
                    st.session_state.trades = trades
                    st.success(f"{texts['backtest_complete']} {texts['signals_found'].format(buy_count, sell_count)}")
                else:
                    st.error(f"Failed to load data for the given symbol and date range")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

    # Main content
    if st.session_state.backtester and st.session_state.backtester.data is not None:
        data = st.session_state.backtester.data
        trades = st.session_state.trades

        # Display data summary in selected currency
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(texts['symbol'], symbol)
        col2.metric(texts['capital'], f"{st.session_state.currency} {initial_capital_converted:,.2f}", 
                   help=f"USD ${initial_capital_usd:,.2f} × {CURRENCY_RATES[st.session_state.currency]} {st.session_state.currency}/USD")
        col3.metric(texts['start'], start_date.strftime("%Y-%m-%d"))
        col4.metric(texts['end'], end_date.strftime("%Y-%m-%d"))
        col5, col6 = st.columns(2)
        col5.metric(texts['strategy'], strategy_type)
        col6.metric(texts['size_percent'], f"{position_size*100:.0f}%")

        # Show strategy-specific parameters
        if strategy_type in ["EMA Crossover"]:
            col7, col8 = st.columns(2)
            col7.metric(texts['fast_ema'], ema_fast)
            col8.metric(texts['slow_ema'], ema_slow)

        if strategy_type in []:
            col9, col10 = st.columns(2)
            col9.metric(texts['buy_threshold'], rsi_buy_threshold)
            col10.metric(texts['sell_threshold'], rsi_sell_threshold)

        # Create charts with converted prices
        currency_symbol = {'THB': '฿', 'USD': '$', 'HKD': 'HK$'}[st.session_state.currency]
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                texts['price_chart'].format(symbol),
                texts['rsi_chart'],
                texts['portfolio_chart'].format(st.session_state.currency)
            ),
            row_heights=[0.4, 0.3, 0.3]
        )

        # Price and EMAs (converted to selected currency)
        fig.add_trace(go.Scatter(x=data.index, y=data['Close']*CURRENCY_RATES[st.session_state.currency], name=f'Close ({st.session_state.currency})', line=dict(color='black')), row=1, col=1)
        if strategy_type in ["EMA Crossover"]:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast']*CURRENCY_RATES[st.session_state.currency], name=f'EMA{ema_fast} ({st.session_state.currency})', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow']*CURRENCY_RATES[st.session_state.currency], name=f'EMA{ema_slow} ({st.session_state.currency})', line=dict(color='blue')), row=1, col=1)
        if strategy_type in ["SuperTrend"] and 'SuperTrend' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend']*CURRENCY_RATES[st.session_state.currency], name=f'SuperTrend ({st.session_state.currency})', line=dict(color='red', dash='dash')), row=1, col=1)

        # Add buy/sell markers
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']

            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices_converted = [t['price'] * CURRENCY_RATES[st.session_state.currency] for t in buy_trades]
                fig.add_trace(go.Scatter(
                    x=buy_dates, 
                    y=buy_prices_converted, 
                    mode='markers', 
                    name='Buy Signals', 
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ), row=1, col=1)

            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices_converted = [t['price'] * CURRENCY_RATES[st.session_state.currency] for t in sell_trades]
                fig.add_trace(go.Scatter(
                    x=sell_dates, 
                    y=sell_prices_converted, 
                    mode='markers', 
                    name='Sell Signals', 
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ), row=1, col=1)

        # RSI - Only show for RSI strategy (removed)
        if strategy_type in []:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=rsi_sell_threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text=texts['sell_level'])
            fig.add_hline(y=rsi_buy_threshold, line_dash="dash", line_color="green", row=2, col=1, annotation_text=texts['buy_level'])

        # Portfolio value in selected currency
        if 'Portfolio_Value' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value']*CURRENCY_RATES[st.session_state.currency], name=f'Portfolio Value ({st.session_state.currency})', line=dict(color='blue')), row=3, col=1)

        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Trading results in selected currency
        if trades:
            st.subheader(texts['results'])

            # Count buy and sell trades
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']

            # Calculate performance metrics
            total_trades = len(sell_trades)  # Only completed trades (buy + sell)
            winning_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
            losing_trades = len([t for t in sell_trades if t.get('profit', 0) < 0])

            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

            # Final portfolio value in selected currency
            final_value_usd = data['Portfolio_Value'].iloc[-1] if 'Portfolio_Value' in data.columns else initial_capital_usd
            final_value_converted = final_value_usd * CURRENCY_RATES[st.session_state.currency]
            total_return = (final_value_usd - initial_capital_usd) / initial_capital_usd * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(texts['total_trades'], total_trades)
            col2.metric(texts['win_rate'], f"{win_rate:.2f}%")
            col3.metric(texts['final_value'], f"{st.session_state.currency} {final_value_converted:,.2f}",
                       help=f"USD ${final_value_usd:,.2f} × {CURRENCY_RATES[st.session_state.currency]} {st.session_state.currency}/USD")
            col4.metric(texts['total_return'], f"{total_return:.2f}%")

            # Detailed trade log - show pairs of buy/sell transactions
            st.subheader(texts['trade_log'])
            
            # Pair up buy and sell transactions - each sell corresponds to the most recent buy
            trade_pairs = []
            available_buys = buy_trades.copy()
            
            for sell_trade in sell_trades:
                if available_buys:  # If there are available buy trades to match
                    # Take the most recent buy (FIFO - First In, First Out)
                    buy_trade = available_buys.pop(0)
                    
                    trade_pairs.append({
                        'buy_date': buy_trade['date'].strftime('%Y-%m-%d'),  # Format date without time
                        'buy_price': buy_trade['price'],
                        'buy_price_converted': buy_trade['price'] * CURRENCY_RATES[st.session_state.currency],
                        'sell_date': sell_trade['date'].strftime('%Y-%m-%d'),  # Format date without time
                        'sell_price': sell_trade['price'],
                        'sell_price_converted': sell_trade['price'] * CURRENCY_RATES[st.session_state.currency],
                        'shares': buy_trade['shares'],
                        'profit_usd': sell_trade['profit'],
                        'profit_converted': sell_trade['profit'] * CURRENCY_RATES[st.session_state.currency],
                        'profit_pct': sell_trade['profit_pct'],
                        'holding_period': sell_trade['holding_period']
                    })
            
            if trade_pairs:
                trade_pairs_df = pd.DataFrame(trade_pairs)
                trade_pairs_df = trade_pairs_df.rename(columns={
                    'buy_date': texts['buy_date'],
                    'buy_price_converted': f"{texts['buy_price']} ({st.session_state.currency})",
                    'sell_date': texts['sell_date'],
                    'sell_price_converted': f"{texts['sell_price']} ({st.session_state.currency})",
                    'shares': texts['shares'],
                    'profit_converted': f"{texts['profit']} ({st.session_state.currency})",
                    'profit_pct': texts['profit_pct'],
                    'holding_period': texts['holding_period']
                })

                # Format the DataFrame to show converted amounts
                display_cols = [texts['buy_date'], f"{texts['buy_price']} ({st.session_state.currency})", 
                               texts['sell_date'], f"{texts['sell_price']} ({st.session_state.currency})", 
                               texts['shares'], f"{texts['profit']} ({st.session_state.currency})", 
                               texts['profit_pct'], texts['holding_period']]
                               
                st.dataframe(trade_pairs_df[display_cols].style.format({
                    f"{texts['buy_price']} ({st.session_state.currency})": f'{currency_symbol}{{:,.2f}}',
                    f"{texts['sell_price']} ({st.session_state.currency})": f'{currency_symbol}{{:,.2f}}',
                    f"{texts['profit']} ({st.session_state.currency})": f'{currency_symbol}{{:,.2f}}',
                    texts['profit_pct']: '{:.2f}%',
                    texts['holding_period']: '{:.0f}'
                }))
            else:
                st.info(texts['no_completed_trades'])
        else:
            st.info(f"{texts['run_backtest']} to see results")

    else:
        st.info(f"Enter parameters and click '{texts['run_backtest']}' to start")

if __name__ == "__main__":
    main()