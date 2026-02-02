import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random

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

class FixedBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.positions = []
        self.trades = []

    def load_data_with_delay(self):
        """Load historical data with delay to avoid rate limits"""
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=self.start_date, end=self.end_date, interval="1d")
        return not self.data.empty

    def add_indicators(self, ema_fast_window=12, ema_slow_window=26, rsi_window=14):
        """Add technical indicators using closing prices"""
        self.data['EMA_Fast'] = calculate_ema(self.data['Close'], ema_fast_window)
        self.data['EMA_Slow'] = calculate_ema(self.data['Close'], ema_slow_window)
        self.data['RSI'] = calculate_rsi(self.data['Close'], rsi_window)
        self.data['ATR'] = calculate_atr(self.data, 14)

    def generate_signals_by_strategy(self, strategy_type, rsi_buy_threshold=30, rsi_sell_threshold=70):
        """Generate buy/sell signals based on selected strategy using closing prices"""
        buy_signals = []
        sell_signals = []
        
        # Ensure we have enough data points for EMA calculations
        for i in range(max(ema_slow_window, rsi_window), len(self.data)):  # Using max window for safety
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
                    if prev_row['RSI'] <= rsi_buy_threshold < current_row['RSI']:
                        buy_signal = True
                    # Sell when RSI is above threshold (using closing prices for indicator)
                    elif prev_row['RSI'] >= rsi_sell_threshold > current_row['RSI']:
                        sell_signal = True
            
            elif strategy_type == "Combined":
                # Combined strategy using both EMA and RSI (using closing prices for indicators)
                # Buy when EMA bullish AND RSI bullish
                ema_bullish = (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] > current_row['EMA_Slow'])
                rsi_bullish = not pd.isna(prev_row['RSI']) and not pd.isna(current_row['RSI']) and \
                              prev_row['RSI'] <= rsi_buy_threshold < current_row['RSI']
                
                # Sell when EMA bearish AND RSI bearish
                ema_bearish = (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] < current_row['EMA_Slow'])
                rsi_bearish = not pd.isna(prev_row['RSI']) and not pd.isna(current_row['RSI']) and \
                              prev_row['RSI'] >= rsi_sell_threshold > current_row['RSI']
                
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
                                rsi_buy_threshold=30, rsi_sell_threshold=70, ema_fast_window=12, ema_slow_window=26):
        """Run backtest with predefined strategy using closing prices for indicators and opening prices for transactions"""
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        in_position = False
        entry_price = 0
        trade_start_date = None
        
        # Generate signals based on strategy (using closing prices for indicators)
        buy_signals, sell_signals = self.generate_signals_by_strategy(
            strategy_type, rsi_buy_threshold, rsi_sell_threshold
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
    st.set_page_config(page_title="Fixed Investment Backtester (Corrected)", layout="wide")
    st.title("📈 ระบบย้อนกลับการลงทุน (หน่วยเงินบาท) - ฉบับแก้ไข")
    st.markdown("""
    ระบบย้อนกลับการลงทุนที่ใช้หน่วยเงินบาทไทย โดยใช้กลยุทธ์ EMA และ RSI ที่คำนวณจากราคาปิดและดำเนินการซื้อขายที่ราคาเปิด
    """)

    # Currency conversion helper (assuming 1 USD = 30 THB)
    USD_TO_THB = 30
    
    # Define symbol groups with Thai equivalents where possible
    symbol_groups = {
        "ดัชนี": [
            ("^GSPC", "S&P 500"),
            ("^STI", "ดัชนี SET"),
            ("^SET50", "ดัชนี SET 50"),
            ("^SET100", "ดัชนี SET 100")
        ],
        "หุ้น US": [
            ("AAPL", "Apple"),
            ("NVDA", "NVIDIA"),
            ("MSFT", "Microsoft"),
            ("GOOGL", "Google"),
            ("AMZN", "Amazon"),
            ("TSLA", "Tesla")
        ],
        "หุ้นต่างประเทศ": [
            ("01810.HK", "Xiaomi"),
            ("2330.TW", "TSMC"),
            ("BMW.DE", "BMW"),
            ("NOKIA.HE", "Nokia")
        ],
        "หุ้นไทย": [
            ("PTT.BK", "PTT"),
            ("SCC.BK", "ซีเมนต์ไทย"),
            ("CPALL.BK", "CP ALL"),
            ("KBANK.BK", "ธนาคารกสิกรไทย"),
            ("TRUE.BK", "TRUE")
        ],
        "สินทรัพย์อื่นๆ": [
            ("GC=F", "ทองคำ"),
            ("CL=F", "น้ำมันดิบ"),
            ("BTC-USD", "Bitcoin"),
            ("ETH-USD", "Ethereum"),
            ("XAU=", "ทองคำ SPOT")
        ]
    }
    
    # Flatten all symbols with descriptions
    all_symbols = {}
    for category, symbols in symbol_groups.items():
        for symbol, name in symbols:
            all_symbols[f"{name} ({symbol})"] = symbol
    
    # Sidebar for inputs
    st.sidebar.header("การตั้งค่าการย้อนกลับ")
    
    # Symbol selection with dropdown
    symbol_option = st.sidebar.selectbox(
        "สัญลักษณ์หุ้น/สินทรัพย์",
        options=list(all_symbols.keys()),
        format_func=lambda x: x
    )
    symbol = all_symbols[symbol_option]
    
    # Date range with new defaults
    col1, col2 = st.sidebar.columns(2)
    # Default start date to 2017/01/01
    start_date = col1.date_input("วันเริ่มต้น", value=datetime(2017, 1, 1))
    # Default end date to last business day before today
    last_business_day = datetime.now() - timedelta(days=1)
    if last_business_day.weekday() >= 5:  # Weekend
        # Go back to Friday
        days_back = last_business_day.weekday() - 4
        last_business_day = last_business_day - timedelta(days=days_back)
    end_date = col2.date_input("วันสิ้นสุด", value=last_business_day.date())
    
    # Initial capital (converted to THB)
    initial_capital_usd = st.sidebar.number_input("ทุนเริ่มต้น ($)", value=10000, min_value=100, step=100)
    initial_capital_thb = initial_capital_usd * USD_TO_THB
    
    # Strategy selection
    st.sidebar.subheader("กลยุทธ์การเทรด")
    strategy_type = st.sidebar.selectbox(
        "เลือกประเภทกลยุทธ์",
        options=[
            "EMA Crossover",
            "RSI Oversold/Oversold",
            "Combined"
        ],
        index=0
    )
    
    # Show indicators based on selected strategy
    if strategy_type in ["EMA Crossover", "Combined"]:
        st.sidebar.subheader("การตั้งค่า EMA")
        ema_fast = st.sidebar.slider("หน้าต่าง EMA เร็ว", 5, 50, 12)
        ema_slow = st.sidebar.slider("หน้าต่าง EMA ช้า", 5, 50, 26)
    else:
        ema_fast = 12  # Default values
        ema_slow = 26
    
    if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
        st.sidebar.subheader("การตั้งค่า RSI")
        rsi_buy_threshold = st.sidebar.slider("เกณฑ์ซื้อ RSI", 10, 50, 30)
        rsi_sell_threshold = st.sidebar.slider("เกณฑ์ขาย RSI", 50, 90, 70)
    else:
        rsi_buy_threshold = 30  # Default values
        rsi_sell_threshold = 70
    
    # Position sizing
    st.sidebar.subheader("ขนาดตำแหน่ง")
    position_size = st.sidebar.slider("ขนาดตำแหน่ง (%)", 1, 100, 10) / 100
    
    # Risk management
    st.sidebar.subheader("การจัดการความเสี่ยง")
    stop_loss = st.sidebar.slider("หยุดขาดทุน (%)", 0, 20, 0)  # 0 means disabled
    take_profit = st.sidebar.slider("ทำกำไร (%)", 0, 30, 0)  # 0 means disabled
    
    # Initialize session state
    if 'backtester' not in st.session_state:
        st.session_state.backtester = None
    if 'trades' not in st.session_state:
        st.session_state.trades = None

    # Run backtest button
    if st.sidebar.button("เริ่มการย้อนกลับ"):
        with st.spinner("กำลังดำเนินการย้อนกลับ (อาจใช้เวลาสักครู่เนื่องจากมีการจำกัดความถี่ของ API)..."):
            try:
                backtester = FixedBacktester(symbol, start_date, end_date, initial_capital_usd)
                
                if backtester.load_data_with_delay():
                    backtester.add_indicators(ema_fast, ema_slow, 14)
                    
                    # Generate signals to check if there are any
                    buy_signals, sell_signals = backtester.generate_signals_by_strategy(
                        strategy_type, rsi_buy_threshold, rsi_sell_threshold
                    )
                    
                    # Count signals
                    buy_count = sum(1 for signal in buy_signals if signal)
                    sell_count = sum(1 for signal in sell_signals if signal)
                    
                    if buy_count == 0 and sell_count == 0:
                        st.warning(f"ไม่พบสัญญาณการซื้อหรือขายสำหรับกลยุทธ์ {strategy_type} บนสินทรัพย์ {symbol}")
                        st.info("อาจเป็นเพราะช่วงเวลาที่เลือกไม่มีการเคลื่อนไหวที่เหมาะสม หรือพารามิเตอร์ที่ตั้งไว้ไม่เหมาะสม")
                    
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
                        ema_slow
                    )
                    
                    st.session_state.backtester = backtester
                    st.session_state.trades = trades
                    st.success(f"การย้อนกลับเสร็จสมบูรณ์! พบสัญญาณซื้อ {buy_count} ครั้ง และสัญญาณขาย {sell_count} ครั้ง")
                else:
                    st.error("ไม่สามารถโหลดข้อมูลสำหรับสัญลักษณ์และช่วงวันที่กำหนดได้")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการย้อนกลับ: {str(e)}")

    # Main content
    if st.session_state.backtester and st.session_state.backtester.data is not None:
        data = st.session_state.backtester.data
        trades = st.session_state.trades
        
        # Display data summary in THB
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("สัญลักษณ์", symbol)
        col2.metric("ทุนเริ่มต้น (THB)", f"฿{initial_capital_thb:,.2f}", 
                   help=f"USD ${initial_capital_usd:,.2f} × {USD_TO_THB} THB/USD")
        col3.metric("วันเริ่มต้น", start_date.strftime("%Y-%m-%d"))
        col4.metric("วันสิ้นสุด", end_date.strftime("%Y-%m-%d"))
        col5, col6 = st.columns(2)
        col5.metric("กลยุทธ์", strategy_type)
        col6.metric("ขนาดตำแหน่ง", f"{position_size*100:.0f}%")
        
        # Show strategy-specific parameters
        if strategy_type in ["EMA Crossover", "Combined"]:
            col7, col8 = st.columns(2)
            col7.metric("EMA เร็ว", ema_fast)
            col8.metric("EMA ช้า", ema_slow)
        
        if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
            col9, col10 = st.columns(2)
            col9.metric("RSI ซื้อ", rsi_buy_threshold)
            col10.metric("RSI ขาย", rsi_sell_threshold)
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f'{symbol} ราคา & EMAs', 'RSI', 'มูลค่าพอร์ต (THB)'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and EMAs
        fig.add_trace(go.Scatter(x=data.index, y=data['Close']*USD_TO_THB, name='ปิด (THB)', line=dict(color='black')), row=1, col=1)
        if strategy_type in ["EMA Crossover", "Combined"]:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast']*USD_TO_THB, name=f'EMA{ema_fast} (THB)', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow']*USD_TO_THB, name=f'EMA{ema_slow} (THB)', line=dict(color='blue')), row=1, col=1)
        
        # Add buy/sell markers
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices_thb = [t['price'] * USD_TO_THB for t in buy_trades]
                fig.add_trace(go.Scatter(
                    x=buy_dates, 
                    y=buy_prices_thb, 
                    mode='markers', 
                    name='สัญญาณซื้อ', 
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ), row=1, col=1)
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices_thb = [t['price'] * USD_TO_THB for t in sell_trades]
                fig.add_trace(go.Scatter(
                    x=sell_dates, 
                    y=sell_prices_thb, 
                    mode='markers', 
                    name='สัญญาณขาย', 
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ), row=1, col=1)
        
        # RSI
        if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=rsi_sell_threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text="ระดับขาย")
            fig.add_hline(y=rsi_buy_threshold, line_dash="dash", line_color="green", row=2, col=1, annotation_text="ระดับซื้อ")
        
        # Portfolio value in THB
        if 'Portfolio_Value' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value']*USD_TO_THB, name='มูลค่าพอร์ต (THB)', line=dict(color='blue')), row=3, col=1)
        
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading results in THB
        if trades:
            st.subheader("ผลลัพธ์การเทรด")
            
            # Count buy and sell trades
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            # Calculate performance metrics
            total_trades = len(sell_trades)  # Only completed trades (buy + sell)
            winning_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
            losing_trades = len([t for t in sell_trades if t.get('profit', 0) < 0])
            
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # Final portfolio value in THB
            final_value_usd = data['Portfolio_Value'].iloc[-1] if 'Portfolio_Value' in data.columns else initial_capital_usd
            final_value_thb = final_value_usd * USD_TO_THB
            total_return = (final_value_usd - initial_capital_usd) / initial_capital_usd * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("จำนวนเทรดทั้งหมด", total_trades)
            col2.metric("อัตราชนะ", f"{win_rate:.2f}%")
            col3.metric("มูลค่าสุดท้าย (THB)", f"฿{final_value_thb:,.2f}",
                       help=f"USD ${final_value_usd:,.2f} × {USD_TO_THB} THB/USD")
            col4.metric("ผลตอบแทนรวม", f"{total_return:.2f}%")
            
            # Detailed trade log - show pairs of buy/sell transactions
            st.subheader("บันทึกการเทรด (คู่ซื้อ/ขาย)")
            if sell_trades:
                # Pair up buy and sell transactions
                trade_pairs = []
                buy_iter = iter(buy_trades)
                sell_iter = iter(sell_trades)
                
                try:
                    current_buy = next(buy_iter)
                    for current_sell in sell_iter:
                        trade_pairs.append({
                            'buy_date': current_buy['date'],
                            'buy_price': current_buy['price'],
                            'buy_price_thb': current_buy['price'] * USD_TO_THB,
                            'sell_date': current_sell['date'],
                            'sell_price': current_sell['price'],
                            'sell_price_thb': current_sell['price'] * USD_TO_THB,
                            'shares': current_buy['shares'],
                            'profit_usd': current_sell['profit'],
                            'profit_thb': current_sell['profit'] * USD_TO_THB,
                            'profit_pct': current_sell['profit_pct'],
                            'holding_period': current_sell['holding_period']
                        })
                        
                        # Get next buy for the next pair
                        current_buy = next(buy_iter)
                except StopIteration:
                    # We've exhausted either buys or sells
                    pass
                
                if trade_pairs:
                    trade_pairs_df = pd.DataFrame(trade_pairs)
                    trade_pairs_df = trade_pairs_df.rename(columns={
                        'buy_date': 'วันที่ซื้อ',
                        'buy_price_thb': 'ราคาซื้อ (THB)',
                        'sell_date': 'วันที่ขาย',
                        'sell_price_thb': 'ราคาขาย (THB)',
                        'shares': 'จำนวนหุ้น',
                        'profit_thb': 'กำไร (THB)',
                        'profit_pct': 'กำไร %',
                        'holding_period': 'ถือครอง (วัน)'
                    })
                    
                    # Format the DataFrame to show THB amounts
                    st.dataframe(trade_pairs_df[['วันที่ซื้อ', 'ราคาซื้อ (THB)', 'วันที่ขาย', 'ราคาขาย (THB)', 'จำนวนหุ้น', 'กำไร (THB)', 'กำไร %', 'ถือครอง (วัน)']].style.format({
                        'ราคาซื้อ (THB)': '฿{:,.2f}',
                        'ราคาขาย (THB)': '฿{:,.2f}',
                        'กำไร (THB)': '฿{:,.2f}',
                        'กำไร %': '{:.2f}%',
                        'ถือครอง (วัน)': '{:.0f}'
                    }))
                else:
                    st.info("ไม่มีคู่ซื้อ/ขายที่แสดง")
            else:
                st.info("ไม่มีการเทรดที่เสร็จสมบูรณ์ (ซื้อ + ขาย)")
        else:
            st.info("เริ่มการย้อนกลับเพื่อดูผลลัพธ์")
    
    else:
        st.info("ป้อนพารามิเตอร์และคลิก 'เริ่มการย้อนกลับ' เพื่อเริ่มต้น")

if __name__ == "__main__":
    main()