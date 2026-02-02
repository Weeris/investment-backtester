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

class FinalBacktester:
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
        for i in range(26, len(self.data)):  # Using 26 as max window for EMA
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
    st.set_page_config(page_title="Final Investment Backtester", layout="wide")
    st.title("📈 Final Investment Backtesting Platform")
    st.markdown("""
    Final backtesting with EMA and RSI strategies using closing prices for indicators and opening prices for transactions.
    """)
    
    # Define symbol groups
    symbol_groups = {
        "Indices": [
            ("^GSPC", "S&P 500"),
            ("^STI", "SET Index"),
            ("^SET50", "SET 50"),
            ("^SET100", "SET 100")
        ],
        "US Stocks": [
            ("AAPL", "Apple"),
            ("NVDA", "NVIDIA"),
            ("MSFT", "Microsoft"),
            ("GOOGL", "Google"),
            ("AMZN", "Amazon"),
            ("TSLA", "Tesla")
        ],
        "International Stocks": [
            ("01810.HK", "Xiaomi"),
            ("2330.TW", "TSMC"),
            ("BMW.DE", "BMW"),
            ("NOKIA.HE", "Nokia")
        ],
        "Thai Stocks": [
            ("PTT.BK", "PTT"),
            ("SCC.BK", "Siam Cement"),
            ("CPALL.BK", "CP ALL"),
            ("KBANK.BK", "Kasikornbank"),
            ("TRUE.BK", "TRUE Corporation")
        ],
        "Commodities & Crypto": [
            ("GC=F", "Gold"),
            ("CL=F", "Crude Oil"),
            ("BTC-USD", "Bitcoin"),
            ("ETH-USD", "Ethereum"),
            ("XAU=", "Gold Spot")
        ]
    }
    
    # Flatten all symbols with descriptions
    all_symbols = {}
    for category, symbols in symbol_groups.items():
        for symbol, name in symbols:
            all_symbols[f"{name} ({symbol})"] = symbol
    
    # Sidebar for inputs
    st.sidebar.header("Backtest Settings")
    
    # Symbol selection with dropdown
    symbol_option = st.sidebar.selectbox(
        "Stock/Symbol",
        options=list(all_symbols.keys()),
        format_func=lambda x: x
    )
    symbol = all_symbols[symbol_option]
    
    # Date range with new defaults
    col1, col2 = st.sidebar.columns(2)
    # Default start date to 2017/01/01
    start_date = col1.date_input("Start Date", value=datetime(2017, 1, 1))
    # Default end date to last business day before today
    last_business_day = datetime.now() - timedelta(days=1)
    if last_business_day.weekday() >= 5:  # Weekend
        # Go back to Friday
        days_back = last_business_day.weekday() - 4
        last_business_day = last_business_day - timedelta(days=days_back)
    end_date = col2.date_input("End Date", value=last_business_day.date())
    
    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=100, step=100)
    
    # Strategy selection
    st.sidebar.subheader("Trading Strategy")
    strategy_type = st.sidebar.selectbox(
        "Choose Strategy Type",
        options=[
            "EMA Crossover",
            "RSI Oversold/Oversold",
            "Combined"
        ],
        index=0
    )
    
    # Show indicators based on selected strategy
    if strategy_type in ["EMA Crossover", "Combined"]:
        st.sidebar.subheader("EMA Settings")
        ema_fast = st.sidebar.slider("Fast EMA Window", 5, 50, 12)
        ema_slow = st.sidebar.slider("Slow EMA Window", 5, 50, 26)
    else:
        ema_fast = 12  # Default values
        ema_slow = 26
    
    if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
        st.sidebar.subheader("RSI Settings")
        rsi_buy_threshold = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
        rsi_sell_threshold = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
    else:
        rsi_buy_threshold = 30  # Default values
        rsi_sell_threshold = 70
    
    # Position sizing
    st.sidebar.subheader("Position Sizing")
    position_size = st.sidebar.slider("Position Size (%)", 1, 100, 10) / 100
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0, 20, 0)  # 0 means disabled
    take_profit = st.sidebar.slider("Take Profit (%)", 0, 30, 0)  # 0 means disabled
    
    # Initialize session state
    if 'backtester' not in st.session_state:
        st.session_state.backtester = None
    if 'trades' not in st.session_state:
        st.session_state.trades = None

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest (this may take a moment due to API rate limits)..."):
            try:
                backtester = FinalBacktester(symbol, start_date, end_date, initial_capital)
                
                if backtester.load_data_with_delay():
                    backtester.add_indicators(ema_fast, ema_slow, 14)
                    
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
                    st.success("Backtest completed successfully!")
                else:
                    st.error("Failed to load data for the given symbol and date range")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

    # Main content
    if st.session_state.backtester and st.session_state.backtester.data is not None:
        data = st.session_state.backtester.data
        trades = st.session_state.trades
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbol", symbol)
        col2.metric("Initial Capital", f"${initial_capital:,}")
        col3.metric("Start Date", start_date.strftime("%Y-%m-%d"))
        col4.metric("End Date", end_date.strftime("%Y-%m-%d"))
        col5, col6 = st.columns(2)
        col5.metric("Strategy", strategy_type)
        col6.metric("Position Size", f"{position_size*100:.0f}%")
        
        # Show strategy-specific parameters
        if strategy_type in ["EMA Crossover", "Combined"]:
            col7, col8 = st.columns(2)
            col7.metric("Fast EMA", ema_fast)
            col8.metric("Slow EMA", ema_slow)
        
        if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
            col9, col10 = st.columns(2)
            col9.metric("RSI Buy", rsi_buy_threshold)
            col10.metric("RSI Sell", rsi_sell_threshold)
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f'{symbol} Price & EMAs', 'RSI', 'Portfolio Value'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and EMAs
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
        if strategy_type in ["EMA Crossover", "Combined"]:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast'], name=f'EMA{ema_fast}', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow'], name=f'EMA{ema_slow}', line=dict(color='blue')), row=1, col=1)
        
        # Add buy/sell markers
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if buy_trades:
                buy_dates = [t['date'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                fig.add_trace(go.Scatter(
                    x=buy_dates, 
                    y=buy_prices, 
                    mode='markers', 
                    name='Buy Signal', 
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ), row=1, col=1)
            
            if sell_trades:
                sell_dates = [t['date'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                fig.add_trace(go.Scatter(
                    x=sell_dates, 
                    y=sell_prices, 
                    mode='markers', 
                    name='Sell Signal', 
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ), row=1, col=1)
        
        # RSI
        if strategy_type in ["RSI Oversold/Oversold", "Combined"]:
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=rsi_sell_threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Sell Level")
            fig.add_hline(y=rsi_buy_threshold, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Buy Level")
        
        # Portfolio value
        if 'Portfolio_Value' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value'], name='Portfolio Value', line=dict(color='blue')), row=3, col=1)
        
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading results
        if trades:
            st.subheader("Trading Results")
            
            # Count buy and sell trades
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            # Calculate performance metrics
            total_trades = len(sell_trades)  # Only completed trades (buy + sell)
            winning_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
            losing_trades = len([t for t in sell_trades if t.get('profit', 0) < 0])
            
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # Final portfolio value
            final_value = data['Portfolio_Value'].iloc[-1] if 'Portfolio_Value' in data.columns else initial_capital
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2f}%")
            col3.metric("Final Value", f"${final_value:,.2f}")
            col4.metric("Total Return", f"{total_return:.2f}%")
            
            # Detailed trade log - show pairs of buy/sell transactions
            st.subheader("Trade Log (Buy/Sell Pairs)")
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
                            'sell_date': current_sell['date'],
                            'sell_price': current_sell['price'],
                            'shares': current_buy['shares'],
                            'profit': current_sell['profit'],
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
                        'buy_date': 'Buy Date',
                        'buy_price': 'Buy Price',
                        'sell_date': 'Sell Date',
                        'sell_price': 'Sell Price',
                        'shares': 'Shares',
                        'profit': 'Profit',
                        'profit_pct': 'Profit %',
                        'holding_period': 'Hold Days'
                    })
                    
                    st.dataframe(trade_pairs_df.style.format({
                        'Buy Price': '${:.2f}',
                        'Sell Price': '${:.2f}',
                        'Profit': '${:.2f}',
                        'Profit %': '{:.2f}%',
                        'Hold Days': '{:.0f}'
                    }))
                else:
                    st.info("No paired buy/sell transactions to display")
            else:
                st.info("No completed trades (buy + sell) to display")
        else:
            st.info("Run backtest to see results")
    
    else:
        st.info("Enter parameters and click 'Run Backtest' to start")

if __name__ == "__main__":
    main()