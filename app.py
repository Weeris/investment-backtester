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
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_atr(data, window=14):
    """Calculate Average True Range for volatility"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

class AdvancedBacktester:
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
        """Add technical indicators"""
        self.data['EMA_Fast'] = calculate_ema(self.data['Close'], ema_fast_window)
        self.data['EMA_Slow'] = calculate_ema(self.data['Close'], ema_slow_window)
        self.data['RSI'] = calculate_rsi(self.data['Close'], rsi_window)
        
        # Additional indicators
        macd, macd_signal, macd_hist = calculate_macd(self.data['Close'])
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Histogram'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(self.data['Close'])
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower
        
        self.data['ATR'] = calculate_atr(self.data, 14)

    def generate_signals_by_strategy(self, strategy_type, rsi_buy_threshold=30, rsi_sell_threshold=70, bb_buy_threshold=20, bb_sell_threshold=80):
        """Generate buy/sell signals based on selected strategy"""
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(self.data)):  # Start from 1 to compare with previous
            current_row = self.data.iloc[i]
            prev_row = self.data.iloc[i-1]
            
            buy_signal = False
            sell_signal = False
            
            if strategy_type == "EMA Crossover":
                # Buy when fast EMA crosses above slow EMA
                if (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] > current_row['EMA_Slow']):
                    buy_signal = True
                # Sell when fast EMA crosses below slow EMA
                elif (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] < current_row['EMA_Slow']):
                    sell_signal = True
            
            elif strategy_type == "RSI Oversold/Oversold":
                # Buy when RSI is below threshold (oversold)
                if prev_row['RSI'] <= rsi_buy_threshold < current_row['RSI']:
                    buy_signal = True
                # Sell when RSI is above threshold (overbought)
                elif prev_row['RSI'] >= rsi_sell_threshold > current_row['RSI']:
                    sell_signal = True
            
            elif strategy_type == "Bollinger Bands":
                # Buy when price touches lower band
                if (prev_row['Close'] > prev_row['BB_Lower']) and (current_row['Close'] <= current_row['BB_Lower']):
                    buy_signal = True
                # Sell when price touches upper band
                elif (prev_row['Close'] < prev_row['BB_Upper']) and (current_row['Close'] >= current_row['BB_Upper']):
                    sell_signal = True
            
            elif strategy_type == "MACD Crossover":
                # Buy when MACD line crosses above signal line
                if (prev_row['MACD'] <= prev_row['MACD_Signal']) and (current_row['MACD'] > current_row['MACD_Signal']):
                    buy_signal = True
                # Sell when MACD line crosses below signal line
                elif (prev_row['MACD'] >= prev_row['MACD_Signal']) and (current_row['MACD'] < current_row['MACD_Signal']):
                    sell_signal = True
            
            elif strategy_type == "Combined":
                # Combined strategy using multiple indicators
                ema_cross_up = (prev_row['EMA_Fast'] <= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] > current_row['EMA_Slow'])
                rsi_oversold = prev_row['RSI'] <= rsi_buy_threshold < current_row['RSI']
                bb_touch_lower = (prev_row['Close'] > prev_row['BB_Lower']) and (current_row['Close'] <= current_row['BB_Lower'])
                
                ema_cross_down = (prev_row['EMA_Fast'] >= prev_row['EMA_Slow']) and (current_row['EMA_Fast'] < current_row['EMA_Slow'])
                rsi_overbought = prev_row['RSI'] >= rsi_sell_threshold > current_row['RSI']
                bb_touch_upper = (prev_row['Close'] < prev_row['BB_Upper']) and (current_row['Close'] >= current_row['BB_Upper'])
                
                # Buy when multiple indicators align
                if (ema_cross_up or rsi_oversold or bb_touch_lower):
                    buy_signal = True
                # Sell when multiple indicators align
                if (ema_cross_down or rsi_overbought or bb_touch_upper):
                    sell_signal = True
            
            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)
        
        # Add False for the first element since we don't have previous data
        buy_signals.insert(0, False)
        sell_signals.insert(0, False)
        
        return buy_signals, sell_signals

    def run_backtest_by_strategy(self, strategy_type, position_size_pct=0.1, stop_loss_pct=None, take_profit_pct=None, 
                                rsi_buy_threshold=30, rsi_sell_threshold=70, bb_buy_threshold=20, bb_sell_threshold=80):
        """Run backtest with predefined strategy"""
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        in_position = False
        entry_price = 0
        trade_start_date = None
        
        # Generate signals based on strategy
        buy_signals, sell_signals = self.generate_signals_by_strategy(
            strategy_type, rsi_buy_threshold, rsi_sell_threshold, bb_buy_threshold, bb_sell_threshold
        )
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            current_price = row['Close']
            
            should_buy = buy_signals[i] and not in_position and cash > 0
            should_sell = sell_signals[i] and in_position and shares > 0
            
            # Apply stop loss and take profit if in position
            if in_position:
                current_profit_pct = (current_price - entry_price) / entry_price
                
                # Stop loss
                if stop_loss_pct and current_profit_pct <= -stop_loss_pct/100:
                    should_sell = True
                
                # Take profit
                if take_profit_pct and current_profit_pct >= take_profit_pct/100:
                    should_sell = True
            
            # Execute buy
            if should_buy:
                shares_to_buy = int((cash * position_size_pct) / current_price)
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cost = shares_to_buy * current_price
                    cash -= cost
                    in_position = True
                    entry_price = current_price
                    trade_start_date = date
                    
                    self.trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': current_price,
                        'shares': shares_to_buy,
                        'amount': cost,
                        'portfolio_value': cash + shares * current_price
                    })
            
            # Execute sell
            elif should_sell:
                sale_amount = shares * current_price
                cash += sale_amount
                
                self.trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': current_price,
                    'shares': shares,
                    'amount': sale_amount,
                    'portfolio_value': cash,
                    'profit': sale_amount - (shares * entry_price),
                    'profit_pct': ((current_price - entry_price) / entry_price) * 100,
                    'holding_period': (date - trade_start_date).days
                })
                
                in_position = False
                shares = 0
            
            # Track portfolio value
            portfolio_value = cash + shares * current_price
            portfolio_values.append(portfolio_value)
        
        self.data['Portfolio_Value'] = portfolio_values
        return self.trades

def main():
    st.set_page_config(page_title="Advanced Investment Backtester", layout="wide")
    st.title("📈 Advanced Investment Backtesting Platform")
    st.markdown("""
    Advanced backtesting with customizable indicators and trading strategies.
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
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=datetime.today() - timedelta(days=365))
    end_date = col2.date_input("End Date", value=datetime.today())
    
    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=100, step=100)
    
    # Strategy selection
    st.sidebar.subheader("Trading Strategy")
    strategy_type = st.sidebar.selectbox(
        "Choose Strategy Type",
        options=[
            "EMA Crossover",
            "RSI Oversold/Oversold",
            "Bollinger Bands",
            "MACD Crossover",
            "Combined"
        ],
        index=0
    )
    
    # Technical indicators settings
    st.sidebar.subheader("Technical Indicators")
    
    # EMA settings
    ema_fast = st.sidebar.slider("Fast EMA Window", 5, 50, 12)
    ema_slow = st.sidebar.slider("Slow EMA Window", 5, 50, 26)
    
    # RSI settings
    rsi_buy_threshold = st.sidebar.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell_threshold = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
    
    # Bollinger Bands settings
    bb_window = st.sidebar.slider("BB Window", 10, 50, 20)
    bb_std = st.sidebar.slider("BB Std Dev", 1, 3, 2)
    
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
                backtester = AdvancedBacktester(symbol, start_date, end_date, initial_capital)
                
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
                        rsi_sell_threshold
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
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Strategy", strategy_type)
        col6.metric("Fast EMA", ema_fast)
        col7.metric("Slow EMA", ema_slow)
        col8.metric("RSI Buy/Sell", f"{rsi_buy_threshold}/{rsi_sell_threshold}")
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f'{symbol} Price & EMAs', 'RSI & Bollinger Bands', 'Portfolio Value'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and EMAs
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Fast'], name=f'EMA{ema_fast}', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_Slow'], name=f'EMA{ema_slow}', line=dict(color='blue')), row=1, col=1)
        
        # Add buy/sell markers
        if trades:
            buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
            buy_prices = [data.loc[t['date']]['Close'] if t['date'] in data.index else None for t in trades if t['type'] == 'BUY']
            sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
            sell_prices = [data.loc[t['date']]['Close'] if t['date'] in data.index else None for t in trades if t['type'] == 'SELL']
            
            if buy_dates:
                fig.add_trace(go.Scatter(
                    x=buy_dates, 
                    y=buy_prices, 
                    mode='markers', 
                    name='Buy Signal', 
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ), row=1, col=1)
            
            if sell_dates:
                fig.add_trace(go.Scatter(
                    x=sell_dates, 
                    y=sell_prices, 
                    mode='markers', 
                    name='Sell Signal', 
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ), row=1, col=1)
        
        # RSI and Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=rsi_sell_threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Sell Level")
        fig.add_hline(y=rsi_buy_threshold, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Buy Level")
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], name='BB Middle', line=dict(color='gray', dash='dash')), row=1, col=1)
        
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
            
            # Detailed trade log
            st.subheader("Trade Log")
            if sell_trades:
                trade_df = pd.DataFrame(sell_trades)
                # Reorder columns for better display
                trade_df = trade_df[['date', 'type', 'price', 'shares', 'amount', 'profit', 'profit_pct', 'holding_period']]
                trade_df = trade_df.rename(columns={
                    'date': 'Date',
                    'type': 'Type',
                    'price': 'Price',
                    'shares': 'Shares',
                    'amount': 'Amount',
                    'profit': 'Profit',
                    'profit_pct': 'Profit %',
                    'holding_period': 'Hold Days'
                })
                st.dataframe(trade_df.style.format({
                    'Price': '${:.2f}',
                    'Amount': '${:.2f}',
                    'Profit': '${:.2f}',
                    'Profit %': '{:.2f}%',
                    'Hold Days': '{:.0f}'
                }))
            else:
                st.info("No completed trades (buy + sell) to display")
        else:
            st.info("Run backtest to see results")
    
    else:
        st.info("Enter parameters and click 'Run Backtest' to start")

if __name__ == "__main__":
    main()