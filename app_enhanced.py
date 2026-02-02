import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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

    def load_data(self):
        """Load historical data"""
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=self.start_date, end=self.end_date, interval="1d")
        return not self.data.empty

    def add_indicators(self, ema_window=20, rsi_window=14):
        """Add technical indicators"""
        self.data['EMA'] = calculate_ema(self.data['Close'], ema_window)
        self.data['RSI'] = calculate_rsi(self.data['Close'], rsi_window)
        self.data['ATR'] = calculate_atr(self.data, 14)

    def run_backtest(self, buy_condition, sell_condition, position_size_pct=0.1, stop_loss_pct=None, take_profit_pct=None):
        """Run backtest with custom conditions"""
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        in_position = False
        entry_price = 0
        trade_start_date = None
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            current_price = row['Close']
            current_rsi = row['RSI']
            current_ema = row['EMA']
            
            # Check if we should buy
            should_buy = eval(buy_condition.format(
                close=current_price,
                ema=current_ema,
                rsi=current_rsi,
                high=row['High'],
                low=row['Low'],
                open=row['Open']
            )) if buy_condition else False
            
            # Check if we should sell
            should_sell = eval(sell_condition.format(
                close=current_price,
                ema=current_ema,
                rsi=current_rsi,
                high=row['High'],
                low=row['Low'],
                open=row['Open']
            )) if sell_condition else False
            
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
            if should_buy and not in_position and cash > 0:
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
            elif should_sell and in_position and shares > 0:
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
    
    # Sidebar for inputs
    st.sidebar.header("Backtest Settings")
    
    # Symbol input
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=datetime.today() - timedelta(days=365))
    end_date = col2.date_input("End Date", value=datetime.today())
    
    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=100, step=100)
    
    # Technical indicators settings
    st.sidebar.subheader("Technical Indicators")
    ema_window = st.sidebar.slider("EMA Window", 5, 50, 20)
    rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
    
    # Position sizing
    st.sidebar.subheader("Position Sizing")
    position_size = st.sidebar.slider("Position Size (%)", 1, 100, 10) / 100
    
    # Risk management
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0, 20, 0)  # 0 means disabled
    take_profit = st.sidebar.slider("Take Profit (%)", 0, 30, 0)  # 0 means disabled
    
    # Trading conditions
    st.sidebar.subheader("Trading Conditions")
    
    # Default buy condition: Price above EMA and RSI below 30 (oversold)
    buy_condition = st.sidebar.text_area(
        "Buy Condition", 
        value="close > ema and rsi < 30",
        help="Use variables: close, ema, rsi, high, low, open. Example: 'close > ema and rsi < 30'"
    )
    
    # Default sell condition: Price below EMA or RSI above 70 (overbought)
    sell_condition = st.sidebar.text_area(
        "Sell Condition", 
        value="close < ema or rsi > 70",
        help="Use variables: close, ema, rsi, high, low, open. Example: 'close < ema or rsi > 70'"
    )
    
    # Initialize session state
    if 'backtester' not in st.session_state:
        st.session_state.backtester = None
    if 'trades' not in st.session_state:
        st.session_state.trades = None

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                backtester = AdvancedBacktester(symbol, start_date, end_date, initial_capital)
                
                if backtester.load_data():
                    backtester.add_indicators(ema_window, rsi_window)
                    
                    # Convert percentage to decimal for stop loss and take profit
                    sl_pct = stop_loss if stop_loss > 0 else None
                    tp_pct = take_profit if take_profit > 0 else None
                    
                    trades = backtester.run_backtest(
                        buy_condition, 
                        sell_condition, 
                        position_size, 
                        sl_pct, 
                        tp_pct
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
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price & EMA', 'RSI', 'Portfolio Value'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and EMA
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], name=f'EMA{ema_window}', line=dict(color='orange')), row=1, col=1)
        
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
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
        
        # Portfolio value
        if 'Portfolio_Value' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio_Value'], name='Portfolio Value', line=dict(color='blue')), row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True)
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