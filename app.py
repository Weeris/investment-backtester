import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Simple technical indicators
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

def simple_backtest(data, buy_condition_str, sell_condition_str):
    """Simple backtest function"""
    df = data.copy()
    
    # Calculate indicators
    df['EMA'] = calculate_ema(df['Close'], 20)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Initialize trading columns
    df['Signal'] = 0
    df['Position'] = 0
    df['Returns'] = df['Close'].pct_change()
    
    # Generate signals based on conditions
    for i in range(len(df)):
        current_row = df.iloc[i]
        close = current_row['Close']
        ema = current_row['EMA']
        rsi = current_row['RSI']
        
        # Evaluate buy condition
        try:
            buy_condition = eval(buy_condition_str.format(close=close, ema=ema, rsi=rsi))
        except:
            buy_condition = False
        
        # Evaluate sell condition
        try:
            sell_condition = eval(sell_condition_str.format(close=close, ema=ema, rsi=rsi))
        except:
            sell_condition = False
        
        if buy_condition:
            df.at[df.index[i], 'Signal'] = 1
        elif sell_condition:
            df.at[df.index[i], 'Signal'] = -1
    
    # Fill positions
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    # Calculate strategy returns
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Benchmark'] = (1 + df['Returns']).cumprod()
    
    return df

def main():
    st.set_page_config(page_title="Simple Investment Backtester", layout="wide")
    st.title("📈 Simple Investment Backtesting Platform")
    
    # Sidebar for inputs
    st.sidebar.header("Backtest Settings")
    
    # Symbol input
    symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", value=datetime.today() - timedelta(days=365))
    end_date = col2.date_input("End Date", value=datetime.today())
    
    # Trading conditions
    st.sidebar.subheader("Trading Conditions")
    
    buy_condition = st.sidebar.text_area(
        "Buy Condition", 
        value="close > ema and rsi < 30",
        help="Available variables: close, ema, rsi"
    )
    
    sell_condition = st.sidebar.text_area(
        "Sell Condition", 
        value="close < ema or rsi > 70",
        help="Available variables: close, ema, rsi"
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Loading data and running backtest..."):
            try:
                # Load data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval="1d")
                
                if data.empty:
                    st.error("No data found for the given symbol and date range")
                    return
                
                # Run backtest
                results = simple_backtest(data, buy_condition, sell_condition)
                
                st.session_state.data = data
                st.session_state.results = results
                st.success("Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        
        total_return_strategy = (results['Cumulative_Strategy'].iloc[-1] - 1) * 100 if 'Cumulative_Strategy' in results.columns else 0
        total_return_benchmark = (results['Cumulative_Benchmark'].iloc[-1] - 1) * 100 if 'Cumulative_Benchmark' in results.columns else 0
        win_rate = (results['Strategy_Returns'][results['Strategy_Returns'] > 0].count() / results['Strategy_Returns'].count()) * 100 if 'Strategy_Returns' in results.columns else 0
        
        col1.metric("Strategy Return", f"{total_return_strategy:.2f}%")
        col2.metric("Benchmark Return", f"{total_return_benchmark:.2f}%")
        col3.metric("Win Rate", f"{win_rate:.2f}%")
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price & EMA', 'RSI', 'Portfolio Value'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and EMA
        fig.add_trace(go.Scatter(x=results.index, y=results['Close'], name='Close', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['EMA'], name='EMA20', line=dict(color='orange')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=results.index, y=results['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
        
        # Portfolio value
        fig.add_trace(go.Scatter(x=results.index, y=results['Cumulative_Strategy'], name='Strategy', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['Cumulative_Benchmark'], name='Benchmark', line=dict(color='red')), row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Enter parameters and click 'Run Backtest' to start")

if __name__ == "__main__":
    main()