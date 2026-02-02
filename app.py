import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

def main():
    st.set_page_config(page_title="Minimal Backtester", layout="wide")
    st.title("📈 Minimal Investment Backtester")
    
    # Simple inputs
    symbol = st.text_input("Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=30))
    end_date = st.date_input("End Date", value=datetime.today())
    
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            try:
                # Load data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    st.error("No data found")
                    return
                
                # Calculate simple moving average
                data['SMA20'] = data['Close'].rolling(window=20).mean()
                
                # Display data
                st.subheader(f"Data for {symbol}")
                st.line_chart(data[['Close', 'SMA20']])
                
                # Show raw data
                st.subheader("Raw Data")
                st.dataframe(data.tail())
                
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()