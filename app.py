import streamlit as st
import sys
import traceback

def main():
    try:
        st.set_page_config(page_title="Debug Backtester", layout="wide")
        st.title("🔍 Debug Investment Backtester")
        st.write(f"Python version: {sys.version}")
        st.write("Dependencies available:")
        
        # Check if key libraries can be imported
        try:
            import pandas as pd
            st.write(f"Pandas version: {pd.__version__}")
        except ImportError as e:
            st.write(f"❌ Pandas import error: {e}")
        
        try:
            import numpy as np
            st.write(f"NumPy version: {np.__version__}")
        except ImportError as e:
            st.write(f"❌ NumPy import error: {e}")
        
        try:
            import yfinance
            st.write(f"yfinance version: {yfinance.__version__}")
        except ImportError as e:
            st.write(f"❌ yfinance import error: {e}")
        
        try:
            import plotly
            st.write(f"Plotly version: {plotly.__version__}")
        except ImportError as e:
            st.write(f"❌ Plotly import error: {e}")
        
        st.write("✅ All imports successful!")
        
        # Simple functionality
        symbol = st.text_input("Stock Symbol", "AAPL")
        if st.button("Test Data Load"):
            try:
                import yfinance as yf
                data = yf.download(symbol, period="1mo")
                st.write(f"Successfully loaded {len(data)} rows of data for {symbol}")
                st.dataframe(data.head())
            except Exception as e:
                st.write(f"Error loading data: {e}")
                st.write(traceback.format_exc())
    
    except Exception as e:
        st.write(f"❌ Unexpected error in main: {e}")
        st.write(traceback.format_exc())

if __name__ == "__main__":
    main()