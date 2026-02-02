import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import time

# Technical indicators
def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

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

class StrategyBuilder:
    """Class to build and manage trading strategies"""
    
    def __init__(self):
        self.indicators = {}
        self.conditions = []
        self.weights = {}
    
    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Backtest the strategy on historical data"""
        # Create a copy of the data to work with
        df = data.copy()
        
        # Apply all conditions to generate signals
        df['position'] = 0  # 1 for buy, -1 for sell, 0 for hold
        df['signal'] = 0    # 1 for buy signal, -1 for sell signal
        
        # Example: Basic strategy - Buy when price crosses above SMA20, sell when below
        if 'Close' in df.columns and 'SMA20' in df.columns:
            df['signal'] = np.where(df['Close'] > df['SMA20'], 1, 0)
            df['signal'] = np.where(df['Close'] < df['SMA20'], -1, df['signal'])
        
        # Calculate positions based on signals
        df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['benchmark_returns'] = (1 + df['returns']).cumprod()
        
        return df

class DataLoader:
    """Load and cache financial data"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_historical_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Get historical data for a symbol with caching"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{period}.pkl")
        
        # Check if cached data exists and is recent
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Check if cache is less than 1 day old
                    if isinstance(cached_data, dict) and 'timestamp' in cached_data:
                        cache_age = datetime.now() - cached_data['timestamp']
                        if cache_age.days < 1:
                            return cached_data['data']
            except:
                pass  # If cache is corrupted, proceed to download
        
        # Download fresh data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # Cache the data
        cache_data = {
            'data': data,
            'timestamp': datetime.now()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return data

class ResultsAnalyzer:
    """Analyze backtesting results"""
    
    @staticmethod
    def calculate_metrics(results_df: pd.DataFrame) -> dict:
        """Calculate performance metrics"""
        if 'strategy_returns' not in results_df.columns:
            return {}
        
        strategy_returns = results_df['strategy_returns'].dropna()
        benchmark_returns = results_df['returns'].dropna()
        
        # Calculate cumulative returns
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()
        
        # Total returns
        total_strategy_return = cumulative_strategy.iloc[-1] - 1
        total_benchmark_return = cumulative_benchmark.iloc[-1] - 1
        
        # Annualized returns
        years = len(strategy_returns) / 252  # Trading days assumption
        annualized_strategy_return = (cumulative_strategy.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        annualized_benchmark_return = (cumulative_benchmark.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annualized_strategy_return - 0) / strategy_volatility if strategy_volatility != 0 else 0
        benchmark_sharpe = (annualized_benchmark_return - 0) / benchmark_volatility if benchmark_volatility != 0 else 0
        
        # Maximum Drawdown
        rolling_max = cumulative_strategy.expanding().max()
        drawdown = (cumulative_strategy - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_strategy_return': total_strategy_return,
            'total_benchmark_return': total_benchmark_return,
            'annualized_strategy_return': annualized_strategy_return,
            'annualized_benchmark_return': annualized_benchmark_return,
            'strategy_volatility': strategy_volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len([x for x in results_df['signal'] if x != 0])
        }

# Streamlit app
def main():
    st.set_page_config(page_title="Investment Backtesting Platform", layout="wide")
    st.title("📈 Investment Backtesting Platform")
    st.markdown("""
    A comprehensive web application for backtesting investment strategies using both fundamental and technical analysis.
    """)

    # Sidebar for inputs
    st.sidebar.header("Settings")
    
    # Symbol input
    symbol = st.sidebar.text_input("Symbol", "AAPL")
    
    # Period selection
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    
    # Indicator selection
    st.sidebar.subheader("Indicators")
    use_sma20 = st.sidebar.checkbox("SMA20", value=True)
    use_sma50 = st.sidebar.checkbox("SMA50", value=True)
    use_rsi = st.sidebar.checkbox("RSI", value=True)
    use_macd = st.sidebar.checkbox("MACD", value=True)
    use_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Load data button
    if st.sidebar.button("Load Data"):
        with st.spinner(f"Loading data for {symbol.upper()}..."):
            loader = DataLoader()
            try:
                st.session_state.data = loader.get_historical_data(symbol.upper(), period)
                st.success(f"Data loaded for {symbol.upper()} ({len(st.session_state.data)} records)")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Stock Data: {symbol.upper()}")
        if st.session_state.data is not None:
            st.dataframe(st.session_state.data.tail())
        else:
            st.info("Load data to begin")
    
    with col2:
        st.subheader("Current Settings")
        st.write(f"**Symbol:** {symbol}")
        st.write(f"**Period:** {period}")
        st.write("**Active Indicators:**")
        indicators = []
        if use_sma20: indicators.append("SMA20")
        if use_sma50: indicators.append("SMA50")
        if use_rsi: indicators.append("RSI")
        if use_macd: indicators.append("MACD")
        if use_bb: indicators.append("Bollinger Bands")
        st.write(", ".join(indicators) if indicators else "None")

    # Process data and calculate indicators
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        # Calculate selected indicators
        if use_sma20:
            data['SMA20'] = calculate_sma(data['Close'], 20)
        if use_sma50:
            data['SMA50'] = calculate_sma(data['Close'], 50)
        if use_rsi:
            data['RSI'] = calculate_rsi(data['Close'])
        if use_macd:
            macd, signal, hist = calculate_macd(data['Close'])
            data['MACD'] = macd
            data['MACD_Signal'] = signal
        if use_bb:
            upper, middle, lower = calculate_bollinger_bands(data['Close'])
            data['BB_Upper'] = upper
            data['BB_Middle'] = middle
            data['BB_Lower'] = lower
        
        # Run backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                strategy = StrategyBuilder()
                st.session_state.results = strategy.backtest(data)
                
                if st.session_state.results is not None:
                    st.success("Backtest completed successfully!")
                else:
                    st.error("Error running backtest")
        
        # Display chart
        if not data.empty:
            st.subheader(f"Price Chart: {symbol.upper()}")
            
            # Create subplot with price and RSI
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1,
                               row_heights=[0.7, 0.3],
                               subplot_titles=(f'{symbol.upper()} Price', 'RSI'))
            
            # Add price and indicators
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                    name='Price', line=dict(color='black')), 
                         row=1, col=1)
            
            if 'SMA20' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], 
                                        name='SMA20', line=dict(color='orange')), 
                             row=1, col=1)
            if 'SMA50' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], 
                                        name='SMA50', line=dict(color='blue')), 
                             row=1, col=1)
            if 'BB_Upper' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                                        name='BB Upper', line=dict(color='gray', dash='dash')), 
                             row=1, col=1)
            if 'BB_Lower' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                                        name='BB Lower', line=dict(color='gray', dash='dash')), 
                             row=1, col=1)
            
            # Add RSI if available
            if 'RSI' in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                        name='RSI', line=dict(color='purple')), 
                             row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display results if backtest was run
        if st.session_state.results is not None:
            st.subheader("Backtest Results")
            
            # Calculate metrics
            analyzer = ResultsAnalyzer()
            metrics = analyzer.calculate_metrics(st.session_state.results)
            
            # Display metrics in columns
            if metrics:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Strategy Return", f"{metrics['total_strategy_return']:.2%}")
                col2.metric("Total Benchmark Return", f"{metrics['total_benchmark_return']:.2%}")
                col3.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Annualized Strategy Return", f"{metrics['annualized_strategy_return']:.2%}")
                col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            
            # Plot equity curve
            if 'cumulative_returns' in st.session_state.results.columns:
                st.subheader("Strategy vs Benchmark Performance")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=st.session_state.results.index, 
                    y=st.session_state.results['cumulative_returns'], 
                    name='Strategy', 
                    line=dict(width=2)
                ))
                if 'benchmark_returns' in st.session_state.results.columns:
                    fig2.add_trace(go.Scatter(
                        x=st.session_state.results.index, 
                        y=st.session_state.results['benchmark_returns'], 
                        name='Benchmark', 
                        line=dict(width=2)
                    ))
                
                fig2.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Investment Backtesting Platform - Designed for comprehensive strategy analysis*")

if __name__ == "__main__":
    main()