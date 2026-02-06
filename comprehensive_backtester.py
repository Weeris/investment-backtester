"""
Comprehensive Multi-Asset Backtesting Framework
Supports: Cryptocurrencies, Stocks, Forex, Indices
Indicators: 20+ Technical Indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ============== Asset Categories ==============
class AssetType(Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    INDEX = "index"
    COMMODITY = "commodity"
    ETF = "etf"


# ============== Predefined Assets ==============
ASSET_UNIVERSE = {
    # Cryptocurrencies
    "BTC-USD": (AssetType.CRYPTO, "Bitcoin"),
    "ETH-USD": (AssetType.CRYPTO, "Ethereum"),
    "SOL-USD": (AssetType.CRYPTO, "Solana"),
    "BNB-USD": (AssetType.CRYPTO, "Binance Coin"),
    "XRP-USD": (AssetType.CRYPTO, "Ripple"),
    "ADA-USD": (AssetType.CRYPTO, "Cardano"),
    "DOGE-USD": (AssetType.CRYPTO, "Dogecoin"),
    "MATIC-USD": (AssetType.CRYPTO, "Polygon"),
    "LTC-USD": (AssetType.CRYPTO, "Litecoin"),
    "DOT-USD": (AssetType.CRYPTO, "Polkadot"),
    
    # US Stocks
    "AAPL": (AssetType.STOCK, "Apple Inc."),
    "MSFT": (AssetType.STOCK, "Microsoft Corp."),
    "GOOGL": (AssetType.STOCK, "Alphabet Inc."),
    "AMZN": (AssetType.STOCK, "Amazon.com Inc."),
    "NVDA": (AssetType.STOCK, "NVIDIA Corp."),
    "META": (AssetType.STOCK, "Meta Platforms"),
    "TSLA": (AssetType.STOCK, "Tesla Inc."),
    "JPM": (AssetType.STOCK, "JPMorgan Chase"),
    "V": (AssetType.STOCK, "Visa Inc."),
    "JNJ": (AssetType.STOCK, "Johnson & Johnson"),
    
    # Thai Stocks (SET)
    "PTT.BK": (AssetType.STOCK, "PTT Public Company"),
    "SCC.BK": (AssetType.STOCK, "Siam Cement"),
    "CPALL.BK": (AssetType.STOCK, "CP All"),
    "KBANK.BK": (AssetType.STOCK, "Kasikorn Bank"),
    "TRUE.BK": (AssetType.STOCK, "True Corporation"),
    
    # Indices
    "^GSPC": (AssetType.INDEX, "S&P 500"),
    "^DJI": (AssetType.INDEX, "Dow Jones"),
    "^IXIC": (AssetType.INDEX, "NASDAQ"),
    "^HSI": (AssetType.INDEX, "Hang Seng"),
    "^SET.BK": (AssetType.INDEX, "SET Index"),
    
    # ETFs
    "SPY": (AssetType.ETF, "SPDR S&P 500 ETF"),
    "QQQ": (AssetType.ETF, "Invesco QQQ"),
    "GLD": (AssetType.ETF, "SPDR Gold Shares"),
    "ARKK": (AssetType.ETF, "ARK Innovation ETF"),
}


# ============== Indicator Categories ==============
class IndicatorCategory(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class Indicator:
    name: str
    category: IndicatorCategory
    periods: List[int]
    description: str


# ============== Supported Indicators ==============
INDICATORS = {
    # Trend Indicators
    "SMA": Indicator("SMA", IndicatorCategory.TREND, [5, 10, 20, 50, 100, 200], "Simple Moving Average"),
    "EMA": Indicator("EMA", IndicatorCategory.TREND, [5, 10, 12, 20, 26, 50, 100, 200], "Exponential Moving Average"),
    "WMA": Indicator("WMA", IndicatorCategory.TREND, [5, 10, 20, 50], "Weighted Moving Average"),
    "DEMA": Indicator("DEMA", IndicatorCategory.TREND, [10, 20, 50], "Double Exponential MA"),
    "TEMA": Indicator("TEMA", IndicatorCategory.TREND, [10, 20, 50], "Triple Exponential MA"),
    
    # Momentum Indicators
    "RSI": Indicator("RSI", IndicatorCategory.MOMENTUM, [7, 14, 21], "Relative Strength Index"),
    "STOCH_K": Indicator("STOCH_K", IndicatorCategory.MOMENTUM, [5, 14, 21], "Stochastic %K"),
    "STOCH_D": Indicator("STOCH_D", IndicatorCategory.MOMENTUM, [3, 14], "Stochastic %D"),
    "MACD": Indicator("MACD", IndicatorCategory.MOMENTUM, [12, 26, 9], "MACD (12, 26, 9)"),
    "CCI": Indicator("CCI", IndicatorCategory.MOMENTUM, [14, 20, 50], "Commodity Channel Index"),
    "WILLIAMS_R": Indicator("WILLIAMS_R", IndicatorCategory.MOMENTUM, [5, 14, 21], "Williams %R"),
    "MOM": Indicator("MOM", IndicatorCategory.MOMENTUM, [10, 14, 21], "Momentum"),
    "ROC": Indicator("ROC", IndicatorCategory.MOMENTUM, [10, 14, 21], "Rate of Change"),
    "ADX": Indicator("ADX", IndicatorCategory.MOMENTUM, [14, 20], "Average Directional Index"),
    "AROON": Indicator("AROON", IndicatorCategory.MOMENTUM, [14, 25], "Aroon Oscillator"),
    
    # Volatility Indicators
    "BB": Indicator("BB", IndicatorCategory.VOLATILITY, [10, 20, 50], "Bollinger Bands"),
    "ATR": Indicator("ATR", IndicatorCategory.VOLATILITY, [7, 14, 21], "Average True Range"),
    "KC": Indicator("KC", IndicatorCategory.VOLATILITY, [10, 20, 50], "Keltner Channels"),
    
    # Volume Indicators
    "OBV": Indicator("OBV", IndicatorCategory.VOLUME, [], "On Balance Volume"),
    "VWAP": Indicator("VWAP", IndicatorCategory.VOLUME, [], "Volume Weighted Average Price"),
    "AD": Indicator("AD", IndicatorCategory.VOLUME, [], "Accumulation/Distribution"),
    "CMF": Indicator("CMF", IndicatorCategory.VOLUME, [20, 30], "Chaikin Money Flow"),
}


# ============== Trading Strategies ==============
class StrategyType(Enum):
    # Trend Following
    SMA_CROSS = "sma_crossover"
    EMA_CROSS = "ema_crossover"
    MACD_CROSS = "macd_cross"
    
    # Mean Reversion
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    BB_REVERSION = "bb_reversion"
    STOCH_CROSS = "stochastic_cross"
    
    # Momentum
    RSI_MOMENTUM = "rsi_momentum"
    MACD_MOMENTUM = "macd_momentum"
    ADX_MOMENTUM = "adx_momentum"
    
    # Volume
    OBV_TREND = "obv_trend"
    VOLUME_SPIKE = "volume_spike"
    
    # Combined
    MULTI_INDICATOR = "multi_indicator"
    TREND_CONFIRM = "trend_confirmation"
    
    # DCA
    SIMPLE_DCA = "simple_dca"


# ============== Backtest Engine ==============
class ComprehensiveBacktester:
    """Comprehensive backtesting engine for multiple assets and strategies"""
    
    def __init__(
        self,
        symbol: str = "BTC-USD",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        daily_investment: float = 0.0
    ):
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.daily_investment = daily_investment
        
        self.data = None
        self.indicators = {}
        self.trades = []
        self.results = {}
        
        self._fetch_data()
        self._calculate_all_indicators()
    
    def _fetch_data(self) -> None:
        """Fetch historical data for the symbol"""
        print(f"📊 Fetching {self.symbol} data...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Flatten multi-index if exists
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        # Standardize column names
        self.data.columns = [col.capitalize() for col in self.data.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in self.data.columns:
                self.data[col] = np.nan
        
        print(f"✅ Loaded {len(self.data)} days of data")
    
    def _calculate_all_indicators(self) -> None:
        """Calculate all supported technical indicators"""
        df = self.data.copy()
        
        # SMA
        for period in INDICATORS["SMA"].periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # EMA
        for period in INDICATORS["EMA"].periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # WMA
        for period in INDICATORS["WMA"].periods:
            weights = np.arange(1, period + 1)
            df[f'WMA_{period}'] = df['Close'].rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        # DEMA
        for period in INDICATORS["DEMA"].periods:
            ema1 = df['Close'].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            df[f'DEMA_{period}'] = 2 * ema1 - ema2
        
        # TEMA
        for period in INDICATORS["TEMA"].periods:
            ema1 = df['Close'].ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            df[f'TEMA_{period}'] = 3 * ema1 - 3 * ema2 + ema3
        
        # RSI
        for period in INDICATORS["RSI"].periods:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        for period in INDICATORS["STOCH_K"].periods:
            low_min = df['Low'].rolling(window=period).min()
            high_max = df['High'].rolling(window=period).max()
            df[f'STOCH_K_{period}'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # Stochastic %D uses %K with period 3
        for k_period in [5, 14, 21]:  # Calculate %D for each %K period
            if f'STOCH_K_{k_period}' in df.columns:
                df[f'STOCH_D_{k_period}'] = df[f'STOCH_K_{k_period}'].rolling(window=3).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
        
        # CCI
        for period in INDICATORS["CCI"].periods:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            df[f'CCI_{period}'] = (tp - sma_tp) / (0.015 * mad)
        
        # Williams %R
        for period in INDICATORS["WILLIAMS_R"].periods:
            highest = df['High'].rolling(window=period).max()
            lowest = df['Low'].rolling(window=period).min()
            df[f'WILLIAMS_R_{period}'] = -100 * (highest - df['Close']) / (highest - lowest)
        
        # Momentum
        for period in INDICATORS["MOM"].periods:
            df[f'MOM_{period}'] = df['Close'].diff(period)
        
        # ROC
        for period in INDICATORS["ROC"].periods:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # ADX
        for period in INDICATORS["ADX"].periods:
            tr = pd.Series(np.maximum(df['High'] - df['Low'],
                           np.maximum(abs(df['High'] - df['Close'].shift()),
                                     abs(df['Low'] - df['Close'].shift()))), index=df.index)
            atr = tr.rolling(window=period).mean()
            
            plus_dm = pd.Series(np.where(df['High'] - df['High'].shift() > df['Low'].shift() - df['Low'],
                              df['High'] - df['High'].shift(), 0), index=df.index)
            minus_dm = pd.Series(np.where(df['Low'].shift() - df['Low'] > df['High'] - df['High'].shift(),
                               df['Low'].shift() - df['Low'], 0), index=df.index)
            
            plus_di = (100 * plus_dm.rolling(window=period).mean() / atr).fillna(0)
            minus_di = (100 * minus_dm.rolling(window=period).mean() / atr).fillna(0)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'ADX_{period}'] = dx.rolling(window=period).mean()
            df[f'PLUS_DI_{period}'] = plus_di.rolling(window=period).mean()
            df[f'MINUS_DI_{period}'] = minus_di.rolling(window=period).mean()
        
        # Aroon
        for period in INDICATORS["AROON"].periods:
            df[f'AROON_Up_{period}'] = 100 * (df['High'].rolling(window=period + 1).apply(lambda x: x.argmax(), raw=True) / period)
            df[f'AROON_Down_{period}'] = 100 * (df['Low'].rolling(window=period + 1).apply(lambda x: x.argmin(), raw=True) / period)
            df[f'AROON_Osc_{period}'] = df[f'AROON_Up_{period}'] - df[f'AROON_Down_{period}']
        
        # Bollinger Bands
        for period in INDICATORS["BB"].periods:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            df[f'BB_MA_{period}'] = sma
            df[f'BB_Upper_{period}'] = sma + (2 * std)
            df[f'BB_Lower_{period}'] = sma - (2 * std)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # ATR
        for period in INDICATORS["ATR"].periods:
            tr = np.maximum(df['High'] - df['Low'],
                           np.maximum(abs(df['High'] - df['Close'].shift()),
                                     abs(df['Low'] - df['Close'].shift())))
            df[f'ATR_{period}'] = tr.rolling(window=period).mean()
            df[f'ATR_Pct_{period}'] = (df[f'ATR_{period}'] / df['Close']) * 100
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_SMA_10'] = df['OBV'].rolling(window=10).mean()
        
        # VWAP
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        self.data = df.fillna(method='bfill').fillna(method='ffill')
        self.indicators = df.columns.tolist()
    
    def generate_signal(self, strategy_type: StrategyType, params: Dict = None) -> pd.Series:
        """Generate trading signals based on strategy"""
        if params is None:
            params = {}
        
        df = self.data.copy()
        signals = pd.Series(False, index=df.index)
        
        if strategy_type == StrategyType.SMA_CROSS:
            short = params.get('short_period', 20)
            long = params.get('long_period', 50)
            
            buy_cond = (df[f'SMA_{short}'] > df[f'SMA_{long}']) & \
                      (df[f'SMA_{short}'].shift(1) <= df[f'SMA_{long}'].shift(1))
            
            sell_cond = (df[f'SMA_{short}'] < df[f'SMA_{long}']) & \
                        (df[f'SMA_{short}'].shift(1) >= df[f'SMA_{long}'].shift(1))
            
            signals = buy_cond | sell_cond
            
        elif strategy_type == StrategyType.EMA_CROSS:
            short = params.get('short_period', 12)
            long = params.get('long_period', 26)
            
            buy_cond = (df[f'EMA_{short}'] > df[f'EMA_{long}']) & \
                      (df[f'EMA_{short}'].shift(1) <= df[f'EMA_{long}'].shift(1))
            
            sell_cond = (df[f'EMA_{short}'] < df[f'EMA_{long}']) & \
                        (df[f'EMA_{short}'].shift(1) >= df[f'EMA_{long}'].shift(1))
            
            signals = buy_cond | sell_cond
            
        elif strategy_type == StrategyType.MACD_CROSS:
            buy_cond = (df['MACD_Line'] > df['MACD_Signal']) & \
                      (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))
            
            sell_cond = (df['MACD_Line'] < df['MACD_Signal']) & \
                       (df['MACD_Line'].shift(1) >= df['MACD_Signal'].shift(1))
            
            signals = buy_cond | sell_cond
            
        elif strategy_type == StrategyType.RSI_OVERSOLD:
            period = params.get('period', 14)
            threshold = params.get('threshold', 30)
            
            signals = df[f'RSI_{period}'] < threshold
            
        elif strategy_type == StrategyType.BB_REVERSION:
            period = params.get('period', 20)
            threshold = params.get('threshold', 0.1)
            
            signals = df[f'BB_Position_{period}'] < threshold
            
        elif strategy_type == StrategyType.STOCH_CROSS:
            k_period = params.get('k_period', 14)
            
            # Use available periods
            actual_k = k_period if f'STOCH_K_{k_period}' in df.columns else 14
            actual_d = 3  # %D always uses period 3
            
            buy_cond = (df[f'STOCH_K_{actual_k}'] > df[f'STOCH_D_{actual_k}']) & \
                       (df[f'STOCH_K_{actual_k}'].shift(1) <= df[f'STOCH_D_{actual_k}'].shift(1))
            
            sell_cond = (df[f'STOCH_K_{actual_k}'] < df[f'STOCH_D_{actual_k}']) & \
                       (df[f'STOCH_K_{actual_k}'].shift(1) >= df[f'STOCH_D_{actual_k}'].shift(1))
            
            signals = buy_cond | sell_cond
            
        elif strategy_type == StrategyType.ADX_MOMENTUM:
            period = params.get('period', 14)
            threshold = params.get('threshold', 25)
            
            signals = df[f'ADX_{period}'] > threshold
            
        elif strategy_type == StrategyType.TREND_CONFIRM:
            signals = (df['Close'] > df['SMA_50']) & \
                     (df['RSI_14'] < 70) & \
                     (df['ADX_14'] > 20)
            
        elif strategy_type == StrategyType.MULTI_INDICATOR:
            signals = (df['RSI_14'] < 35) & \
                     (df['BB_Position_20'] < 0.2) & \
                     (df['ROC_14'] > -5)
        
        return signals.fillna(False)
    
    def run_backtest(
        self,
        strategy_type: StrategyType,
        params: Dict = None,
        position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """Run backtest with specified strategy"""
        if params is None:
            params = {}
        
        df = self.data.copy()
        signals = self.generate_signal(strategy_type, params)
        
        cash = self.initial_capital
        shares = 0.0
        position = False
        entry_price = 0.0
        trades = []
        portfolio_values = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            price = row['Close']
            signal = signals.iloc[i]
            
            if signal and not position:
                investment = cash * position_size
                if investment > 0:
                    shares_to_buy = investment / price
                    shares += shares_to_buy
                    cash -= investment
                    position = True
                    entry_price = price
                    
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': investment,
                        'signal': strategy_type.value
                    })
            
            elif position:
                pnl_pct = (price - entry_price) / entry_price * 100
                should_sell = False
                
                if stop_loss and pnl_pct <= -stop_loss:
                    should_sell = True
                elif take_profit and pnl_pct >= take_profit:
                    should_sell = True
                
                if should_sell:
                    sell_amount = shares * price
                    profit = sell_amount - (shares * entry_price)
                    
                    trades.append({
                        'date': date,
                        'type': 'SELL',
                        'price': price,
                        'shares': shares,
                        'amount': sell_amount,
                        'profit': profit,
                        'profit_pct': pnl_pct
                    })
                    
                    cash += sell_amount
                    shares = 0
                    position = False
            
            portfolio_value = cash + (shares * price)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': cash,
                'shares': shares
            })
        
        final_price = df['Close'].iloc[-1]
        final_value = cash + (shares * final_price)
        
        # Buy & Hold comparison
        shares_bh = self.initial_capital / df['Close'].iloc[0]
        bh_final = shares_bh * df['Close'].iloc[-1]
        bh_return = ((bh_final - self.initial_capital) / self.initial_capital) * 100
        
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Max drawdown
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['peak'] = portfolio_df['value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Trades count
        buys = len([t for t in trades if t['type'] == 'BUY'])
        sells = len([t for t in trades if t['type'] == 'SELL'])
        
        results = {
            'strategy': strategy_type.value,
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': bh_return,
            'outperformance': total_return - bh_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': buys + sells,
            'buy_trades': buys,
            'sell_trades': sells,
            'trades': trades,
            'portfolio_history': portfolio_df,
            'sharpe_ratio': self._calculate_sharpe(portfolio_df['value']),
        }
        
        self.results = results
        return results
    
    def _calculate_sharpe(self, values: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = values.pct_change().dropna()
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def run_multi_strategy_test(self) -> List[Dict]:
        """Run multiple strategies and compare results"""
        print(f"\n🚀 Running comprehensive backtest for {self.symbol}")
        print("=" * 60)
        
        strategies = [
            {'type': StrategyType.SMA_CROSS, 'params': {'short_period': 5, 'long_period': 20}, 'name': 'SMA 5/20'},
            {'type': StrategyType.SMA_CROSS, 'params': {'short_period': 20, 'long_period': 50}, 'name': 'SMA 20/50'},
            {'type': StrategyType.SMA_CROSS, 'params': {'short_period': 50, 'long_period': 200}, 'name': 'SMA 50/200'},
            {'type': StrategyType.EMA_CROSS, 'params': {'short_period': 12, 'long_period': 26}, 'name': 'EMA 12/26'},
            {'type': StrategyType.MACD_CROSS, 'params': {}, 'name': 'MACD Cross'},
            {'type': StrategyType.RSI_OVERSOLD, 'params': {'period': 14, 'threshold': 30}, 'name': 'RSI < 30'},
            {'type': StrategyType.RSI_OVERSOLD, 'params': {'period': 7, 'threshold': 25}, 'name': 'RSI < 25 (Fast)'},
            {'type': StrategyType.BB_REVERSION, 'params': {'period': 20, 'threshold': 0.1}, 'name': 'BB Lower Band'},
            {'type': StrategyType.STOCH_CROSS, 'params': {'k_period': 14, 'd_period': 3}, 'name': 'Stochastic Cross'},
            {'type': StrategyType.ADX_MOMENTUM, 'params': {'period': 14, 'threshold': 25}, 'name': 'ADX > 25'},
            {'type': StrategyType.TREND_CONFIRM, 'params': {}, 'name': 'Trend Confirm'},
            {'type': StrategyType.MULTI_INDICATOR, 'params': {}, 'name': 'Multi Indicator'},
        ]
        
        results = []
        
        for s in strategies:
            print(f"  Testing: {s['name']}...", end=" ")
            try:
                result = self.run_backtest(s['type'], s['params'])
                result['strategy_name'] = s['name']
                results.append(result)
                print(f"✅ Return: {result['total_return_pct']:.2f}%")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        results.sort(key=lambda x: x['total_return_pct'], reverse=True)
        return results
    
    def print_results_summary(self, results: List[Dict]) -> None:
        """Print formatted results summary"""
        print("\n" + "=" * 100)
        print(f"📊 BACKTEST RESULTS: {self.symbol} ({self.start_date} to {self.end_date})")
        print("=" * 100)
        print(f"{'Rank':<4} {'Strategy':<20} {'Return %':<12} {'vs Buy&Hold':<12} {'Max DD %':<10} {'Trades':<8} {'Sharpe':<8}")
        print("-" * 100)
        
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r['strategy_name']:<20} {r['total_return_pct']:>8.2f}% {r['outperformance']:>+8.2f}% "
                  f"{r['max_drawdown_pct']:>8.2f}% {r['total_trades']:<8} {r['sharpe_ratio']:<8.2f}")
        
        print("=" * 100)


# ============== Main Entry Point ==============
if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    backtester = ComprehensiveBacktester(
        symbol=symbol,
        start_date=(datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d'),
        initial_capital=10000
    )
    
    results = backtester.run_multi_strategy_test()
    backtester.print_results_summary(results)
