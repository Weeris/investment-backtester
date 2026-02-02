# Investment Backtesting Platform - Streamlit Version

A comprehensive web application for backtesting investment strategies using both fundamental and technical analysis.

## Features

- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands, and more
- **Strategy Builder**: Visual interface to create custom trading strategies
- **Backtesting Engine**: Historical performance simulation
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility analysis
- **Interactive Charts**: Powered by Plotly for better visualization
- **Web-Based**: Accessible from any device with a web browser
- **Easy Sharing**: Deployable on cloud platforms for sharing with others

## Deployment Options

### Option 1: Streamlit Community Cloud (Free)
1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Enter your repository URL
4. Click "Duplicate" to deploy

### Option 2: Heroku (Free Tier Available)
1. Create a Heroku account
2. Install Heroku CLI
3. Push your code to the repository
4. Connect to Heroku for deployment

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app locally:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock symbol (e.g., AAPL, TSLA, NVDA)
2. Select the time period for historical data
3. Choose technical indicators to analyze
4. Click "Load Data" to retrieve historical prices
5. Click "Run Backtest" to test your strategy
6. View performance metrics and equity curves

## Architecture

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- Various technical analysis functions integrated

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

MIT License