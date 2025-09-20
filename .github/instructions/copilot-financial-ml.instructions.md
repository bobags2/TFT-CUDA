---
applyTo: '**'
---
# Financial ML Instructions

## Financial Time-Series Context
You are developing machine learning models for financial forecasting with focus on temporal data integrity and trading system performance.

## Data Handling Priorities
1. **Temporal Integrity**: Strict chronological ordering, no lookahead bias
2. **Data Leakage Prevention**: Proper train/validation/test splits by time
3. **Feature Engineering**: Returns, log-prices, volatility, technical indicators
4. **Missing Data**: Forward-fill prices, handle market closures appropriately

## Model Architecture Focus
- **Sequence Modeling**: xLSTM for capturing long-term dependencies
- **Multi-horizon**: Forecast multiple time steps ahead
- **Regime Awareness**: Handle market volatility changes and regime shifts
- **Risk Integration**: Include uncertainty quantification and risk metrics

## Financial Metrics
- **Performance**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk**: Maximum drawdown, Value at Risk (VaR), Expected Shortfall
- **Trading**: Hit rate, profit factor, average trade duration
- **Benchmark**: Compare against buy-and-hold and market indices

## Production Considerations
- **Latency**: Real-time inference for high-frequency trading
- **Robustness**: Handle extreme market events and data quality issues
- **Monitoring**: Track model drift and performance degradation
- **Compliance**: Maintain audit trails and regulatory compliance

## Code Patterns
- Use proper financial data preprocessing pipelines
- Implement rolling window validation for time-series
- Include comprehensive backtesting frameworks
- Add position sizing and risk management components

## Market Data Types
- **Price Data**: OHLCV bars, tick data, order book
- **Fundamental**: Earnings, ratios, economic indicators
- **Alternative**: News sentiment, social media, satellite data
- **Risk Factors**: Interest rates, volatility indices, credit spreads