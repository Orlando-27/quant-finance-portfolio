# Project 14 - Automated Trading Bot with Interactive Brokers API

End-to-end automated trading system connecting to Interactive Brokers
via `ib_insync`, featuring multi-indicator signal generation, position
sizing, bracket order execution, real-time position monitoring, and
email notifications.

---

## Architecture

```
main.py
  └── TradingBot (trading_bot.py)
        ├── IBConnector      -- IB Gateway / TWS connection & data
        ├── SignalGenerator  -- RSI + MACD + Bollinger + EMA consensus
        ├── RiskManager      -- Position sizing, stop-loss, drawdown guard
        ├── OrderManager     -- Limit / bracket order submission & tracking
        ├── PositionMonitor  -- Real-time P&L and equity curve
        ├── Notifier         -- SMTP email alerts and daily summary
        └── Plotter          -- Publication-quality charts (headless Agg)
```

---

## Signal Generation

Four indicators produce independent votes in {-1, 0, +1}:

| Indicator        | Long (+1)               | Short (-1)              |
|------------------|-------------------------|-------------------------|
| RSI(14)          | RSI <= 30               | RSI >= 70               |
| MACD(12,26,9)    | MACD crosses above sig  | MACD crosses below sig  |
| Bollinger(20,2)  | Price touches lower band | Price touches upper band |
| EMA(9 / 21)      | Short crosses above long | Short crosses below long |

A trade is placed when **3 or more indicators agree** (configurable via
`SignalConfig.consensus_long / consensus_short`).

---

## Risk Controls

| Control              | Default   | Config key             |
|----------------------|-----------|------------------------|
| Max position size    | 5 % NAV   | `max_position_pct`     |
| Max deployed capital | 80 % NAV  | `max_portfolio_pct`    |
| Hard stop-loss       | 2 %       | `stop_loss_pct`        |
| Take-profit target   | 4 %       | `take_profit_pct`      |
| Daily loss limit     | 3 %       | `max_daily_loss`       |
| Max drawdown         | 10 %      | `max_drawdown`         |

---

## Quick Start

### 1. Install dependencies

```bash
cd ~/quant-finance-portfolio/14-automated-trading-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run offline demo (no IB required)

```bash
python main.py --demo
```

### 3. Connect to IB Paper Trading

Ensure IB Gateway is running on port 7497 with API enabled, then:

```bash
export IB_HOST=127.0.0.1
export IB_PORT=7497
export IB_ACCOUNT=DU123456
python main.py
```

### 4. Dry run (connect to IB, no orders placed)

```bash
python main.py --dry-run
```

### 5. Enable email notifications

```bash
export NOTIFY_EMAIL=true
export SMTP_SENDER=you@gmail.com
export SMTP_PASS=app_password
export NOTIFY_TO=recipient@example.com
python main.py --demo
```

---

## Environment Variables

| Variable         | Default       | Description                        |
|------------------|---------------|------------------------------------|
| `IB_HOST`        | `127.0.0.1`   | IB Gateway host                    |
| `IB_PORT`        | `7497`        | 7497=paper, 7496=live              |
| `IB_CID`         | `1`           | Client ID                          |
| `IB_ACCOUNT`     |               | Account ID (e.g. DU123456)         |
| `PAPER_TRADING`  | `true`        | Enable paper trading mode          |
| `DRY_RUN`        | `false`       | Log orders only, no submissions    |
| `DEMO_MODE`      | `false`       | Offline demo, synthetic data       |
| `SCAN_SECS`      | `300`         | Seconds between scan cycles        |
| `NOTIFY_EMAIL`   | `false`       | Enable email notifications         |
| `SMTP_HOST`      | smtp.gmail.com | SMTP server                       |
| `SMTP_PORT`      | `587`         | SMTP port (TLS)                    |
| `SMTP_SENDER`    |               | Sender email address               |
| `SMTP_PASS`      |               | SMTP password / app password       |
| `NOTIFY_TO`      |               | Recipient email address            |
| `LOG_LEVEL`      | `INFO`        | Logging verbosity                  |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Expected output: **32 tests passing**.

---

## Output Charts

All charts are saved under `outputs/charts/` with UTC timestamps:

| Chart                    | Description                                     |
|--------------------------|-------------------------------------------------|
| `signal_{SYM}.png`       | Price, Bollinger, EMA, Volume, RSI, MACD panels |
| `signal_heatmap.png`     | All indicators across all symbols               |
| `equity_curve.png`       | Net liquidation + drawdown time series          |
| `position_pnl.png`       | Unrealised P&L per open position                |
| `session_dashboard.png`  | Consolidated session overview                   |

---

## Project Structure

```
14-automated-trading-bot/
├── main.py                     # Entry point & demo runner
├── src/
│   ├── config.py               # Centralised configuration (dataclasses)
│   ├── ib_connector.py         # IB Gateway connection & data fetcher
│   ├── signal_generator.py     # RSI + MACD + BB + EMA consensus signals
│   ├── risk_manager.py         # Position sizing & risk guards
│   ├── order_manager.py        # Order lifecycle (limit, bracket, dry-run)
│   ├── position_monitor.py     # Real-time P&L & equity curve
│   ├── notifier.py             # SMTP email alerts
│   ├── plotter.py              # Publication-quality charts (Agg backend)
│   └── trading_bot.py          # Top-level orchestrator
├── tests/
│   └── test_trading_bot.py     # 32-test suite (unit + integration)
├── outputs/
│   ├── charts/                 # Generated PNG charts
│   └── logs/                   # Daily rotating log files
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Stack

- **Broker API**: ib_insync 0.9.x (asyncio wrapper for IB TWS API)
- **Data Processing**: pandas, numpy
- **Visualisation**: matplotlib (Agg backend - headless / Cloud Shell safe)
- **Notifications**: smtplib (TLS SMTP)
- **Testing**: pytest

---

## License

MIT License
