# TradeOS - Algorithmic Trading Operating System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/FastAPI-0.104-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-blue.svg" alt="React 18">
  <img src="https://img.shields.io/badge/PostgreSQL-16-blue.svg" alt="PostgreSQL 16">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker Ready">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

**TradeOS** is a comprehensive, multi-asset algorithmic trading SaaS platform designed for professional traders and institutions. It provides a complete infrastructure stack for developing, backtesting, and executing trading strategies across multiple asset classes.

## 🎯 Vision

TradeOS is not just a trading bot or signal service—it's a **complete trading operating system** with modular architecture supporting:

- **Crypto**: BTC, ETH, SOL, and 100+ cryptocurrencies
- **Forex**: EUR/USD, GBP/USD, and major currency pairs
- **Gold**: XAU/USD spot and futures
- **Commodities**: Oil, natural gas, agricultural products
- **Indices**: S&P 500, NASDAQ, DOW, and global indices

## 🏗 Architecture

TradeOS follows a 6-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 6: SaaS Infrastructure                  │
│  Auth, Billing, Subscriptions, Admin Panel, User Dashboard      │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 5: Quantum/Quant Layer                  │
│  Monte Carlo, Portfolio Optimization, RL, Bayesian Inference    │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 4: Execution Engine                     │
│  Exchange APIs, Order Management, Slippage Modeling             │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 3: Risk Engine                          │
│  Position Sizing, Drawdown Control, VaR, Kill Switch            │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 2: Strategy Engine                      │
│  Trend Following, Mean Reversion, Volatility, Liquidity         │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 1: Data Layer                           │
│  WebSocket Feeds, OHLC Aggregation, Market Microstructure       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git
- Make (optional, for convenience commands)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tradeos.git
   cd tradeos
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start all services**
   ```bash
   make up
   # Or: docker-compose up -d
   ```

4. **Run database migrations**
   ```bash
   make db-migrate
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Backend API: http://localhost:8000
   - Grafana: http://localhost:3001

### Windows One-Command Health Check

If VS Code feels stuck or you want a single automated check, run this in **PowerShell** from the `tradeos` folder:

```powershell
./scripts/windows-health-check.ps1
```

This script will:
- create `.env` from `.env.example` if missing,
- enforce valid `CORS_ORIGINS` JSON array format,
- validate Compose config,
- start/rebuild services,
- auto-fix a legacy `nginx.dev.conf` mount reference in `docker-compose.override.yml` if found,
- avoid the common Grafana/Frontend 3000 port collision by defaulting Grafana to 3001, and
- test backend health endpoints (`/health`, `/ready`, `/live`) with readiness retries.

PowerShell tip: keep the full compose command on **one line**.

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml down --remove-orphans
```

If you split after `-f`, PowerShell treats the next token as a separate command and shows `flag needs an argument: 'f'`.

### Makefile Commands

```bash
make up              # Start all services
make down            # Stop all services
make logs            # View logs
make build           # Rebuild containers
make db-migrate      # Run database migrations
make db-upgrade      # Upgrade database schema
make shell-backend   # Access backend container shell
make shell-db        # Access database shell
make test            # Run tests
make lint            # Run linters
make deploy          # Deploy to production
```

## 📁 Project Structure

```
tradeos/
├── backend/                 # FastAPI Backend
│   ├── app.py              # Main application
│   ├── config/             # Configuration
│   ├── database/           # Database models & connection
│   ├── dependencies/       # FastAPI dependencies
│   ├── middleware/         # Custom middleware
│   ├── routers/            # API route handlers
│   ├── data/               # Data Layer (Layer 1)
│   ├── strategies/         # Strategy Engine (Layer 2)
│   ├── risk/               # Risk Engine (Layer 3)
│   ├── execution/          # Execution Engine (Layer 4)
│   ├── quant/              # Quant Layer (Layer 5)
│   ├── saas/               # SaaS Layer (Layer 6)
│   └── utils/              # Utility functions
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── store/          # Zustand state management
│   │   ├── api/            # API client
│   │   └── styles/         # CSS/Tailwind
│   └── public/
├── nginx/                  # Nginx configuration
├── monitoring/             # Prometheus & Grafana
├── scripts/                # Utility scripts
├── docker-compose.yml      # Docker orchestration
├── Makefile               # Convenience commands
└── README.md
```


## 🔄 Syncing Local Fixes to GitHub

Changes made in this environment are local commits until you push them to your remote repository.

```bash
# 1) Add your GitHub remote once (if missing)
git remote add origin https://github.com/reginald441/tradeos-monorepo.git

# 2) Verify remotes
git remote -v

# 3) Push your current branch (example: work)
git push -u origin work

# 4) Open a PR on GitHub from work -> main
```

If `origin` already exists and points somewhere else, update it:

```bash
git remote set-url origin https://github.com/reginald441/tradeos-monorepo.git
```

## 🔧 Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

```env
# Database
DATABASE_URL=postgresql+asyncpg://tradeos:tradeos@postgres:5432/tradeos

# Redis
REDIS_URL=redis://redis:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Stripe (for billing)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Exchange APIs
BINANCE_API_KEY=your-binance-key
BINANCE_SECRET_KEY=your-binance-secret

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 📊 Features

### Data Layer
- ✅ Real-time WebSocket market data feeds
- ✅ Multi-exchange data aggregation (Binance, Coinbase, Kraken)
- ✅ OHLCV candle generation and timeframe compression
- ✅ Market microstructure analysis
- ✅ Historical data import and storage

### Strategy Engine
- ✅ 30+ technical indicators
- ✅ Trend following strategies (EMA, MACD, ADX)
- ✅ Mean reversion strategies (RSI, Bollinger Bands)
- ✅ Volatility-based strategies
- ✅ Liquidity sweep detection
- ✅ Multi-timeframe confirmation
- ✅ Walk-forward optimization

### Risk Engine
- ✅ Dynamic position sizing (Kelly, ATR, Risk-per-trade)
- ✅ Drawdown circuit breakers
- ✅ Portfolio exposure limits
- ✅ Cross-asset correlation monitoring
- ✅ Value at Risk (VaR) calculations
- ✅ Emergency kill switch

### Execution Engine
- ✅ Multi-exchange execution (Binance, Coinbase, Kraken)
- ✅ Forex broker integration (MT5, cTrader)
- ✅ Paper trading mode
- ✅ Slippage modeling
- ✅ Order lifecycle management
- ✅ Latency tracking

### Quant Layer
- ✅ Monte Carlo simulation
- ✅ Portfolio optimization (Markowitz, Risk Parity)
- ✅ Reinforcement Learning agents (PPO, DQN)
- ✅ Bayesian inference
- ✅ GARCH volatility modeling
- ✅ Hidden Markov Models for regime detection

### SaaS Layer
- ✅ JWT/OAuth2 authentication
- ✅ Role-based access control
- ✅ Subscription tiers (Free, Pro, Enterprise)
- ✅ Stripe billing integration
- ✅ API key management
- ✅ Usage tracking

## 💰 Subscription Tiers

| Feature | Free | Pro ($99/mo) | Enterprise ($499/mo) |
|---------|------|--------------|---------------------|
| Strategies | 1 | 10 | Unlimited |
| Backtests/month | 5 | Unlimited | Unlimited |
| API Calls/day | 100 | 10,000 | 100,000 |
| Exchanges | 1 | 3 | Unlimited |
| Data History | 30 days | 1 year | Unlimited |
| Live Trading | ❌ | ✅ | ✅ |
| Risk Management | Basic | Advanced | Custom |
| Support | Community | Priority | Dedicated |
| Custom Strategies | ❌ | ❌ | ✅ |
| White Label | ❌ | ❌ | ✅ |

## 🔌 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user

### Trading
- `POST /api/v1/trading/orders` - Place order
- `GET /api/v1/trading/orders` - List orders
- `DELETE /api/v1/trading/orders/{id}` - Cancel order
- `GET /api/v1/trading/positions` - List positions
- `GET /api/v1/trading/trades` - Trade history
- `GET /api/v1/trading/portfolio` - Portfolio summary

### Strategies
- `GET /api/v1/strategies` - List strategies
- `POST /api/v1/strategies` - Create strategy
- `GET /api/v1/strategies/{id}` - Get strategy
- `PUT /api/v1/strategies/{id}` - Update strategy
- `POST /api/v1/strategies/{id}/toggle` - Activate/deactivate

### Risk Management
- `GET /api/v1/risk/profile` - Get risk profile
- `PUT /api/v1/risk/profile` - Update risk profile
- `GET /api/v1/risk/exposure` - Current exposure
- `GET /api/v1/risk/metrics` - Risk metrics
- `POST /api/v1/risk/kill-switch` - Emergency stop

### Backtesting
- `POST /api/v1/backtest/run` - Run backtest
- `GET /api/v1/backtest/results` - List results
- `GET /api/v1/backtest/results/{id}` - Get result
- `GET /api/v1/backtest/results/{id}/equity-curve` - Equity curve

### Market Data
- `GET /api/v1/market/symbols` - List symbols
- `GET /api/v1/market/ohlcv/{symbol}` - OHLCV data
- `GET /api/v1/market/ticker/{symbol}` - Current price
- `GET /api/v1/market/orderbook/{symbol}` - Order book
- `WS /api/v1/market/ws/price/{symbol}` - Real-time prices

### Billing
- `GET /api/v1/billing/plans` - Subscription plans
- `GET /api/v1/billing/subscription` - Current subscription
- `POST /api/v1/billing/subscribe` - Create subscription
- `POST /api/v1/billing/cancel` - Cancel subscription
- `GET /api/v1/billing/invoices` - Billing history

## 🧪 Testing

```bash
# Run all tests
make test

# Run backend tests only
cd backend && pytest

# Run frontend tests only
cd frontend && npm test

# Run with coverage
pytest --cov=backend --cov-report=html
```

## 📈 Monitoring

TradeOS includes comprehensive monitoring with Prometheus and Grafana:

- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request latency, error rates, throughput
- **Trading Metrics**: PnL, trade volume, strategy performance
- **Database Metrics**: Query performance, connection pool

Access Grafana at http://localhost:3001 (admin/admin)

## 🔒 Security

- JWT-based authentication with refresh tokens
- Password hashing with bcrypt
- API key authentication for programmatic access
- Rate limiting per endpoint
- CORS protection
- SQL injection prevention via SQLAlchemy
- XSS protection in frontend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database toolkit
- [React](https://reactjs.org/) - Frontend library
- [TimescaleDB](https://www.timescale.com/) - Time-series database
- [ ccxt](https://github.com/ccxt/ccxt) - Cryptocurrency trading library

## 📞 Support

- **Documentation**: https://docs.tradeos.io
- **Discord**: https://discord.gg/tradeos
- **Email**: support@tradeos.io
- **Twitter**: [@TradeOS](https://twitter.com/tradeos)

## 🗺 Roadmap

### Q1 2024
- [x] Core platform architecture
- [x] Basic trading strategies
- [x] Risk management system
- [x] Paper trading mode

### Q2 2024
- [ ] Mobile app (iOS/Android)
- [ ] Social trading features
- [ ] Advanced ML models
- [ ] More exchange integrations

### Q3 2024
- [ ] Options trading support
- [ ] Futures trading
- [ ] Institutional features
- [ ] White-label solutions

### Q4 2024
- [ ] AI-powered strategy builder
- [ ] Cross-chain DeFi integration
- [ ] Regulatory compliance tools
- [ ] Global market expansion

---

<p align="center">
  Built with ❤️ by the TradeOS Team
</p>
