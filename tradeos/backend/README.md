# TradeOS Backend

Advanced Algorithmic Trading Platform Backend API built with FastAPI and SQLAlchemy 2.0.

## Features

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy 2.0**: Async ORM for database operations
- **PostgreSQL**: Primary database with asyncpg driver
- **Redis**: Caching and rate limiting
- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Configurable rate limiting with multiple strategies
- **Structured Logging**: JSON logging with correlation IDs
- **Pydantic Settings**: Environment-based configuration
- **Comprehensive Models**: 10+ database models for trading operations

## Project Structure

```
tradeos/backend/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── config/
│   ├── __init__.py
│   └── settings.py        # Pydantic settings configuration
├── database/
│   ├── __init__.py
│   ├── connection.py      # Async database engine & session management
│   └── models.py          # All SQLAlchemy models
├── dependencies/
│   ├── __init__.py
│   └── auth.py            # JWT authentication & dependencies
├── middleware/
│   ├── __init__.py
│   ├── logging.py         # Request/response logging middleware
│   └── rate_limit.py      # Rate limiting middleware
└── utils/
    ├── __init__.py
    ├── helpers.py         # Utility functions
    └── validators.py      # Input validation utilities
```

## Database Models

1. **User** - Platform users with roles and subscription tiers
2. **Strategy** - Trading strategies with configuration
3. **Trade** - Executed trades with PnL tracking
4. **Position** - Open positions with unrealized PnL
5. **MarketData** - OHLCV candle data
6. **Order** - Exchange orders with status tracking
7. **RiskProfile** - User risk management settings
8. **Subscription** - User billing/subscription management
9. **ApiKey** - API keys for programmatic access
10. **BacktestResult** - Strategy backtesting results

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Set Up Database

```bash
# Create PostgreSQL database
createdb tradeos

# Run migrations (when alembic is set up)
# alembic upgrade head
```

### 4. Run Application

```bash
# Development mode
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Checks
- `GET /health` - Health status
- `GET /ready` - Kubernetes readiness probe
- `GET /live` - Kubernetes liveness probe

### API v1 Routes

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/auth/register` | User registration |
| `POST /api/v1/auth/login` | User login |
| `POST /api/v1/auth/refresh` | Token refresh |
| `GET /api/v1/users/me` | Get current user |
| `GET /api/v1/strategies` | List strategies |
| `POST /api/v1/strategies` | Create strategy |
| `GET /api/v1/trades` | List trades |
| `POST /api/v1/trades` | Create trade |
| `GET /api/v1/positions` | List positions |
| `GET /api/v1/orders` | List orders |
| `POST /api/v1/orders` | Create order |
| `GET /api/v1/market/candles/{symbol}` | Get candle data |
| `GET /api/v1/market/ticker/{symbol}` | Get ticker |
| `GET /api/v1/risk/profile` | Get risk profile |
| `GET /api/v1/subscriptions/current` | Get subscription |
| `GET /api/v1/api-keys` | List API keys |
| `GET /api/v1/backtests` | List backtests |

## Configuration

All configuration is managed through environment variables with sensible defaults.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `DB_HOST` | Database host | `localhost` |
| `DB_NAME` | Database name | `tradeos` |
| `JWT_SECRET_KEY` | JWT signing key | Required |
| `REDIS_HOST` | Redis host | `localhost` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `true` |

## Authentication

The API supports two authentication methods:

### 1. JWT Token (Browser/Mobile)

```bash
# Login to get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token in requests
curl http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer <token>"
```

### 2. API Key (Programmatic)

```bash
curl http://localhost:8000/api/v1/trades \
  -H "X-API-Key: trd_your_api_key_here"
```

## Rate Limiting

Rate limits are configurable per endpoint:

| Endpoint | Default Limit |
|----------|---------------|
| `/api/v1/auth/*` | 10/minute |
| `/api/v1/trades/*` | 50/minute |
| `/api/v1/orders/*` | 50/minute |
| `/api/v1/market/*` | 200/minute |
| Authenticated users | 1000/minute |

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy .
```

## License

MIT License - See LICENSE file for details.
