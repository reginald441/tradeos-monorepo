# Nalvenix Innovations (TradeOS)

A production-ready multi-asset algorithmic trading platform supporting Crypto, Forex, Gold, Commodities, and Indices.

## Features

- **6-Layer Architecture**: Data, Strategy, Risk, Execution, Quant, and SaaS layers
- **Real-time Trading**: Live market data and trade execution
- **Strategy Engine**: Create, test, and deploy algorithmic strategies
- **Risk Management**: Position sizing, VaR calculations, kill switch
- **Backtesting**: Test strategies against historical data
- **Analytics**: Comprehensive performance metrics and visualizations
- **Multi-Asset Support**: Crypto, Forex, Commodities, Indices

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL 16, TimescaleDB, Redis
- **Frontend**: React 18, TypeScript, Tailwind CSS, Recharts
- **Infrastructure**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for local development)

### Using Docker Compose

1. Clone the repository:
```bash
git clone <repository-url>
cd nalvenix-tradeos
```

2. Copy the environment file:
```bash
cp .env.example .env
```

3. Start all services:
```bash
docker-compose up -d
```

4. Access the application:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Default Login

- **Email**: reginald@nalvenix.com
- **Password**: password

## Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - Login
- `POST /api/auth/register` - Register
- `GET /api/auth/me` - Get current user

### Trading
- `GET /api/dashboard` - Dashboard data
- `GET /api/trades` - List trades
- `POST /api/trades` - Create trade
- `POST /api/trades/{id}/close` - Close trade

### Strategies
- `GET /api/strategies` - List strategies
- `POST /api/strategies` - Create strategy
- `POST /api/strategies/{id}/toggle` - Toggle strategy

### Risk
- `GET /api/risk` - Get risk settings
- `PUT /api/risk` - Update risk settings

### Analytics
- `GET /api/analytics` - Get analytics data
- `GET /api/market-data` - Get market data

## Project Structure

```
nalvenix-tradeos/
├── backend/
│   ├── app.py              # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/           # API client functions
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── store/         # Zustand stores
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## License

Copyright © 2024 Nalvenix Innovations. All rights reserved.
