#!/bin/bash

# Nalvenix Innovations (TradeOS) - Start Script
# This script starts all services using Docker Compose

echo "🚀 Starting Nalvenix Innovations (TradeOS)..."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
fi

# Start services
echo "🔧 Building and starting services..."
docker-compose up --build -d

echo ""
echo "✅ Services started successfully!"
echo ""
echo "📱 Frontend: http://localhost"
echo "🔌 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🔑 Default Login:"
echo "   Email: reginald@nalvenix.com"
echo "   Password: password"
echo ""
echo "🛑 To stop services, run: docker-compose down"
