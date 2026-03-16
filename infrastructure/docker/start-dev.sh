#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Start core services
echo "🚀 Starting PostgreSQL and Redis..."
docker compose up -d postgres redis

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Show status
docker compose ps

# Optionally start tools
read -p "Start management tools (pgAdmin, Redis Commander)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🛠️  Starting management tools..."
    docker compose --profile tools up -d
fi

echo "✅ Development environment ready!"
echo "📊 PostgreSQL: localhost:5432"
echo "⚡ Redis: localhost:6379"
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🐘 pgAdmin: http://localhost:5050 (admin@finrag.local / admin)"
    echo "🔧 Redis Commander: http://localhost:8081"
fi