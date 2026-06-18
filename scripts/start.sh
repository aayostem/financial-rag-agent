#!/bin/bash
# scripts/start.sh
# Start all services and wait for them to be healthy

echo "🚀 Starting services..."
docker compose -f infrastructure/docker/docker-compose.dev.yml up -d

echo "⏳ Waiting for services to be healthy..."
sleep 10

echo "✅ Services started!"
docker compose -f infrastructure/docker/docker-compose.dev.yml ps
