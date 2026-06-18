#!/bin/bash
# scripts/stop.sh
# Stop all services

echo "🛑 Stopping services..."
docker compose -f infrastructure/docker/docker-compose.dev.yml down
echo "✅ Services stopped!"
