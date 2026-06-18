#!/bin/bash
# scripts/cleanup.sh
# Stop all services and remove volumes

echo "🧹 Cleaning up..."
docker compose -f infrastructure/docker/docker-compose.dev.yml down -v
echo "✅ Cleanup complete!"
