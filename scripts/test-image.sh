#!/bin/bash
# scripts/fix-container.sh

echo "🧹 Cleaning up old containers..."
docker rm -f finrag-test finrag-app 2>/dev/null || true

echo "🚀 Starting fresh container..."
HOST_IP=$(ipconfig | grep -i "IPv4" | grep -v "127.0.0.1" | head -1 | awk '{print $NF}')

docker run -d --name finrag-app \
  -p 8000:8000 \
  -e POSTGRES_PASSWORD=finrag-dev-password-change-in-prod \
  -e REDIS_PASSWORD=redis-dev-password-change-in-prod \
  -e APP_ENV=testing \
  -e MOCK_EXTERNAL_APIS=true \
  -e POSTGRES_HOST=$HOST_IP \
  -e REDIS_HOST=$HOST_IP \
  -e POSTGRES_USER=finrag \
  -e POSTGRES_DB=financial_rag \
  financial-rag-agent:latest

echo "⏳ Waiting for container to start..."
sleep 10

echo "📦 Installing sentence-transformers..."
docker exec finrag-app python -m pip install sentence-transformers

echo "🔄 Restarting container..."
docker restart finrag-app

echo "⏳ Waiting for restart..."
sleep 20

echo "🏥 Testing health endpoint..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
    curl -s http://localhost:8000/health | python -m json.tool
else
    echo "❌ Health check failed!"
    echo "Container logs:"
    docker logs --tail 50 finrag-app
fi