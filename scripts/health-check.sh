#!/bin/bash

set -e

echo "🏥 Running comprehensive health check..."

NAMESPACE=${1:-financial-rag}
SERVICE=${2:-financial-rag-service}
PORT=${3:-8000}

# Check if namespace exists
echo "1. Checking namespace..."
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || {
    echo "❌ Namespace $NAMESPACE does not exist"
    exit 1
}

# Check deployment status
echo "2. Checking deployment..."
DEPLOYMENT_STATUS=$(kubectl get deployment financial-rag-api -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
if [ "$DEPLOYMENT_STATUS" != "True" ]; then
    echo "❌ Deployment not available"
    exit 1
fi

# Check pod status
echo "3. Checking pods..."
POD_READY=$(kubectl get pods -n $NAMESPACE -l app=financial-rag-api -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')
if [ "$POD_READY" != "True" ]; then
    echo "❌ Pod not ready"
    exit 1
fi

# Port forward and test API
echo "4. Testing API health endpoint..."
kubectl port-forward service/$SERVICE $PORT:$PORT -n $NAMESPACE > /dev/null 2>&1 &
PORT_FORWARD_PID=$!

# Wait for port forward to be established
sleep 5

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health || true)

# Kill port forward
kill $PORT_FORWARD_PID > /dev/null 2>&1 || true

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Health check passed - API is responding"
else
    echo "❌ Health check failed - API returned HTTP $HEALTH_RESPONSE"
    exit 1
fi

echo "🎉 All health checks passed! System is operational."