#!/bin/bash

set -e

echo "🚀 Deploying Financial RAG Agent to Kubernetes..."

# Validate environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required"
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t financial-rag-agent:latest .

# If using remote registry, push the image
# docker tag financial-rag-agent:latest your-registry/financial-rag-agent:latest
# docker push your-registry/financial-rag-agent:latest

# Create namespace if it doesn't exist
echo "📁 Creating Kubernetes namespace..."
kubectl apply -f kubernetes/namespace.yaml

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic financial-rag-secrets \
    --namespace=financial-rag \
    --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
    --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply all Kubernetes manifests
echo "📄 Applying Kubernetes manifests..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/persistent-volume-claim.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/financial-rag-api -n financial-rag --timeout=300s

# Get service information
echo "🌐 Service information:"
kubectl get service -n financial-rag

echo "✅ Deployment completed successfully!"
echo "📊 Check logs: kubectl logs -f deployment/financial-rag-api -n financial-rag"
echo "🌐 Access API: kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag"