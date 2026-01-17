# Create and activate virtual environment (choose one)

# Option A: venv (built-in)
python -m venv .venv
source financial-rag-env/bin/activate  # Linux/Mac
# financial-rag-env\Scripts\activate  # Windows

# Install package in editable mode
pip install -e .

# For development dependencies
pip install -e ".[dev]"

# Run the test
python run_test.py


# Test production readiness
python test_production.py

# Build Docker image
docker build -t financial-rag-agent .

# Run with Docker Compose
docker-compose up -d

# Test the running API
python scripts/test_production.py

# Check logs
docker-compose logs -f


python scripts/start_api.py

docker-compose up -d



# section 3 starts

# Make scripts executable
chmod +x scripts/*.sh

# Deploy to Kubernetes
./scripts/deploy.sh

# Run health check
./scripts/health-check.sh

# Or deploy with kustomize
kubectl apply -k kubernetes/

# Check status
kubectl get all -n financial-rag

# View logs
kubectl logs -f deployment/financial-rag-api -n financial-rag

# Port forward for local access
kubectl port-forward service/financial-rag-service 8000:8000 -n financial-rag



# HELM WORKFLOW
# deploy to development
./scripts/deploy-helm.sh -e development


# deploy to production
./scripts/deploy-helm.sh -e production


# Manual deployment
# development
helm dependency update helm/
helm install financial-rag-dev helm/ \
  --namespace financial-rag \
  --values helm/values-development.yaml \
  --create-namespace


# production
helm dependency update helm/
helm install financial-rag-prod helm/ \
  --namespace financial-rag \
  --values helm/values-production.yaml \
  --create-namespace \
  --wait

#   you cn create a custom-values.yaml
helm upgrade financial-rag-dev helm/ \
  --namespace financial-rag \
  -f helm/values-production.yaml \
  -f custom-values.yaml

# accesing logs
# Application logs
kubectl logs -l app.kubernetes.io/name=financial-rag-agent -n financial-rag

# PostgreSQL logs
kubectl logs -l app=postgresql -n financial-rag

# Redis logs
kubectl logs -l app=redis -n financial-rag

# monitoring
# Port forward to Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000


# automted backups
# Run full backup
python scripts/backup-manager.py --environment production

# Backup specific component
python scripts/backup-manager.py --environment production --component postgresql

# manual backups
# PostgreSQL backup
kubectl exec -n financial-rag financial-rag-prod-postgresql-0 -- pg_dump -U financial_user financial_rag > backup.sql

# Redis backup
kubectl exec -n financial-rag financial-rag-prod-redis-master-0 -- redis-cli SAVE
kubectl cp financial-rag/financial-rag-prod-redis-master-0:/data/dump.rdb ./redis-backup.rdb


# clean up
helm uninstall financial-rag-prod -n financial-rag

complete cleanup including PVCs
helm uninstall financial-rag-prod -n financial-rag
kubectl delete pvc -l app.kubernetes.io/name=financial-rag-agent -n financial-rag
kubectl delete namespace financial-rag



# COMPLETE HELM DEPLOYMENT COMMANDS

## 🎯 Complete Deployment Commands

### **Quick Deployment Script** (`deploy-all.sh`)

```bash
#!/bin/bash

# Make scripts executable
chmod +x scripts/deploy-helm.sh
chmod +x scripts/backup-manager.py

# Deploy to development
echo "Deploying to development..."
./scripts/deploy-helm.sh -e development

# Wait for deployment to complete
sleep 30

# Run initial backup
echo "Running initial backup..."
python scripts/backup-manager.py --environment development

echo "Deployment completed!"
echo "Access your application at: http://financial-rag-development.yourcompany.com"



# TERRAFORM WORKFLOW
# DEPLOY
# Plan deployment
./scripts/deploy-env.sh dev plan

# Apply deployment
./scripts/deploy-env.sh dev apply


# AFTER DEPLOY
# Configure kubectl
export KUBECONFIG=$(terraform output -raw kubeconfig_filename)

# Deploy the application using Helm
cd ../helm
./scripts/deploy-helm.sh -e dev


# ENVIRONMENT MNAGER
./scripts/deploy-env.sh dev apply # DEVELOPMENT
./scripts/deploy-env.sh staging apply # STAGING
./scripts/deploy-env.sh prod apply # PRODUCTION


# CLEAN UP
./scripts/deploy-env.sh dev destroy


