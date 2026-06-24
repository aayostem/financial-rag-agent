#!/usr/bin/env bash
set -euo pipefail
CLUSTER_NAME="${CLUSTER_NAME:-financial-rag-prod-cluster}"
ARGOCD_VERSION="${ARGOCD_VERSION:-2.10.4}"
ENV="${ENV:-prod}"
GITHUB_TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN}"
SLACK_TOKEN="${SLACK_TOKEN:-}"
PAGERDUTY_PROD_KEY="${PAGERDUTY_PROD_KEY:-}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Adding ArgoCD Helm repo..."
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace argocd istio-injection=disabled --overwrite

log "Installing ArgoCD v${ARGOCD_VERSION}..."
helm upgrade --install argocd argo/argo-cd \
  --version "$ARGOCD_VERSION" \
  --namespace argocd \
  --wait --timeout 10m

log "Waiting for ArgoCD server..."
kubectl rollout status deployment/argocd-server -n argocd --timeout=5m

ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d)

log "Adding Git repository..."
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
PF_PID=$!
sleep 3

argocd login localhost:8080 --username admin --password "$ARGOCD_PASSWORD" --insecure --grpc-web
argocd repo add https://github.com/aayostem/financial-rag-agent.git \
  --username git --password "$GITHUB_TOKEN" --name financial-rag-agent

POLICY=$(cat "$(dirname "$0")/../rbac/policy.csv")
kubectl patch configmap argocd-rbac-cm -n argocd --type merge \
  --patch "{\"data\":{\"policy.csv\":$(echo "$POLICY" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}}"

if [[ -n "$SLACK_TOKEN" ]]; then
  kubectl create secret generic argocd-notifications-secret \
    --namespace argocd \
    --from-literal=slack-token="$SLACK_TOKEN" \
    --from-literal=pagerduty-prod-key="$PAGERDUTY_PROD_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

kubectl apply -f "$(dirname "$0")/../notifications/templates.yaml"
kubectl apply -f "$(dirname "$0")/../projects/financial-rag-project.yaml"
kubectl apply -f "$(dirname "$0")/../appsets/infra-appset.yaml"
kubectl apply -f "$(dirname "$0")/../appsets/env-appset.yaml"
kubectl apply -f "$(dirname "$0")/../appsets/apps-appset.yaml"

kill "$PF_PID" 2>/dev/null || true

log "✅ ArgoCD bootstrap complete."
log "   kubectl get applicationsets -n argocd"
log "   kubectl get applications -n argocd"
