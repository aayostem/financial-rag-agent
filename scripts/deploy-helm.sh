#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
ENVIRONMENT="development"
NAMESPACE="financial-rag"
DRY_RUN=false
UPGRADE=false

# Function to print usage
usage() {
    echo "Usage: $0 [-e environment] [-n namespace] [-d] [-u]"
    echo "  -e  Environment (development|staging|production) [default: development]"
    echo "  -n  Kubernetes namespace [default: financial-rag]"
    echo "  -d  Dry run mode"
    echo "  -u  Upgrade existing release"
    echo "  -h  Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "e:n:duh" opt; do
    case $opt in
        e) ENVIRONMENT="$OPTARG" ;;
        n) NAMESPACE="$OPTARG" ;;
        d) DRY_RUN=true ;;
        u) UPGRADE=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|staging|production) ;;
    *) echo -e "${RED}Error: Environment must be one of: development, staging, production${NC}"; exit 1 ;;
esac

# Set values file
VALUES_FILE="values-${ENVIRONMENT}.yaml"

# Check if values file exists
if [[ ! -f "helm/${VALUES_FILE}" ]]; then
    echo -e "${RED}Error: Values file helm/${VALUES_FILE} not found${NC}"
    exit 1
fi

echo -e "${GREEN}Deploying Financial RAG Agent to ${ENVIRONMENT} environment${NC}"
echo -e "${YELLOW}Namespace: ${NAMESPACE}${NC}"
echo -e "${YELLOW}Values file: ${VALUES_FILE}${NC}"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating namespace ${NAMESPACE}...${NC}"
    kubectl create namespace "$NAMESPACE"
fi

# Add bitnami repository if not already added
if ! helm repo list | grep -q bitnami; then
    echo -e "${YELLOW}Adding bitnami helm repository...${NC}"
    helm repo add bitnami https://charts.bitnami.com/bitnami
fi

# Update helm dependencies
echo -e "${YELLOW}Updating helm dependencies...${NC}"
helm dependency update helm/

# Build helm command
HELM_CMD="helm"
if [[ "$DRY_RUN" == true ]]; then
    HELM_CMD="$HELM_CMD --dry-run"
    echo -e "${YELLOW}Running in dry-run mode${NC}"
fi

if [[ "$UPGRADE" == true ]]; then
    HELM_CMD="$HELM_CMD upgrade --install"
else
    HELM_CMD="$HELM_CMD install"
fi

# Deploy the chart
RELEASE_NAME="financial-rag-agent-${ENVIRONMENT}"

echo -e "${YELLOW}Deploying release ${RELEASE_NAME}...${NC}"

$HELM_CMD "$RELEASE_NAME" helm/ \
    --namespace "$NAMESPACE" \
    --values "helm/${VALUES_FILE}" \
    --set global.env="$ENVIRONMENT" \
    --set global.domain="financial-rag-${ENVIRONMENT}.yourcompany.com" \
    --timeout 10m \
    --wait

if [[ "$DRY_RUN" == false ]]; then
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    
    # Wait for pods to be ready
    echo -e "${YELLOW}Waiting for all pods to be ready...${NC}"
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=financial-rag-agent -n "$NAMESPACE" --timeout=300s
    
    # Run tests
    echo -e "${YELLOW}Running post-deployment tests...${NC}"
    helm test "$RELEASE_NAME" --namespace "$NAMESPACE"
    
    # Show services
    echo -e "${GREEN}Deployment Summary:${NC}"
    kubectl get svc -n "$NAMESPACE" | grep financial-rag
fi