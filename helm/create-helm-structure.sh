#!/bin/bash

# create-helm-structure.sh

echo "Creating Helm folder structure..."

# Create main directories
mkdir -p helm/templates/tests
mkdir -p helm/charts/redis/templates
mkdir -p helm/charts/postgresql/templates
mkdir -p helm/charts/vector-store/templates
mkdir -p helm/charts/monitoring/templates

# Create empty files in main helm directory
touch helm/Chart.yaml
touch helm/values.yaml
touch helm/values-production.yaml
touch helm/values-staging.yaml
touch helm/values-development.yaml
touch helm/.helmignore

# Create empty template files
touch helm/templates/_helpers.tpl
touch helm/templates/namespace.yaml
touch helm/templates/configmap.yaml
touch helm/templates/secret.yaml
touch helm/templates/serviceaccount.yaml
touch helm/templates/deployment.yaml
touch helm/templates/service.yaml
touch helm/templates/ingress.yaml
touch helm/templates/hpa.yaml
touch helm/templates/pdb.yaml
touch helm/templates/network-policy.yaml
touch helm/templates/service-monitor.yaml

# Create empty test files
touch helm/templates/tests/test-connection.yaml
touch helm/templates/tests/test-health.yaml

# Create empty chart files for subcharts
touch helm/charts/redis/Chart.yaml
touch helm/charts/redis/values.yaml
touch helm/charts/redis/templates/deployment.yaml
touch helm/charts/redis/templates/service.yaml
touch helm/charts/redis/templates/configmap.yaml

touch helm/charts/postgresql/Chart.yaml
touch helm/charts/postgresql/values.yaml
touch helm/charts/postgresql/templates/statefulset.yaml
touch helm/charts/postgresql/templates/service.yaml
touch helm/charts/postgresql/templates/pvc.yaml

touch helm/charts/vector-store/Chart.yaml
touch helm/charts/vector-store/values.yaml
touch helm/charts/vector-store/templates/deployment.yaml
touch helm/charts/vector-store/templates/service.yaml

touch helm/charts/monitoring/Chart.yaml
touch helm/charts/monitoring/values.yaml
touch helm/charts/monitoring/templates/prometheus.yaml
touch helm/charts/monitoring/templates/grafana.yaml

echo "Helm folder structure created successfully!"
echo "Location: $(pwd)/helm"