#!/bin/bash

# create-terraform-structure.sh

echo "Creating Terraform folder structure..."

# Create main terraform directory
mkdir -p terraform

# Create root Terraform files
touch terraform/main.tf
touch terraform/variables.tf
touch terraform/outputs.tf
touch terraform/terraform.tfvars.example
touch terraform/versions.tf
touch terraform/providers.tf

# Create modules directory structure
mkdir -p terraform/modules/kubernetes
mkdir -p terraform/modules/networking
mkdir -p terraform/modules/database
mkdir -p terraform/modules/storage
mkdir -p terraform/modules/monitoring
mkdir -p terraform/modules/dns

# Create module files for kubernetes
touch terraform/modules/kubernetes/main.tf
touch terraform/modules/kubernetes/variables.tf
touch terraform/modules/kubernetes/outputs.tf
touch terraform/modules/kubernetes/kubeconfig.tf

# Create module files for networking
touch terraform/modules/networking/main.tf
touch terraform/modules/networking/variables.tf
touch terraform/modules/networking/outputs.tf
touch terraform/modules/networking/security-groups.tf

# Create module files for database
touch terraform/modules/database/main.tf
touch terraform/modules/database/variables.tf
touch terraform/modules/database/outputs.tf
touch terraform/modules/database/backups.tf

# Create module files for storage
touch terraform/modules/storage/main.tf
touch terraform/modules/storage/variables.tf
touch terraform/modules/storage/outputs.tf
touch terraform/modules/storage/backups.tf

# Create module files for monitoring
touch terraform/modules/monitoring/main.tf
touch terraform/modules/monitoring/variables.tf
touch terraform/modules/monitoring/outputs.tf
touch terraform/modules/monitoring/dashboards.tf

# Create module files for dns
touch terraform/modules/dns/main.tf
touch terraform/modules/dns/variables.tf
touch terraform/modules/dns/outputs.tf

# Create environments directory structure
mkdir -p terraform/environments/dev
mkdir -p terraform/environments/staging
mkdir -p terraform/environments/prod

# Create environment files for dev
touch terraform/environments/dev/main.tf
touch terraform/environments/dev/variables.tf
touch terraform/environments/dev/terraform.tfvars
touch terraform/environments/dev/backend.tf

# Create environment files for staging
touch terraform/environments/staging/main.tf
touch terraform/environments/staging/variables.tf
touch terraform/environments/staging/terraform.tfvars
touch terraform/environments/staging/backend.tf

# Create environment files for prod
touch terraform/environments/prod/main.tf
touch terraform/environments/prod/variables.tf
touch terraform/environments/prod/terraform.tfvars
touch terraform/environments/prod/backend.tf

# Create scripts directory
mkdir -p terraform/scripts

# Create script files
touch terraform/scripts/init-backend.sh
touch terraform/scripts/deploy-env.sh
touch terraform/scripts/destroy-env.sh
touch terraform/scripts/terraform-wrapper.sh

echo "Terraform folder structure created successfully!"
echo "Location: $(pwd)/terraform"