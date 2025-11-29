#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 <environment> [action]"
    echo "  environment: dev, staging, prod"
    echo "  action: plan, apply, destroy (default: apply)"
    exit 1
}

# Validate arguments
if [ $# -lt 1 ]; then
    usage
fi

ENVIRONMENT=$1
ACTION=${2:-"apply"}

# Validate environment
case $ENVIRONMENT in
    dev|staging|prod) ;;
    *) echo -e "${RED}Error: Environment must be one of: dev, staging, prod${NC}"; exit 1 ;;
esac

# Validate action
case $ACTION in
    plan|apply|destroy) ;;
    *) echo -e "${RED}Error: Action must be one of: plan, apply, destroy${NC}"; exit 1 ;;
esac

ENV_DIR="$TERRAFORM_DIR/environments/$ENVIRONMENT"

# Check if environment directory exists
if [ ! -d "$ENV_DIR" ]; then
    echo -e "${RED}Error: Environment directory $ENV_DIR not found${NC}"
    exit 1
fi

# Check for terraform.tfvars
if [ ! -f "$ENV_DIR/terraform.tfvars" ]; then
    echo -e "${YELLOW}Warning: terraform.tfvars not found in $ENV_DIR${NC}"
    echo -e "${YELLOW}You may need to create it from terraform.tfvars.example${NC}"
fi

echo -e "${GREEN}Performing $ACTION on $ENVIRONMENT environment${NC}"
echo -e "${YELLOW}Terraform directory: $ENV_DIR${NC}"

cd "$ENV_DIR"

# Initialize terraform
echo -e "${YELLOW}Initializing Terraform...${NC}"
terraform init -reconfigure

# Run terraform action
case $ACTION in
    plan)
        echo -e "${YELLOW}Running terraform plan...${NC}"
        terraform plan -var-file="terraform.tfvars"
        ;;
    apply)
        echo -e "${YELLOW}Running terraform apply...${NC}"
        terraform apply -var-file="terraform.tfvars" -auto-approve
        echo -e "${GREEN}Deployment completed successfully!${NC}"
        
        # Show outputs
        echo -e "${YELLOW}Deployment outputs:${NC}"
        terraform output
        ;;
    destroy)
        echo -e "${RED}WARNING: This will destroy all resources in $ENVIRONMENT environment${NC}"
        read -p "Are you sure you want to continue? (yes/no): " confirmation
        
        if [ "$confirmation" = "yes" ]; then
            echo -e "${YELLOW}Running terraform destroy...${NC}"
            terraform destroy -var-file="terraform.tfvars" -auto-approve
            echo -e "${GREEN}Destruction completed!${NC}"
        else
            echo -e "${YELLOW}Destruction cancelled${NC}"
        fi
        ;;
esac