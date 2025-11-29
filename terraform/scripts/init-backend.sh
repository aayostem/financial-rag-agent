#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 <environment>"
    echo "  environment: dev, staging, prod"
    exit 1
}

# Validate arguments
if [ $# -ne 1 ]; then
    usage
fi

ENVIRONMENT=$1

# Validate environment
case $ENVIRONMENT in
    dev|staging|prod) ;;
    *) echo -e "${RED}Error: Environment must be one of: dev, staging, prod${NC}"; exit 1 ;;
esac

# Generate backend configuration
BACKEND_CONFIG=$(cat << EOF
bucket = "financial-rag-tfstate-${ENVIRONMENT}"
key    = "terraform.tfstate"
region = "us-west-2"
dynamodb_table = "financial-rag-tfstate-lock-${ENVIRONMENT}"
encrypt = true
EOF
)

ENV_DIR="$TERRAFORM_DIR/environments/$ENVIRONMENT"

# Create backend.tf if it doesn't exist
if [ ! -f "$ENV_DIR/backend.tf" ]; then
    echo -e "${YELLOW}Creating backend.tf for $ENVIRONMENT...${NC}"
    cat > "$ENV_DIR/backend.tf" << EOF
terraform {
  backend "s3" {
$BACKEND_CONFIG
  }
}
EOF
    echo -e "${GREEN}Backend configuration created at $ENV_DIR/backend.tf${NC}"
else
    echo -e "${YELLOW}backend.tf already exists in $ENV_DIR${NC}"
fi

# Create S3 bucket for Terraform state
echo -e "${YELLOW}Creating S3 bucket for Terraform state...${NC}"
aws s3 mb s3://financial-rag-tfstate-${ENVIRONMENT} --region us-west-2 || true

# Create DynamoDB table for state locking
echo -e "${YELLOW}Creating DynamoDB table for state locking...${NC}"
aws dynamodb create-table \
    --table-name financial-rag-tfstate-lock-${ENVIRONMENT} \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --region us-west-2 \
    --query TableDescription.TableArn \
    --output text || true

echo -e "${GREEN}Backend initialization completed for $ENVIRONMENT!${NC}"
echo -e "${YELLOW}You can now run: ./scripts/deploy-env.sh $ENVIRONMENT plan${NC}"