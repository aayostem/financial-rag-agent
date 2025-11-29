# Financial RAG Agent - Terraform Infrastructure

This Terraform project deploys the complete infrastructure for the Financial RAG Agent on AWS.

## Architecture Overview

The infrastructure includes:

- **VPC** with public and private subnets across multiple AZs
- **EKS Cluster** with managed node groups
- **RDS PostgreSQL** database with encryption and backups
- **S3 Buckets** for Terraform state, backups, and logs
- **CloudWatch** for monitoring and alerting
- **Route53** for DNS management
- **Security Groups** and **IAM Roles** for security

## Prerequisites

1. **AWS CLI** configured with appropriate permissions
2. **Terraform** 1.0.0 or later
3. **kubectl** for Kubernetes management
4. **helm** for application deployment

## Quick Start

### 1. Initialize Backend

```bash
# Initialize Terraform backend for your environment
./scripts/init-backend.sh dev


# USEFUL COMMAND
# View Terraform state
terraform state list

# Import existing resources
terraform import aws_resource.name id

# Taint resources for recreation
terraform taint aws_resource.name

# View outputs
terraform output

