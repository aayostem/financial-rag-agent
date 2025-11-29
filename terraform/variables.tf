# Project Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "financial-rag-agent"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "ai-engineering-team"
}

# AWS Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

# Kubernetes Variables
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "financial-rag-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "kubeconfig_path" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

# Node Group Variables
variable "node_groups" {
  description = "Map of node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size       = number
    max_size       = number
    desired_size   = number
    disk_size      = number
  }))
  default = {
    main = {
      instance_types = ["t3.medium", "t3a.medium"]
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 10
      desired_size   = 2
      disk_size      = 20
    }
    spot = {
      instance_types = ["t3.medium", "t3a.medium", "t2.medium"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 5
      desired_size   = 1
      disk_size      = 20
    }
  }
}

# Database Variables
variable "database_config" {
  description = "Database configuration"
  type = object({
    instance_class    = string
    allocated_storage = number
    engine_version    = string
    multi_az          = bool
    backup_retention  = number
  })
  default = {
    instance_class    = "db.t3.medium"
    allocated_storage = 20
    engine_version    = "15.2"
    multi_az          = false
    backup_retention  = 7
  }
}

# Storage Variables
variable "storage_config" {
  description = "Storage configuration for backups and data"
  type = object({
    backup_retention_days = number
    s3_lifecycle_rules = list(object({
      id      = string
      enabled = bool
      prefix  = string
      transition = list(object({
        days          = number
        storage_class = string
      }))
      expiration = object({
        days = number
      })
    }))
  })
  default = {
    backup_retention_days = 30
    s3_lifecycle_rules = [
      {
        id      = "backup-transition"
        enabled = true
        prefix  = "backups/"
        transition = [
          {
            days          = 30
            storage_class = "STANDARD_IA"
          },
          {
            days          = 60
            storage_class = "GLACIER"
          }
        ]
        expiration = {
          days = 365
        }
      }
    ]
  }
}

# Monitoring Variables
variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    enabled           = bool
    retention_days    = number
    alarm_email       = string
    slack_webhook_url = string
  })
  default = {
    enabled        = true
    retention_days = 30
    alarm_email    = "alerts@yourcompany.com"
    slack_webhook_url = ""
  }
}

# DNS Variables
variable "domain_name" {
  description = "Base domain name for the application"
  type        = string
  default     = "yourcompany.com"
}

variable "create_dns_records" {
  description = "Whether to create DNS records"
  type        = bool
  default     = true
}

# Cost Tracking Variables
variable "cost_center" {
  description = "Cost center for resource tagging"
  type        = string
  default     = "ai-platform"
}

variable "budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 500
}