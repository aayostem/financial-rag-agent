# Local values for consistent naming
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
    CostCenter  = var.cost_center
  }
  
  # S3 bucket names
  tfstate_bucket    = "${var.project_name}-tfstate-${var.environment}"
  backups_bucket    = "${var.project_name}-backups-${var.environment}"
  logs_bucket       = "${var.project_name}-logs-${var.environment}"
}

# Random ID for unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# S3 Bucket for Terraform State
module "terraform_state" {
  source = "./modules/storage"

  bucket_name = local.tfstate_bucket
  versioning  = true
  encryption  = true
  tags        = local.common_tags
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  name_prefix       = local.name_prefix
  vpc_cidr          = var.vpc_cidr
  availability_zones = var.availability_zones
  environment       = var.environment
  tags              = local.common_tags
}

# Kubernetes Cluster Module
module "kubernetes" {
  source = "./modules/kubernetes"

  cluster_name    = "${local.name_prefix}-cluster"
  cluster_version = var.cluster_version
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
  node_groups     = var.node_groups
  environment     = var.environment
  tags            = local.common_tags

  depends_on = [module.networking]
}

# Database Module
module "database" {
  source = "./modules/database"

  name_prefix      = local.name_prefix
  vpc_id           = module.networking.vpc_id
  subnet_ids       = module.networking.private_subnet_ids
  security_group_ids = [module.networking.database_security_group_id]
  database_config  = var.database_config
  environment      = var.environment
  tags             = local.common_tags

  depends_on = [module.networking, module.kubernetes]
}

# Storage Module for Backups
module "backup_storage" {
  source = "./modules/storage"

  bucket_name = local.backups_bucket
  versioning  = true
  encryption  = true
  lifecycle_rules = var.storage_config.s3_lifecycle_rules
  tags        = local.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"

  count = var.monitoring_config.enabled ? 1 : 0

  name_prefix          = local.name_prefix
  cluster_name         = module.kubernetes.cluster_name
  environment          = var.environment
  alarm_email          = var.monitoring_config.alarm_email
  slack_webhook_url    = var.monitoring_config.slack_webhook_url
  retention_days       = var.monitoring_config.retention_days
  tags                 = local.common_tags

  depends_on = [module.kubernetes]
}

# DNS Module
module "dns" {
  source = "./modules/dns"

  count = var.create_dns_records ? 1 : 0

  domain_name        = var.domain_name
  environment        = var.environment
  load_balancer_dns  = module.kubernetes.load_balancer_dns
  tags               = local.common_tags

  depends_on = [module.kubernetes]
}

# Budget Alert
module "budget" {
  source = "./modules/monitoring/budget"

  budget_amount    = var.budget_amount
  budget_name      = "${local.name_prefix}-budget"
  email_addresses  = [var.monitoring_config.alarm_email]
  time_unit        = "MONTHLY"

  tags = local.common_tags
}

# Outputs
output "cluster_info" {
  description = "Kubernetes cluster information"
  value = {
    name       = module.kubernetes.cluster_name
    endpoint   = module.kubernetes.cluster_endpoint
    version    = module.kubernetes.cluster_version
    kubeconfig = module.kubernetes.kubeconfig_filename
  }
}

output "database_info" {
  description = "Database connection information"
  value = {
    endpoint = module.database.endpoint
    port     = module.database.port
    name     = module.database.database_name
    username = module.database.username
  }
  sensitive = true
}

output "network_info" {
  description = "Network information"
  value = {
    vpc_id     = module.networking.vpc_id
    vpc_cidr   = module.networking.vpc_cidr
    public_subnets  = module.networking.public_subnet_ids
    private_subnets = module.networking.private_subnet_ids
  }
}

output "storage_info" {
  description = "Storage information"
  value = {
    tfstate_bucket = module.terraform_state.bucket_name
    backups_bucket = module.backup_storage.bucket_name
  }
}

output "dns_info" {
  description = "DNS information"
  value = module.dns[*].dns_records
}