module "financial_rag" {
  source = "../../"

  # Project Configuration
  project_name = "financial-rag-agent"
  environment  = "dev"
  owner        = "ai-engineering-team"

  # AWS Configuration
  aws_region = "us-west-2"
  vpc_cidr   = "10.1.0.0/16"

  # Kubernetes Configuration
  cluster_name    = "financial-rag-dev"
  cluster_version = "1.28"

  node_groups = {
    main = {
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 3
      desired_size   = 1
      disk_size      = 20
    }
  }

  # Database Configuration
  database_config = {
    instance_class    = "db.t3.small"
    allocated_storage = 20
    engine_version    = "15.2"
    multi_az          = false
    backup_retention  = 3
  }

  # Monitoring Configuration
  monitoring_config = {
    enabled        = true
    retention_days = 7
    alarm_email    = "dev-alerts@yourcompany.com"
  }

  # DNS Configuration
  domain_name        = "dev.yourcompany.com"
  create_dns_records = true

  # Cost Tracking
  cost_center   = "ai-platform-dev"
  budget_amount = 200
}