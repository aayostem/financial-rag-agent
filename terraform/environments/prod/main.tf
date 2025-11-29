module "financial_rag" {
  source = "../../"

  # Project Configuration
  project_name = "financial-rag-agent"
  environment  = "prod"
  owner        = "ai-engineering-team"

  # AWS Configuration
  aws_region = "us-west-2"
  vpc_cidr   = "10.0.0.0/16"

  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]

  # Kubernetes Configuration
  cluster_name    = "financial-rag-prod"
  cluster_version = "1.28"

  node_groups = {
    main = {
      instance_types = ["m5.large", "m5a.large"]
      capacity_type  = "ON_DEMAND"
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      disk_size      = 50
    }
    spot = {
      instance_types = ["m5.large", "m5a.large", "m4.large"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 5
      desired_size   = 1
      disk_size      = 50
    }
  }

  # Database Configuration
  database_config = {
    instance_class    = "db.r5.large"
    allocated_storage = 100
    engine_version    = "15.2"
    multi_az          = true
    backup_retention  = 14
  }

  # Monitoring Configuration
  monitoring_config = {
    enabled           = true
    retention_days    = 30
    alarm_email       = "prod-alerts@yourcompany.com"
    slack_webhook_url = "https://hooks.slack.com/services/your/webhook/url"
  }

  # DNS Configuration
  domain_name        = "yourcompany.com"
  create_dns_records = true

  # Cost Tracking
  cost_center   = "ai-platform-prod"
  budget_amount = 2000
}