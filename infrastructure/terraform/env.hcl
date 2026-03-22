locals {
  aws_region = "us-east-1"
  account_id = "123456789012"          # replace with your AWS account ID

  # CIDR blocks
  vpc_cidr            = "10.0.0.0/16"
  private_subnets     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets      = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  availability_zones  = ["us-east-1a", "us-east-1b", "us-east-1c"]

  # EKS
  eks_cluster_version = "1.29"
  eks_node_groups = {
    application = {
      instance_types = ["m5.2xlarge"]
      min_size       = 3
      desired_size   = 5
      max_size       = 20
      disk_size      = 100
      labels         = { role = "application" }
    }
    ingestion = {
      instance_types = ["c5.xlarge"]
      min_size       = 1
      desired_size   = 2
      max_size       = 5
      disk_size      = 50
      labels         = { role = "ingestion" }
    }
  }

  # RDS Aurora PostgreSQL (pgvector)
  rds_instance_class        = "db.r6g.2xlarge"
  rds_allocated_storage     = 500
  rds_backup_retention_days = 14
  rds_multi_az              = true

  # ElastiCache Redis
  elasticache_node_type       = "cache.r6g.xlarge"
  elasticache_num_cache_nodes = 2
  elasticache_multi_az        = true

  # ECR
  ecr_image_retention_count = 30
}
