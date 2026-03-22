locals {
  aws_region = "us-east-1"
  account_id = "123456789012"

  vpc_cidr            = "10.1.0.0/16"
  private_subnets     = ["10.1.1.0/24", "10.1.2.0/24"]
  public_subnets      = ["10.1.101.0/24", "10.1.102.0/24"]
  availability_zones  = ["us-east-1a", "us-east-1b"]

  eks_cluster_version = "1.29"
  eks_node_groups = {
    application = {
      instance_types = ["m5.xlarge"]
      min_size       = 2
      desired_size   = 3
      max_size       = 8
      disk_size      = 50
      labels         = { role = "application" }
    }
    ingestion = {
      instance_types = ["c5.large"]
      min_size       = 1
      desired_size   = 1
      max_size       = 3
      disk_size      = 30
      labels         = { role = "ingestion" }
    }
  }

  rds_instance_class        = "db.r6g.large"
  rds_allocated_storage     = 100
  rds_backup_retention_days = 7
  rds_multi_az              = false

  elasticache_node_type       = "cache.r6g.large"
  elasticache_num_cache_nodes = 1
  elasticache_multi_az        = false

  ecr_image_retention_count = 15
}
