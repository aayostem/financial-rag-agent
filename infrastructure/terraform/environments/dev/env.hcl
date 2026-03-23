locals {
  aws_region = "us-east-1"
  account_id = "637423656824"

  vpc_cidr            = "10.2.0.0/16"
  private_subnets     = ["10.2.1.0/24", "10.2.2.0/24"]
  public_subnets      = ["10.2.101.0/24", "10.2.102.0/24"]
  availability_zones  = ["us-east-1a", "us-east-1b"]

  eks_cluster_version = "1.29"
  eks_node_groups = {
    application = {
      instance_types = ["t3.large"]
      min_size       = 1
      desired_size   = 2
      max_size       = 4
      disk_size      = 30
      labels         = { role = "application" }
    }
    ingestion = {
      instance_types = ["t3.medium"]
      min_size       = 0
      desired_size   = 1
      max_size       = 2
      disk_size      = 20
      labels         = { role = "ingestion" }
    }
  }

  rds_instance_class        = "db.t3.medium"
  rds_allocated_storage     = 20
  rds_backup_retention_days = 1
  rds_multi_az              = false

  elasticache_node_type       = "cache.t3.micro"
  elasticache_num_cache_nodes = 1
  elasticache_multi_az        = false

  ecr_image_retention_count = 5
}
