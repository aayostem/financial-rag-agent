# =============================================================================
# Module: elasticache
# ElastiCache Redis 7 replication group. Replaces the in-cluster Redis
# StatefulSet with a managed service: automatic failover, at-rest + in-transit
# encryption, and no PVC management.
# Auth token stored in Secrets Manager.
# =============================================================================

locals {
  name = "${var.project}-${var.environment}-redis"
}

# ---------------------------------------------------------------------------
# Auth token — Redis AUTH, stored in Secrets Manager
# ---------------------------------------------------------------------------
resource "random_password" "redis_auth" {
  length  = 32
  special = false   # Redis AUTH token cannot contain special chars
}

resource "aws_secretsmanager_secret" "redis" {
  name                    = "${var.project}/${var.environment}/redis/auth-token"
  description             = "ElastiCache Redis AUTH token for ${local.name}"
  recovery_window_in_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "redis" {
  secret_id     = aws_secretsmanager_secret.redis.id
  secret_string = jsonencode({
    auth_token = random_password.redis_auth.result
    host       = aws_elasticache_replication_group.main.primary_endpoint_address
    port       = 6379
    url        = "rediss://:${random_password.redis_auth.result}@${aws_elasticache_replication_group.main.primary_endpoint_address}:6379"
  })
}

# ---------------------------------------------------------------------------
# Security Group — EKS nodes only
# ---------------------------------------------------------------------------
resource "aws_security_group" "redis" {
  name        = "${local.name}-sg"
  description = "ElastiCache Redis — allow EKS nodes only"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Redis from EKS nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.eks_node_security_group]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-sg" }
}

# ---------------------------------------------------------------------------
# Subnet Group
# ---------------------------------------------------------------------------
resource "aws_elasticache_subnet_group" "main" {
  name        = local.name
  subnet_ids  = var.private_subnet_ids
  description = "Redis subnet group — ${var.environment}"
}

# ---------------------------------------------------------------------------
# Parameter Group — tune for embedding cache workload
# ---------------------------------------------------------------------------
resource "aws_elasticache_parameter_group" "main" {
  name        = "${local.name}-params"
  family      = "redis7"
  description = "RAG embedding cache tuning"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"   # evict LRU keys — keeps hottest embeddings in memory
  }
  parameter {
    name  = "activerehashing"
    value = "yes"
  }
  parameter {
    name  = "lazyfree-lazy-eviction"
    value = "yes"           # async eviction — avoids blocking on large key deletes
  }
}

# ---------------------------------------------------------------------------
# Replication Group
# ---------------------------------------------------------------------------
resource "aws_elasticache_replication_group" "main" {
  replication_group_id = local.name
  description          = "Financial RAG embedding cache — ${var.environment}"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.node_type
  num_cache_clusters   = var.num_cache_nodes
  port                 = 6379

  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]
  parameter_group_name       = aws_elasticache_parameter_group.main.name

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth.result

  automatic_failover_enabled = var.num_cache_nodes > 1
  multi_az_enabled           = var.multi_az

  snapshot_retention_limit = var.environment == "prod" ? 7 : 1
  snapshot_window          = "04:00-05:00"
  maintenance_window       = "mon:05:00-mon:06:00"

  apply_immediately = var.environment != "prod"

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
}

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/aws/elasticache/${local.name}/slow-log"
  retention_in_days = var.environment == "prod" ? 30 : 7
}

# ---------------------------------------------------------------------------
# CloudWatch alarms — prod only
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  count               = var.environment == "prod" ? 1 : 0
  alarm_name          = "${local.name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 75
  alarm_description   = "Redis CPU > 75%"
  dimensions          = { ReplicationGroupId = aws_elasticache_replication_group.main.id }
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  count               = var.environment == "prod" ? 1 : 0
  alarm_name          = "${local.name}-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Redis memory > 80%"
  dimensions          = { ReplicationGroupId = aws_elasticache_replication_group.main.id }
}
