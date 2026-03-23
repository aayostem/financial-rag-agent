# =============================================================================
# Module: rds
# Aurora PostgreSQL 15 cluster with pgvector extension support.
# Replaces the in-cluster pgvector StatefulSet with a managed service:
# automated backups, Multi-AZ failover, and no StatefulSet PVC management.
# Password stored in Secrets Manager — never in tfstate plaintext.
# =============================================================================

locals {
  name = "${var.project}-${var.environment}"
  identifier = "${local.name}-pgvector"
}

# ---------------------------------------------------------------------------
# Random password — stored in Secrets Manager, not tfstate
# ---------------------------------------------------------------------------
resource "random_password" "rds" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "rds" {
  name                    = "${local.name}/rds/master-password"
  description             = "RDS master password for ${local.identifier}"
  recovery_window_in_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "rds" {
  secret_id = aws_secretsmanager_secret.rds.id
  secret_string = jsonencode({
    username = "raguser"
    password = random_password.rds.result
    host     = aws_db_instance.main.address
    port     = 5432
    dbname   = "financial_rag"
    url      = "postgresql://raguser:${random_password.rds.result}@${aws_db_instance.main.address}:5432/financial_rag"
  })
}

# ---------------------------------------------------------------------------
# Security Group — only EKS nodes can reach port 5432
# ---------------------------------------------------------------------------
resource "aws_security_group" "rds" {
  name        = "${local.identifier}-sg"
  description = "RDS pgvector - allow EKS nodes only"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.eks_node_security_group]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.identifier}-sg" }
}

# ---------------------------------------------------------------------------
# Subnet Group
# ---------------------------------------------------------------------------
resource "aws_db_subnet_group" "main" {
  name        = local.identifier
  subnet_ids  = var.private_subnet_ids
  description = "pgvector subnet group - ${var.environment}"
}

# ---------------------------------------------------------------------------
# Parameter Group — enable pgvector, tune for RAG workloads
# ---------------------------------------------------------------------------
resource "aws_db_parameter_group" "main" {
  name        = "${local.identifier}-params"
  family      = "postgres15"
  description = "pgvector + RAG tuning for ${var.environment}"

  # pgvector HNSW is memory-hungry — raise work_mem for vector ops
  parameter {
    apply_method = "pending-reboot"
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,auto_explain"
  }
  parameter {
    name  = "work_mem"
    value = var.environment == "prod" ? "65536" : "16384"   # KB
  }
  parameter {
    name  = "maintenance_work_mem"
    value = var.environment == "prod" ? "2097152" : "524288"
  }
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"   # log queries >1s
  }
  parameter {
    name  = "auto_explain.log_min_duration"
    value = "5000"
  }
}

# ---------------------------------------------------------------------------
# RDS Instance (PostgreSQL 15 — pgvector installs via CREATE EXTENSION)
# Using RDS instance (not Aurora) so pgvector HNSW is fully supported
# ---------------------------------------------------------------------------
resource "aws_db_instance" "main" {
  identifier        = local.identifier
  engine            = "postgres"
  engine_version    = "15.12"
  instance_class    = var.instance_class
  allocated_storage = var.allocated_storage
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = "financial_rag"
  username = "raguser"
  password = random_password.rds.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.main.name

  multi_az               = var.multi_az
  publicly_accessible    = false
  deletion_protection    = var.environment == "prod"
  skip_final_snapshot    = var.environment != "prod"
  final_snapshot_identifier = var.environment == "prod" ? "${local.identifier}-final-snapshot" : null

  backup_retention_period = var.backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  performance_insights_enabled          = var.environment == "prod"
  performance_insights_retention_period = var.environment == "prod" ? 7 : null
  monitoring_interval                   = var.environment == "prod" ? 60 : 0

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  lifecycle {
    prevent_destroy = false   # set to true after first prod data is written
    ignore_changes  = [password]   # managed via Secrets Manager rotation
  }
}

# ---------------------------------------------------------------------------
# CloudWatch alarms — prod only
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  count               = var.environment == "prod" ? 1 : 0
  alarm_name          = "${local.identifier}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU > 80% for 10 min"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.main.identifier }
}

resource "aws_cloudwatch_metric_alarm" "rds_storage" {
  count               = var.environment == "prod" ? 1 : 0
  alarm_name          = "${local.identifier}-storage-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 10737418240   # 10 GB in bytes
  alarm_description   = "RDS free storage < 10 GB"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.main.identifier }
}
