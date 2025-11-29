# Random password for database
resource "random_password" "database" {
  length  = 16
  special = false
}

# Database Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.name_prefix}-db-subnet-group"
  subnet_ids = var.subnet_ids

  tags = var.tags
}

# PostgreSQL Database Instance
resource "aws_db_instance" "main" {
  identifier = "${var.name_prefix}-db"

  engine         = "postgres"
  engine_version = var.database_config.engine_version
  instance_class = var.database_config.instance_class

  allocated_storage     = var.database_config.allocated_storage
  max_allocated_storage = var.database_config.allocated_storage * 2

  db_name  = "financial_rag"
  username = "financial_user"
  password = random_password.database.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = var.security_group_ids

  multi_az               = var.database_config.multi_az
  storage_encrypted      = true
  backup_retention_period = var.database_config.backup_retention
  skip_final_snapshot    = var.environment == "prod" ? false : true
  final_snapshot_identifier = var.environment == "prod" ? "${var.name_prefix}-final-snapshot" : null

  performance_insights_enabled = true
  monitoring_interval          = 60

  apply_immediately = true

  tags = var.tags
}

# Database Parameter Group
resource "aws_db_parameter_group" "main" {
  name   = "${var.name_prefix}-db-params"
  family = "postgres15"

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  tags = var.tags
}

# Store database password in SSM Parameter Store
resource "aws_ssm_parameter" "database_password" {
  name  = "/${var.name_prefix}/database/password"
  type  = "SecureString"
  value = random_password.database.result

  tags = var.tags
}

# Database backup schedule
resource "aws_backup_plan" "database" {
  count = var.environment == "prod" ? 1 : 0

  name = "${var.name_prefix}-db-backup-plan"

  rule {
    rule_name         = "daily-backup"
    target_vault_name = aws_backup_vault.main[0].name
    schedule          = "cron(0 2 * * ? *)"

    lifecycle {
      delete_after = var.database_config.backup_retention
    }

    copy_action {
      destination_vault_arn = aws_backup_vault.secondary[0].arn
      lifecycle {
        delete_after = var.database_config.backup_retention * 2
      }
    }
  }

  tags = var.tags
}

resource "aws_backup_vault" "main" {
  count = var.environment == "prod" ? 1 : 0

  name = "${var.name_prefix}-backup-vault"
  tags = var.tags
}

resource "aws_backup_vault" "secondary" {
  count = var.environment == "prod" ? 1 : 0

  name = "${var.name_prefix}-secondary-vault"
  tags = var.tags
}

resource "aws_backup_selection" "database" {
  count = var.environment == "prod" ? 1 : 0

  name         = "${var.name_prefix}-db-selection"
  plan_id      = aws_backup_plan.database[0].id
  iam_role_arn = aws_iam_role.backup[0].arn

  resources = [
    aws_db_instance.main.arn
  ]
}