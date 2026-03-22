output "endpoint"        { value = aws_db_instance.main.address }
output "port"            { value = aws_db_instance.main.port }
output "secret_arn"      { value = aws_secretsmanager_secret.rds.arn }
output "secret_name"     { value = aws_secretsmanager_secret.rds.name }
output "security_group"  { value = aws_security_group.rds.id }
output "db_name"         { value = aws_db_instance.main.db_name }
