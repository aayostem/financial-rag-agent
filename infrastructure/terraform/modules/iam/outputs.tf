output "api_role_arn"       { value = aws_iam_role.api.arn }
output "agent_role_arn"     { value = aws_iam_role.agent.arn }
output "ingestion_role_arn" { value = aws_iam_role.ingestion.arn }
output "api_role_name"      { value = aws_iam_role.api.name }
output "agent_role_name"    { value = aws_iam_role.agent.name }
output "ingestion_role_name" { value = aws_iam_role.ingestion.name }
