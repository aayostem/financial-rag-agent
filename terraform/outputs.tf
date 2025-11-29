# Kubernetes Outputs
output "kubeconfig" {
  description = "Kubectl config file contents"
  value       = module.kubernetes.kubeconfig
  sensitive   = true
}

output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = module.kubernetes.cluster_endpoint
}

output "cluster_ca_certificate" {
  description = "Kubernetes cluster CA certificate"
  value       = module.kubernetes.cluster_certificate_authority_data
  sensitive   = true
}

# Database Outputs
output "database_endpoint" {
  description = "Database endpoint"
  value       = module.database.endpoint
}

output "database_connection_string" {
  description = "Database connection string (without password)"
  value       = "postgresql://${module.database.username}@${module.database.endpoint}:${module.database.port}/${module.database.database_name}"
  sensitive   = true
}

# Network Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.networking.public_subnet_ids
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.networking.private_subnet_ids
}

# Storage Outputs
output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    tfstate = module.terraform_state.bucket_name
    backups = module.backup_storage.bucket_name
    logs    = module.monitoring[0].logs_bucket_name
  }
}

# DNS Outputs
output "application_url" {
  description = "Application URL"
  value       = var.create_dns_records ? module.dns[0].application_url : null
}

# Monitoring Outputs
output "monitoring_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = var.monitoring_config.enabled ? module.monitoring[0].dashboard_url : null
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = var.monitoring_config.enabled ? module.monitoring[0].grafana_url : null
}

# Instructions Output
output "deployment_instructions" {
  description = "Instructions for deploying the application"
  value = <<-EOT

  Financial RAG Agent Infrastructure Deployment Complete!

  Next Steps:
  1. Configure kubectl: 
     export KUBECONFIG=${module.kubernetes.kubeconfig_filename}

  2. Deploy the application using Helm:
     cd helm/
     ./scripts/deploy-helm.sh -e ${var.environment}

  3. Access the application:
     ${var.create_dns_records ? "URL: https://${module.dns[0].application_url}" : "Use kubectl port-forward to access the application"}

  4. Monitor the deployment:
     ${var.monitoring_config.enabled ? "Dashboard: ${module.monitoring[0].dashboard_url}" : "Monitoring is disabled"}

  Database Connection:
  - Endpoint: ${module.database.endpoint}
  - Database: ${module.database.database_name}
  - Username: ${module.database.username}

  Remember to:
  - Secure your database password
  - Configure API keys in Kubernetes secrets
  - Set up proper backup schedules

  EOT
}