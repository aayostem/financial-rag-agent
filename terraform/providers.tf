# AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

# Kubernetes Provider
provider "kubernetes" {
  host                   = module.kubernetes.cluster_endpoint
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_certificate_authority_data)
  token                  = module.kubernetes.cluster_auth_token

  # Load configuration from kubeconfig if cluster is not created yet
  config_path = var.kubeconfig_path
}

# Helm Provider
provider "helm" {
  kubernetes {
    host                   = module.kubernetes.cluster_endpoint
    cluster_ca_certificate = base64decode(module.kubernetes.cluster_certificate_authority_data)
    token                  = module.kubernetes.cluster_auth_token
    config_path            = var.kubeconfig_path
  }
}

# Kubectl Provider
provider "kubectl" {
  host                   = module.kubernetes.cluster_endpoint
  cluster_ca_certificate = base64decode(module.kubernetes.cluster_certificate_authority_data)
  token                  = module.kubernetes.cluster_auth_token
  load_config_file       = false
  config_path            = var.kubeconfig_path
}