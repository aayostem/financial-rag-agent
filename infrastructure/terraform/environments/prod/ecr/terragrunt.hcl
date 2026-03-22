include "root" {
  path = find_in_parent_folders()
}

include "env" {
  path   = find_in_parent_folders("env.hcl")
  expose = true
}

terraform {
  source = "../../../modules/ecr"
}

# ECR is account-level — no VPC or EKS dependency required
inputs = {
  repositories = [
    "financial-rag-agent/api",
    "financial-rag-agent/agent",
    "financial-rag-agent/ingestion",
  ]
  image_retention_count = include.env.locals.ecr_image_retention_count
}
