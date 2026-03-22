# =============================================================================
# Root terragrunt.hcl
# Inherited by every environment module via find_in_parent_folders().
# Defines: remote state backend, provider generation, common inputs.
#
# Usage:
#   cd environments/prod/vpc && terragrunt plan
#   cd environments/prod     && terragrunt run-all apply
# =============================================================================

locals {
  env_vars   = read_terragrunt_config(find_in_parent_folders("env.hcl"))
  aws_region = local.env_vars.locals.aws_region
  account_id = local.env_vars.locals.account_id
  project    = "financial-rag"
  env        = basename(dirname(find_in_parent_folders("env.hcl")))
}

# ---------------------------------------------------------------------------
# Remote State — S3 + DynamoDB
# Bootstrap: cd environments/prod/s3 && terragrunt apply (local backend first)
# ---------------------------------------------------------------------------
remote_state {
  backend = "s3"
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
  config = {
    bucket         = "${local.project}-tfstate-${local.account_id}-${local.aws_region}"
    key            = "${local.env}/${path_relative_to_include()}/terraform.tfstate"
    region         = local.aws_region
    encrypt        = true
    dynamodb_table = "${local.project}-tflock-${local.env}"

    # Prevent accidental state deletion
    skip_bucket_versioning         = false
    skip_bucket_ssencryption       = false
    skip_bucket_public_access_blocking = false
  }
}

# ---------------------------------------------------------------------------
# Provider — generated into each module directory at plan/apply time
# ---------------------------------------------------------------------------
generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<-EOF
    terraform {
      required_version = ">= 1.7.0"
      required_providers {
        aws = {
          source  = "hashicorp/aws"
          version = "~> 5.40"
        }
        kubernetes = {
          source  = "hashicorp/kubernetes"
          version = "~> 2.27"
        }
        helm = {
          source  = "hashicorp/helm"
          version = "~> 2.13"
        }
        tls = {
          source  = "hashicorp/tls"
          version = "~> 4.0"
        }
        random = {
          source  = "hashicorp/random"
          version = "~> 3.6"
        }
      }
    }

    provider "aws" {
      region = "${local.aws_region}"

      default_tags {
        tags = {
          Project     = "${local.project}"
          Environment = "${local.env}"
          ManagedBy   = "terraform"
          Owner       = "cloudfrugal"
          Repo        = "aayostem/financial-rag-agent"
        }
      }
    }
  EOF
}

# ---------------------------------------------------------------------------
# Common inputs — available in every module without re-declaring
# ---------------------------------------------------------------------------
inputs = {
  project     = local.project
  environment = local.env
  aws_region  = local.aws_region
  account_id  = local.account_id
}
