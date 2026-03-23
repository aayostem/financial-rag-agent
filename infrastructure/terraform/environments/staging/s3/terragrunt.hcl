include "root" {
  path = find_in_parent_folders()
}

include "env" {
  path   = find_in_parent_folders("env.hcl")
  expose = true
}

terraform {
  source = "../../../modules/s3"
}

# Bootstrap note: apply this module FIRST with a local backend.
# After the state bucket exists, all subsequent applies use it as remote state.
# cd environments/prod/s3 && terragrunt apply
inputs = {}
