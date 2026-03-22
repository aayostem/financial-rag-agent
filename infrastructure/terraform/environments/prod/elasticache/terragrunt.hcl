include "root" {
  path = find_in_parent_folders()
}

include "env" {
  path   = find_in_parent_folders("env.hcl")
  expose = true
}

terraform {
  source = "../../../modules/elasticache"
}

dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    vpc_id             = "vpc-00000000"
    private_subnet_ids = ["subnet-00000001", "subnet-00000002"]
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

dependency "eks" {
  config_path = "../eks"
  mock_outputs = {
    node_security_group_id = "sg-00000000"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

inputs = {
  vpc_id                 = dependency.vpc.outputs.vpc_id
  private_subnet_ids     = dependency.vpc.outputs.private_subnet_ids
  eks_node_security_group = dependency.eks.outputs.node_security_group_id
  node_type              = include.env.locals.elasticache_node_type
  num_cache_nodes        = include.env.locals.elasticache_num_cache_nodes
  multi_az               = include.env.locals.elasticache_multi_az
}
