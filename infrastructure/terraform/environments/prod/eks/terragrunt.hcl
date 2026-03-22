include "root" {
  path = find_in_parent_folders()
}

include "env" {
  path   = find_in_parent_folders("env.hcl")
  expose = true
}

terraform {
  source = "../../../modules/eks"
}

dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    vpc_id             = "vpc-00000000"
    private_subnet_ids = ["subnet-00000001", "subnet-00000002", "subnet-00000003"]
    node_security_group_id = "sg-00000000"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

inputs = {
  vpc_id                    = dependency.vpc.outputs.vpc_id
  private_subnet_ids        = dependency.vpc.outputs.private_subnet_ids
  cluster_version           = include.env.locals.eks_cluster_version
  node_groups               = include.env.locals.eks_node_groups
  terraform_admin_role_name = "TerraformDeployRole"
}
