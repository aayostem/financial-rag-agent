include "root" {
  path = find_in_parent_folders()
}

include "env" {
  path   = find_in_parent_folders("env.hcl")
  expose = true
}

terraform {
  source = "../../../modules/vpc"
}

inputs = {
  vpc_cidr           = include.env.locals.vpc_cidr
  private_subnets    = include.env.locals.private_subnets
  public_subnets     = include.env.locals.public_subnets
  availability_zones = include.env.locals.availability_zones
}
