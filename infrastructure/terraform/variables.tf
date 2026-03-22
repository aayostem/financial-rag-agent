variable "project"           { type = string }
variable "environment"       { type = string }
variable "aws_region"        { type = string }
variable "vpc_cidr"          { type = string }
variable "private_subnets"   { type = list(string) }
variable "public_subnets"    { type = list(string) }
variable "availability_zones" { type = list(string) }
