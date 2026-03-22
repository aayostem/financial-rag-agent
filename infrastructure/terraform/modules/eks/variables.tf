variable "project"                   { type = string }
variable "environment"               { type = string }
variable "aws_region"                { type = string }
variable "vpc_id"                    { type = string }
variable "private_subnet_ids"        { type = list(string) }
variable "cluster_version"           { type = string }
variable "terraform_admin_role_name" {
  type        = string
  default     = "TerraformDeployRole"
  description = "IAM role name granted EKS cluster admin via Access Entry"
}

variable "node_groups" {
  type = map(object({
    instance_types = list(string)
    min_size       = number
    desired_size   = number
    max_size       = number
    disk_size      = number
    labels         = map(string)
  }))
}
