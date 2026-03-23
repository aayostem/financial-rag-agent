variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "private_subnets" {
  description = "List of private subnet CIDRs"
  type        = list(string)
}

variable "public_subnets" {
  description = "List of public subnet CIDRs"
  type        = list(string)
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "cluster_name" {
  description = "EKS cluster name — used for subnet tags"
  type        = string
  default     = ""
}

variable "environment" {
  description = "Environment name (dev/staging/prod)"
  type        = string
}

variable "project" {
  description = "Project name for tagging"
  type        = string
  default     = "financial-rag"
}