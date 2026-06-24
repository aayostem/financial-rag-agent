# =============================================================================
# policies/rego/terraform_policy.rego
# Conftest policies for Terraform plan JSON.
# Run in CI: terraform plan -out plan.out && terraform show -json plan.out | conftest test -
# Catches: public S3, unencrypted RDS, missing KMS, public EKS endpoints in prod.
# =============================================================================
package financial_rag.terraform

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# S3 — no public buckets
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_public_access_block"
    resource.values.block_public_acls != true
    msg := sprintf("S3 bucket '%v' does not block public ACLs — all buckets must block public access.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_public_access_block"
    resource.values.restrict_public_buckets != true
    msg := sprintf("S3 bucket '%v' does not restrict public buckets.", [resource.name])
}

# ---------------------------------------------------------------------------
# S3 — encryption required
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket"
    not bucket_has_encryption(resource.name)
    msg := sprintf("S3 bucket '%v' has no server-side encryption configuration.", [resource.name])
}

bucket_has_encryption(bucket_name) if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_s3_bucket_server_side_encryption_configuration"
    contains(resource.values.bucket, bucket_name)
}

# ---------------------------------------------------------------------------
# RDS — encryption, deletion protection, no public access
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    not resource.values.storage_encrypted == true
    msg := sprintf("RDS instance '%v' storage_encrypted must be true.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    resource.values.publicly_accessible == true
    msg := sprintf("RDS instance '%v' must not be publicly accessible.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_db_instance"
    resource.values.backup_retention_period == 0
    msg := sprintf("RDS instance '%v' has no backup retention — minimum 1 day required.", [resource.name])
}

# ---------------------------------------------------------------------------
# EKS — private endpoint in prod, secrets encryption, audit logs
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    resource.values.vpc_config[_].endpoint_public_access == true
    contains(resource.name, "prod")
    msg := sprintf("EKS cluster '%v' has public API endpoint enabled in prod — must be private-only.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    not resource.values.encryption_config
    msg := sprintf("EKS cluster '%v' has no encryption_config — Kubernetes Secrets must be encrypted at rest with KMS.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_eks_cluster"
    log_types := resource.values.enabled_cluster_log_types
    required := {"api", "audit", "authenticator", "controllerManager", "scheduler"}
    missing := required - {l | l := log_types[_]}
    count(missing) > 0
    msg := sprintf("EKS cluster '%v' missing control plane log types: %v", [resource.name, missing])
}

# ---------------------------------------------------------------------------
# ElastiCache — encryption in transit and at rest required
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_elasticache_replication_group"
    not resource.values.at_rest_encryption_enabled == true
    msg := sprintf("ElastiCache replication group '%v' at_rest_encryption_enabled must be true.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_elasticache_replication_group"
    not resource.values.transit_encryption_enabled == true
    msg := sprintf("ElastiCache replication group '%v' transit_encryption_enabled must be true.", [resource.name])
}

# ---------------------------------------------------------------------------
# IAM — no wildcard Resource in policies, no * Action
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_iam_role_policy"
    policy := json.unmarshal(resource.values.policy)
    statement := policy.Statement[_]
    statement.Effect == "Allow"
    statement.Resource == "*"
    statement.Action == "*"
    msg := sprintf("IAM policy '%v' has Action=* with Resource=* — overly permissive.", [resource.name])
}

# ---------------------------------------------------------------------------
# Security Groups — no 0.0.0.0/0 inbound on sensitive ports
# ---------------------------------------------------------------------------
deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_security_group_rule"
    resource.values.type == "ingress"
    resource.values.cidr_blocks[_] == "0.0.0.0/0"
    resource.values.from_port <= 22
    resource.values.to_port >= 22
    msg := sprintf("Security group rule '%v' allows SSH (22) from 0.0.0.0/0.", [resource.name])
}

deny contains msg if {
    resource := input.planned_values.root_module.resources[_]
    resource.type == "aws_security_group_rule"
    resource.values.type == "ingress"
    resource.values.cidr_blocks[_] == "0.0.0.0/0"
    resource.values.from_port <= 5432
    resource.values.to_port >= 5432
    msg := sprintf("Security group rule '%v' allows PostgreSQL (5432) from 0.0.0.0/0.", [resource.name])
}
