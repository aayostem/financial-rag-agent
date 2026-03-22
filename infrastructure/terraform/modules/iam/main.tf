# =============================================================================
# Module: iam
# IRSA roles for all application service accounts.
# Each role uses a dual StringEquals condition on :sub AND :aud —
# the :aud check is required to prevent cross-account token acceptance.
#
# Roles created:
#   api       — Secrets Manager read, CloudWatch logs write
#   agent     — Secrets Manager read, S3 read (docs), LLM secret read
#   ingestion — Secrets Manager read, S3 write (filings), ECR pull
# =============================================================================

locals {
  name        = "${var.project}-${var.environment}"
  oidc_issuer = replace(var.cluster_oidc_issuer_url, "https://", "")
}

# ---------------------------------------------------------------------------
# Reusable assume-role policy factory
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "irsa_assume" {
  for_each = toset(["api", "agent", "ingestion"])

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer}:sub"
      values   = ["system:serviceaccount:${var.namespace}:${local.name}-${each.key}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# API Role
# ---------------------------------------------------------------------------
resource "aws_iam_role" "api" {
  name               = "${local.name}-api-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume["api"].json
  tags               = { Component = "api" }
}

resource "aws_iam_role_policy" "api" {
  name = "api-least-privilege"
  role = aws_iam_role.api.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAppSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = [
          var.rds_secret_arn,
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/redis/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/llm/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/config/*",
        ]
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:ListMetrics",
        ]
        Resource = "*"
        Condition = {
          StringEquals = { "cloudwatch:namespace" = "${var.project}/${var.environment}" }
        }
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/eks/${local.name}*:*"
      },
      {
        Sid    = "XRayTracing"
        Effect = "Allow"
        Action = ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules"]
        Resource = "*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Agent Role
# ---------------------------------------------------------------------------
resource "aws_iam_role" "agent" {
  name               = "${local.name}-agent-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume["agent"].json
  tags               = { Component = "agent" }
}

resource "aws_iam_role_policy" "agent" {
  name = "agent-least-privilege"
  role = aws_iam_role.agent.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAppSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = [
          var.rds_secret_arn,
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/redis/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/llm/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/embeddings/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/config/*",
        ]
      },
      {
        Sid    = "S3ReadDocuments"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project}-${var.environment}-documents",
          "arn:aws:s3:::${var.project}-${var.environment}-documents/*",
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/eks/${local.name}*:*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Ingestion Role
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ingestion" {
  name               = "${local.name}-ingestion-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume["ingestion"].json
  tags               = { Component = "ingestion" }
}

resource "aws_iam_role_policy" "ingestion" {
  name = "ingestion-least-privilege"
  role = aws_iam_role.ingestion.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAppSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = [
          var.rds_secret_arn,
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/redis/*",
          "arn:aws:secretsmanager:${var.aws_region}:${var.account_id}:secret:${var.project}/${var.environment}/edgar/*",
        ]
      },
      {
        Sid    = "S3WriteFilings"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [
          "arn:aws:s3:::${var.project}-${var.environment}-filings",
          "arn:aws:s3:::${var.project}-${var.environment}-filings/*",
        ]
      },
      {
        # ECR pull — ingestion builds its own image layers at runtime for plugins
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/eks/${local.name}*:*"
      }
    ]
  })
}
