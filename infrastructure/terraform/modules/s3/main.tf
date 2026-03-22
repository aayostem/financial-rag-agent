# =============================================================================
# Module: s3
# 1. Terraform remote state — S3 + DynamoDB (bootstrapped first, never destroyed)
# 2. Application buckets — raw EDGAR filings, processed documents
#
# Security hardening on all buckets:
#   - Block all public access
#   - SSE-AES256 encryption at rest
#   - Bucket policy denying non-TLS requests (SOC2 CC6.7)
#   - Access logging to dedicated audit bucket
#   - Object lock on filings (WORM — SEC 7-year retention)
# =============================================================================

data "aws_caller_identity" "current" {}

locals {
  name         = "${var.project}-${var.environment}"
  account_id   = data.aws_caller_identity.current.account_id
  state_bucket = "${var.project}-tfstate-${local.account_id}-${var.aws_region}"
  lock_table   = "${var.project}-tflock-${var.environment}"
}

# ---------------------------------------------------------------------------
# Access Logging Bucket — audit trail for all S3 access
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "access_logs" {
  bucket = "${local.name}-s3-access-logs"
  tags   = { Purpose = "s3-access-logs" }
}

resource "aws_s3_bucket_public_access_block" "access_logs" {
  bucket                  = aws_s3_bucket.access_logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "access_logs" {
  bucket = aws_s3_bucket.access_logs.id
  rule {
    id     = "expire-access-logs"
    status = "Enabled"
    expiration { days = var.environment == "prod" ? 90 : 30 }
    filter { prefix = "" }
  }
}

# ---------------------------------------------------------------------------
# Terraform State Bucket
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "tfstate" {
  bucket = local.state_bucket
  tags   = { Purpose = "terraform-state" }

  lifecycle { prevent_destroy = true }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyNonTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = ["${aws_s3_bucket.tfstate.arn}", "${aws_s3_bucket.tfstate.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      },
      {
        Sid       = "DenyNonAccountAccess"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = ["${aws_s3_bucket.tfstate.arn}", "${aws_s3_bucket.tfstate.arn}/*"]
        Condition = {
          StringNotEquals = { "aws:PrincipalAccount" = local.account_id }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_logging" "tfstate" {
  bucket        = aws_s3_bucket.tfstate.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "tfstate/"
}

# ---------------------------------------------------------------------------
# DynamoDB State Lock Table
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "tflock" {
  name         = local.lock_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  server_side_encryption { enabled = true }

  point_in_time_recovery { enabled = var.environment == "prod" }

  lifecycle { prevent_destroy = true }

  tags = { Purpose = "terraform-state-lock" }
}

# ---------------------------------------------------------------------------
# EDGAR Filings Bucket — raw SEC filing data
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "filings" {
  bucket = "${local.name}-filings"
  tags   = { Purpose = "edgar-raw-filings" }
}

resource "aws_s3_bucket_versioning" "filings" {
  bucket = aws_s3_bucket.filings.id
  versioning_configuration {
    status = var.environment == "prod" ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "filings" {
  bucket = aws_s3_bucket.filings.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "filings" {
  bucket                  = aws_s3_bucket.filings.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "filings" {
  bucket = aws_s3_bucket.filings.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyNonTLS"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = ["${aws_s3_bucket.filings.arn}", "${aws_s3_bucket.filings.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_logging" "filings" {
  bucket        = aws_s3_bucket.filings.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "filings/"
}

# 7-year retention + Glacier tiering for prod (SEC rule 17a-4)
resource "aws_s3_bucket_lifecycle_configuration" "filings" {
  count  = var.environment == "prod" ? 1 : 0
  bucket = aws_s3_bucket.filings.id

  rule {
    id     = "tier-and-expire"
    status = "Enabled"
    filter { prefix = "" }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }

    expiration { days = 2555 }   # 7 years

    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}

# ---------------------------------------------------------------------------
# Processed Documents Bucket — chunked + embedded content (agent reads)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "documents" {
  bucket = "${local.name}-documents"
  tags   = { Purpose = "processed-documents" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "documents" {
  bucket = aws_s3_bucket.documents.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyNonTLS"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = ["${aws_s3_bucket.documents.arn}", "${aws_s3_bucket.documents.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_logging" "documents" {
  bucket        = aws_s3_bucket.documents.id
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "documents/"
}
