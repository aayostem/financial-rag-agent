# =============================================================================
# Module: ecr
# One ECR repository per application image. Lifecycle policy keeps the last N
# tagged images and purges untagged layers older than 1 day — prevents
# registry bloat across dev/staging CI builds.
# Cross-account pull access wired for multi-account setups.
# =============================================================================

resource "aws_ecr_repository" "main" {
  for_each = toset(var.repositories)

  name                 = each.value
  image_tag_mutability = "MUTABLE"   # allows :latest re-tag; lock down to IMMUTABLE post-MVP

  image_scanning_configuration {
    scan_on_push = true    # Trivy runs in CI; ECR basic scanning is a free backstop
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = each.value
  }
}

# ---------------------------------------------------------------------------
# Lifecycle Policy — applied to every repository
# ---------------------------------------------------------------------------
resource "aws_ecr_lifecycle_policy" "main" {
  for_each   = aws_ecr_repository.main
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        # Keep the last N tagged images (controlled per env)
        rulePriority = 1
        description  = "Keep last ${var.image_retention_count} tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "sha-", "release-"]
          countType     = "imageCountMoreThan"
          countNumber   = var.image_retention_count
        }
        action = { type = "expire" }
      },
      {
        # Purge untagged layers fast — CI pushes untagged intermediates constantly
        rulePriority = 2
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Repository Policy — restrict pull to this account only
# (extend with cross-account ARNs for multi-account setups)
# ---------------------------------------------------------------------------
data "aws_caller_identity" "current" {}

resource "aws_ecr_repository_policy" "main" {
  for_each   = aws_ecr_repository.main
  repository = each.value.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowAccountPull"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
      }
    ]
  })
}
