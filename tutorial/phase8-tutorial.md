# Financial RAG Agent — Phase 8: Helm, Karpenter & EKS Infrastructure

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase:** 8 of 12 — Helm Chart, Karpenter & EKS Infrastructure  
> **Components:** #75–100  
> **Time:** ~3 hours  
> **Prerequisite:** Phases 1–7 complete — Docker image built, CI pipeline green

---

## What You Will Build

A production-grade Kubernetes deployment for the entire Financial RAG Agent
stack, running on AWS EKS:

- **Helm chart** with `_helpers.tpl`, base `values.yaml`, and production overrides
- **Five workloads:** API Deployment, Agent Pool Deployment, Ingestion CronJob,
  pgvector StatefulSet, Redis StatefulSet
- **HorizontalPodAutoscalers** on API and Agent with asymmetric scale-up/down behaviour
- **AWS ALB Ingress** with TLS termination routed directly to pod IPs
- **Karpenter** for node provisioning — on-demand for API, Spot for ingestion
- **Terragrunt** modules for VPC, EKS, RDS, ElastiCache, IAM, S3, ECR
- **Three environments:** dev, staging, prod — each with isolated state

---

## Phase 8 File Tree

```
infrastructure/
├── helm/
│   ├── Chart.yaml                      ← Component #75
│   ├── values.yaml                     ← Component #75 (base)
│   ├── values.prod.yaml                ← Component #75 (prod overrides)
│   └── templates/
│       ├── _helpers.tpl                ← Component #76
│       ├── namespace.yaml              ← Component #77
│       ├── api-deployment.yaml         ← Component #81–82
│       ├── agent-deployment.yaml       ← Component #84
│       ├── pgvector-statefulset.yaml   ← Component #78–79
│       ├── redis-statefulset.yaml      ← Component #80
│       ├── api-hpa.yaml                ← Component #83
│       ├── agent-hpa.yaml              ← Component #85
│       ├── ingestion-cronjob.yaml      ← Component #86
│       ├── api-ingress.yaml            ← Component #87
│       └── servicemonitor.yaml         ← Component #88
└── terraform/
    ├── terragrunt.hcl                  ← Component #94 (root)
    ├── modules/
    │   ├── vpc/                        ← Component #95
    │   ├── eks/                        ← Component #96
    │   ├── rds/                        ← Component #97
    │   ├── elasticache/                ← Component #98
    │   ├── ecr/
    │   ├── iam/
    │   └── s3/
    └── environments/
        ├── dev/
        │   ├── env.hcl
        │   ├── vpc/terragrunt.hcl
        │   ├── eks/terragrunt.hcl
        │   ├── rds/terragrunt.hcl
        │   └── ...
        ├── staging/
        └── prod/                       ← Component #99–100
```

---

# PART 1 — HELM CHART

## Step 1 — Chart Structure (Component #75)

The Helm chart packages all Kubernetes manifests into a single versioned,
configurable unit. The key design principle: **never hardcode anything**.
Every value that differs between environments lives in `values.yaml` and
is overridden per environment.

```bash
mkdir -p infrastructure/helm/templates
```

Use `Chart.yaml` from your repo as-is. Key fields:

```yaml
apiVersion: v2          # Helm 3 only (apiVersion: v1 = Helm 2)
type: application       # vs 'library' for shared helper charts
version: 0.1.0          # chart version — bump on every chart change
appVersion: "1.0.0"     # application version — matches Docker image tag
dependencies: []        # no sub-charts — all components are in this chart
```

---

## Step 2 — Helpers Template (Component #76)

`_helpers.tpl` defines reusable template functions used across all manifests.
The underscore prefix tells Helm this file produces no Kubernetes objects —
it is a library of helper functions only.

Use `_helpers.tpl` from your repo as-is. Four helpers explained:

### `financial-rag-agent.fullname`

```
{{- define "financial-rag-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
...
```

Every resource name in the chart is prefixed with this. `trunc 63` enforces
Kubernetes' 63-character label value limit. `trimSuffix "-"` prevents names
like `financial-rag-agent-` with a trailing dash.

### `financial-rag-agent.selectorLabels` vs `financial-rag-agent.labels`

This is critical and a common Helm mistake:

```yaml
# selectorLabels — used in matchLabels (MUST be stable, never change)
app.kubernetes.io/name: financial-rag-agent
app.kubernetes.io/instance: my-release

# labels — used in metadata.labels (can include version, chart)
helm.sh/chart: financial-rag-agent-0.1.0   ← changes on upgrade
app.kubernetes.io/version: "1.0.0"          ← changes on upgrade
```

`matchLabels` in a Deployment selector is **immutable** after creation —
Kubernetes rejects updates. If you include `helm.sh/chart` (which changes
on every chart version bump) in `matchLabels`, every upgrade fails with
`field is immutable`. The fix: selector uses only stable `selectorLabels`,
metadata uses the full `labels`.

### `financial-rag-agent.image`

```
{{- define "financial-rag-agent.image" -}}
{{- $registry := .global.image.registry -}}
{{- $repo     := .image.repository -}}
{{- $tag      := .image.tag | default "latest" -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry $repo $tag -}}
...
```

Usage: `{{ include "financial-rag-agent.image" (dict "global" .Values.global "image" .Values.api.image) }}`

This allows the registry to be set globally (e.g. your ECR registry)
while repository and tag vary per component. In dev, `registry: ""` gives
`financial-rag-agent/api:latest`. In prod with ECR:
`123456789.dkr.ecr.us-east-1.amazonaws.com/financial-rag-agent/api:1.0.0`.

### `financial-rag-agent.podAnnotations`

```
{{- define "financial-rag-agent.podAnnotations" -}}
{{- $merged := merge (default dict .componentAnnotations) .Values.global.podAnnotations -}}
```

Merges global pod annotations (e.g. Prometheus scrape labels) with
per-component overrides. The `merge` function is Helm's deep merge —
component values override global values for the same key.

---

## Step 3 — Base Values (Component #75)

Use `values.yaml` from your repo as-is.

The structure follows a consistent pattern for every workload:

```yaml
componentName:
  enabled: true             # feature flag — set false to disable entirely
  image:
    repository: ...
    tag: latest
  replicaCount: 2
  resources:
    requests: { cpu, memory }
    limits: { cpu, memory }
  env: {}                   # KEY: value pairs → env vars
  envFrom: []               # secretRef / configMapRef
  livenessProbe: {}
  readinessProbe: {}
  hpa:
    enabled: true
    minReplicas: 2
    maxReplicas: 6
    targetCPUUtilizationPercentage: 70
```

Two things worth highlighting for your freeCodeCamp audience:

**`envFrom` with `secretRef`:**

```yaml
envFrom:
  - secretRef:
      name: financial-rag-secrets
```

This injects every key from the `financial-rag-secrets` Kubernetes Secret
as an environment variable. The Secret itself is created separately
(by Vault Agent, Sealed Secrets, or `kubectl create secret`) — the Helm
chart does not manage secret values, only references.

**`livenessProbe` vs `readinessProbe`:**

| Probe | Failure action | Purpose |
|---|---|---|
| `livenessProbe` | Restart the pod | Detect deadlock/crash |
| `readinessProbe` | Remove from Service endpoints | Detect not-yet-ready |

The API uses `/health` for both. The difference is timing — readiness has
`initialDelaySeconds: 10` (allow startup) while liveness has `15` (allow
slightly more time before declaring dead).

---

## Step 4 — Production Values Override (Component #99)

Use `values.prod.yaml` from your repo as-is.

Applied with: `helm upgrade --install fra . -f values.yaml -f values.prod.yaml`

Helm merges the two files — `values.prod.yaml` overrides only the keys it
specifies, everything else comes from `values.yaml`. This is the standard
multi-environment pattern.

Key production differences:

```yaml
# prod: ECR registry, pinned tag (never 'latest' in production)
global:
  image:
    registry: "123456789.dkr.ecr.us-east-1.amazonaws.com"
api:
  image:
    tag: "1.0.0"

# prod: more replicas, more resources
api:
  replicaCount: 5
  resources:
    limits:
      cpu: "2000m"
      memory: "2Gi"

# prod: ALB ingress enabled with ACM certificate
api:
  ingress:
    enabled: true
    annotations:
      alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:..."
```

> **Never use `latest` tag in production.** If a pod restarts and pulls a
> new `latest` image, your production environment silently changes without
> a deployment. Always pin to a specific semver tag or git SHA in production.

---

## Step 5 — StatefulSets: pgvector and Redis (Components #78–80)

Use `pgvector-statefulset.yaml` and `redis-statefulset.yaml` from your repo.

### Why StatefulSet, not Deployment?

Deployments are for stateless workloads — pods are interchangeable. If a
Deployment pod is replaced, its local storage is lost. That's fine for the
API (state lives in PostgreSQL), but catastrophic for PostgreSQL itself.

StatefulSets provide:
- **Stable pod names:** `fra-pgvector-0`, not `fra-pgvector-abc123`
- **Stable DNS:** `fra-pgvector-0.fra-pgvector-headless.financial-rag.svc.cluster.local`
- **Ordered startup/shutdown:** pod 0 before pod 1 (critical for primary/replica)
- **PVC per pod:** `volumeClaimTemplates` creates a dedicated PVC per replica

### Two services for pgvector

```
fra-pgvector-headless  (clusterIP: None)    ← StatefulSet DNS, replication
fra-pgvector           (clusterIP: <auto>)  ← API/Agent read/write
```

The headless service (`clusterIP: None`) is required by the StatefulSet for
stable DNS. It does not load-balance — it returns all pod IPs directly. The
regular ClusterIP service load-balances across replicas (currently one, but
the architecture is ready for read replicas).

### Redis with `medium: Memory` volume

```yaml
volumes:
  - name: model-cache
    emptyDir:
      medium: Memory     # tmpfs — in RAM, not disk
      sizeLimit: 500Mi
```

The agent pod uses an in-memory volume for the local embedding model cache.
The model stays in RAM between requests — no disk I/O on every embedding
call. `sizeLimit: 500Mi` prevents runaway memory usage.

---

## Step 6 — API and Agent Deployments (Components #81, #84)

Use `api-deployment.yaml` and `agent-deployment.yaml` from your repo.

### Security context (applied to every workload)

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

containers:
  - securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
```

Four security settings, each with a purpose:

| Setting | What it prevents |
|---|---|
| `runAsNonRoot: true` | Running as uid 0 (root) |
| `allowPrivilegeEscalation: false` | `setuid` binaries gaining root |
| `readOnlyRootFilesystem: true` | Writing to container filesystem |
| `capabilities: drop: ALL` | Linux kernel capabilities (raw sockets, etc.) |

`readOnlyRootFilesystem: true` requires a `/tmp` volume:

```yaml
volumeMounts:
  - name: tmp
    mountPath: /tmp
volumes:
  - name: tmp
    emptyDir: {}
```

Without this, the application cannot write temporary files and will crash.
Your Dockerfile already uses `/tmp` for this — the emptyDir volume satisfies
that requirement.

### Rolling update strategy

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 1    # at most 1 pod down during upgrade
    maxSurge: 2          # at most 2 extra pods during upgrade
```

With `replicaCount: 5` in production:
- `maxUnavailable: 1` → at least 4 pods serving traffic during rollout
- `maxSurge: 2` → at most 7 pods running simultaneously (5 + 2)

This gives you fast rollouts without traffic interruption.

---

## Step 7 — HPAs (Components #83, #85)

Use the HPA definitions from your repo as-is.

### Asymmetric scale behaviour

The API and Agent HPAs have different scale-up vs scale-down behaviour:

```yaml
# API HPA
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300   # 5 min cool-down
    policies:
      - type: Pods
        value: 2
        periodSeconds: 120            # remove at most 2 pods per 2 minutes
  scaleUp:
    stabilizationWindowSeconds: 60
    policies:
      - type: Pods
        value: 4
        periodSeconds: 60             # add up to 4 pods per minute

# Agent HPA — longer scale-down (agent loops must drain)
behavior:
  scaleDown:
    stabilizationWindowSeconds: 600   # 10 min cool-down
    policies:
      - type: Pods
        value: 1
        periodSeconds: 180
```

**Scale up fast, scale down slow.** A sudden traffic spike needs immediate
capacity — wait 60 seconds, add 4 pods. After the spike, wait 5–10 minutes
before removing pods — avoids removing capacity that's still needed.

The Agent has a 10-minute scale-down window because agent reasoning loops
can run for 30–60 seconds. Removing a pod mid-loop drops the request.

---

## Step 8 — Ingestion CronJob (Component #86)

Use the CronJob from your repo as-is.

```yaml
schedule: "0 2 * * *"           # 02:00 UTC — before US market open
concurrencyPolicy: Forbid        # never run two ingestion jobs simultaneously
activeDeadlineSeconds: 10800     # 3-hour hard cap — kill if hung
backoffLimit: 2                  # retry at most 2 times on failure
```

`concurrencyPolicy: Forbid` is critical. Without it, if an ingestion job
runs longer than expected (EDGAR is slow), the next scheduled run starts
while the first is still running. Two concurrent ingestion jobs hitting
EDGAR simultaneously would exceed the rate limit and corrupt the dedup logic.

The ingestion container has `readOnlyRootFilesystem: true` with a `/tmp`
emptyDir — consistent with the API and Agent security posture.

---

## Step 9 — ALB Ingress (Component #87)

Use the Ingress template from your repo as-is.

The AWS Load Balancer Controller reads these annotations and provisions a
real AWS Application Load Balancer:

```yaml
annotations:
  alb.ingress.kubernetes.io/scheme: internet-facing
  alb.ingress.kubernetes.io/target-type: ip        # route to pod IPs directly
  alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
  alb.ingress.kubernetes.io/ssl-redirect: "443"
  alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:..."
  alb.ingress.kubernetes.io/healthcheck-path: /health
  alb.ingress.kubernetes.io/group.name: financial-rag-prod
```

**`target-type: ip`** vs `target-type: instance`:
- `instance`: traffic goes to the node, then kube-proxy routes to the pod (double-hop)
- `ip`: traffic goes directly to the pod IP (single-hop, lower latency)

`ip` mode requires pods to be in subnets the ALB can reach — use private
subnets with the ALB in public subnets.

**`group.name`:** Multiple Ingress objects share one ALB. Without grouping,
each Ingress creates a new ALB ($0.008/hour each). With grouping, all
Ingresses in the same group share one ALB.

---

## Step 10 — ServiceMonitor (Component #88)

Use `servicemonitor.yaml` from your repo as-is.

```yaml
labels:
  release: kube-prometheus-stack   # MUST match Prometheus operator selector
```

This label is not arbitrary — the Prometheus operator only picks up
ServiceMonitors that match its label selector. If you installed
kube-prometheus-stack with the default release name, this label is correct.
If you used a different release name, change it to match.

---

# PART 2 — KARPENTER NODE PROVISIONING

## Step 11 — Karpenter IAM Role (Component #89)

Karpenter needs IAM permissions to create EC2 instances, manage spot
interruptions, and read from SSM for AMI IDs.

```bash
# Create the IAM role (using eksctl for simplicity)
eksctl create iamserviceaccount \
  --name karpenter \
  --namespace karpenter \
  --cluster financial-rag-prod \
  --role-name KarpenterControllerRole \
  --attach-policy-arn arn:aws:iam::ACCOUNT_ID:policy/KarpenterControllerPolicy \
  --approve
```

The Karpenter controller policy needs:
- `ec2:RunInstances`, `ec2:TerminateInstances`
- `ec2:DescribeInstances`, `ec2:DescribeInstanceTypes`
- `iam:PassRole` (to pass the node IAM role to new instances)
- `ssm:GetParameter` (for AMI discovery)

---

## Step 12 — Karpenter Installation (Component #90)

```bash
helm repo add karpenter https://charts.karpenter.sh
helm repo update

helm upgrade --install karpenter karpenter/karpenter \
  --namespace karpenter \
  --create-namespace \
  --set "serviceAccount.annotations.eks\.amazonaws\.com/role-arn=arn:aws:iam::ACCOUNT_ID:role/KarpenterControllerRole" \
  --set "settings.clusterName=financial-rag-prod" \
  --set "settings.interruptionQueue=financial-rag-prod" \
  --version 0.37.0
```

---

## Step 13 — EC2NodeClass and NodePools (Components #91–93)

Create `infrastructure/helm/karpenter/ec2nodeclass.yaml`:

```yaml
# infrastructure/helm/karpenter/ec2nodeclass.yaml
# Component #91 — defines what kind of EC2 instances Karpenter can launch
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: financial-rag-default
spec:
  amiFamily: AL2           # Amazon Linux 2
  role: "KarpenterNodeRole"

  # Subnets where nodes launch — must match your VPC
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "financial-rag-prod"

  # Security groups for launched nodes
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "financial-rag-prod"

  # Block device — gp3, encrypted at rest
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 50Gi
        volumeType: gp3
        encrypted: true

  # User data to install SSM agent and configure the node
  userData: |
    #!/bin/bash
    /etc/eks/bootstrap.sh financial-rag-prod
```

Create `infrastructure/helm/karpenter/nodepool-api.yaml` (Component #92 — on-demand):

```yaml
# infrastructure/helm/karpenter/nodepool-api.yaml
# On-demand nodes for the API — consistent latency, no interruption risk
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: api-ondemand
spec:
  template:
    metadata:
      labels:
        role: application
    spec:
      nodeClassRef:
        name: financial-rag-default
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["m5", "m6i", "m6a"]   # General purpose, consistent perf
        - key: karpenter.k8s.aws/instance-size
          operator: In
          values: ["xlarge", "2xlarge"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
      taints:
        - key: dedicated
          value: application
          effect: NoSchedule

  # Scale down aggressively when underutilised
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 5m

  limits:
    cpu: "100"
    memory: 400Gi
```

Create `infrastructure/helm/karpenter/nodepool-spot.yaml` (Component #93 — spot for ingestion):

```yaml
# infrastructure/helm/karpenter/nodepool-spot.yaml
# Spot nodes for ingestion — 70% cheaper, interruption tolerated
# Ingestion is idempotent (SHA-256 dedup) so interruption just retries
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: ingestion-spot
spec:
  template:
    metadata:
      labels:
        role: ingestion
    spec:
      nodeClassRef:
        name: financial-rag-default
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot"]
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["m5", "m6i", "c5", "c6i"]   # Diverse family = better spot availability
        - key: karpenter.k8s.aws/instance-size
          operator: In
          values: ["large", "xlarge", "2xlarge"]
      taints:
        - key: dedicated
          value: ingestion
          effect: NoSchedule

  # Spot interruption: consolidate quickly, don't keep expensive nodes idle
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 30s

  limits:
    cpu: "50"
    memory: 200Gi
```

Apply to the cluster:

```bash
kubectl apply -f infrastructure/helm/karpenter/ec2nodeclass.yaml
kubectl apply -f infrastructure/helm/karpenter/nodepool-api.yaml
kubectl apply -f infrastructure/helm/karpenter/nodepool-spot.yaml
```

---

# PART 3 — TERRAGRUNT INFRASTRUCTURE

## Step 14 — Terragrunt Root Config (Component #94)

Terragrunt wraps Terraform to provide DRY infrastructure code — one module
definition, multiple environment instantiations.

Create `infrastructure/terraform/terragrunt.hcl`:

```hcl
# infrastructure/terraform/terragrunt.hcl
# Root terragrunt.hcl — inherited by all child modules via find_in_parent_folders()

locals {
  # Read environment-specific config from env.hcl
  env_vars = read_terragrunt_config(find_in_parent_folders("env.hcl"))
  env      = local.env_vars.locals.env
  region   = local.env_vars.locals.aws_region
  account  = local.env_vars.locals.account_id
}

# Remote state — each environment + module gets its own state file
remote_state {
  backend = "s3"
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
  config = {
    bucket         = "financial-rag-terraform-state-${local.account}"
    key            = "${local.env}/${path_relative_to_include()}/terraform.tfstate"
    region         = local.region
    encrypt        = true
    dynamodb_table = "financial-rag-terraform-locks"
  }
}

# Generate provider config for every module
generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<-EOF
    provider "aws" {
      region = "${local.region}"
      default_tags {
        tags = {
          Environment = "${local.env}"
          Project     = "financial-rag-agent"
          ManagedBy   = "terragrunt"
        }
      }
    }
  EOF
}
```

---

## Step 15 — Environment Configs (Component #99)

Each environment has an `env.hcl` that sets all environment-specific values.
Child modules read these via `include "env"`.

Create `infrastructure/terraform/environments/dev/env.hcl`:

```hcl
# infrastructure/terraform/environments/dev/env.hcl
locals {
  env        = "dev"
  aws_region = "us-east-1"
  account_id = "123456789012"     # replace with your AWS account ID

  # VPC
  vpc_cidr           = "10.10.0.0/16"
  private_subnets    = ["10.10.1.0/24", "10.10.2.0/24", "10.10.3.0/24"]
  public_subnets     = ["10.10.101.0/24", "10.10.102.0/24", "10.10.103.0/24"]
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

  # EKS
  eks_cluster_version = "1.30"
  eks_node_groups = {
    general = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3
      desired_size   = 2
    }
  }

  # RDS
  rds_instance_class        = "db.t3.medium"
  rds_allocated_storage     = 20
  rds_backup_retention_days = 1
  rds_multi_az              = false

  # ElastiCache
  elasticache_node_type       = "cache.t3.micro"
  elasticache_num_cache_nodes = 1
  elasticache_multi_az        = false

  # ECR
  ecr_image_retention_count = 10
}
```

Create `infrastructure/terraform/environments/staging/env.hcl`:

```hcl
# infrastructure/terraform/environments/staging/env.hcl
locals {
  env        = "staging"
  aws_region = "us-east-1"
  account_id = "123456789012"

  vpc_cidr           = "10.20.0.0/16"
  private_subnets    = ["10.20.1.0/24", "10.20.2.0/24", "10.20.3.0/24"]
  public_subnets     = ["10.20.101.0/24", "10.20.102.0/24", "10.20.103.0/24"]
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

  eks_cluster_version = "1.30"
  eks_node_groups = {
    general = {
      instance_types = ["t3.large"]
      min_size       = 2
      max_size       = 5
      desired_size   = 2
    }
  }

  rds_instance_class        = "db.t3.large"
  rds_allocated_storage     = 50
  rds_backup_retention_days = 7
  rds_multi_az              = false

  elasticache_node_type       = "cache.t3.small"
  elasticache_num_cache_nodes = 1
  elasticache_multi_az        = false

  ecr_image_retention_count = 20
}
```

Create `infrastructure/terraform/environments/prod/env.hcl`:

```hcl
# infrastructure/terraform/environments/prod/env.hcl
locals {
  env        = "prod"
  aws_region = "us-east-1"
  account_id = "123456789012"

  vpc_cidr           = "10.30.0.0/16"
  private_subnets    = ["10.30.1.0/24", "10.30.2.0/24", "10.30.3.0/24"]
  public_subnets     = ["10.30.101.0/24", "10.30.102.0/24", "10.30.103.0/24"]
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

  eks_cluster_version = "1.30"
  eks_node_groups = {
    system = {
      instance_types = ["m5.large"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2
      labels         = { role = "system" }
    }
  }

  rds_instance_class        = "db.r6g.xlarge"
  rds_allocated_storage     = 500
  rds_backup_retention_days = 30
  rds_multi_az              = true

  elasticache_node_type       = "cache.r6g.large"
  elasticache_num_cache_nodes = 2
  elasticache_multi_az        = true

  ecr_image_retention_count = 50
}
```

---

## Step 16 — Terragrunt Module Configs (Component #100)

Each module in each environment gets a `terragrunt.hcl` that declares:
- Which Terraform module source to use
- Which other modules it depends on (with mock outputs for `plan`)
- Which `env.hcl` values to pass as inputs

Use the module configs from your repo as-is. The dependency pattern is worth
explaining to your audience:

```hcl
# eks/terragrunt.hcl
dependency "vpc" {
  config_path = "../vpc"
  mock_outputs = {
    vpc_id             = "vpc-00000000"
    private_subnet_ids = ["subnet-00000001", "subnet-00000002"]
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}
```

`mock_outputs` allows `terragrunt plan` to run without first applying the
VPC module. This is essential for PR checks — you want to validate that the
EKS module will work without actually provisioning infrastructure.
`mock_outputs_allowed_terraform_commands` restricts mocks to `validate` and
`plan` only — `apply` always uses real outputs.

---

## Step 17 — Apply Order

Terragrunt manages the apply order automatically based on `dependency` blocks.
To apply a full environment:

```bash
cd infrastructure/terraform/environments/prod

# Apply in dependency order (VPC → EKS → RDS → ElastiCache → IAM)
terragrunt run-all apply --terragrunt-non-interactive

# Apply a single module
cd vpc && terragrunt apply
```

The S3 module is a special case — it must be applied first with a local
backend (chicken-and-egg: you need S3 for remote state, but need Terragrunt
to create S3):

```bash
# First time only:
cd environments/prod/s3
terragrunt apply    # uses local backend
# After this, all other modules use S3 for state
```

---

## Step 18 — Deploy the Helm Chart

With infrastructure provisioned, deploy the application:

```bash
# Create the namespace
kubectl create namespace financial-rag

# Create the secrets (from Vault or manual for first deployment)
kubectl create secret generic financial-rag-secrets \
  --namespace financial-rag \
  --from-literal=POSTGRES_PASSWORD="$(vault read -field=password database/creds/api-role)" \
  --from-literal=REDIS_PASSWORD="your-redis-password" \
  --from-literal=OPENAI_API_KEY="your-llm-key" \
  --from-literal=API_KEY="your-api-key"

# Dry-run to validate (renders all templates, validates against K8s API)
helm upgrade --install financial-rag-agent \
  infrastructure/helm/ \
  --namespace financial-rag \
  -f infrastructure/helm/values.yaml \
  -f infrastructure/helm/values.prod.yaml \
  --dry-run --debug

# Deploy
helm upgrade --install financial-rag-agent \
  infrastructure/helm/ \
  --namespace financial-rag \
  -f infrastructure/helm/values.yaml \
  -f infrastructure/helm/values.prod.yaml \
  --wait \
  --timeout 10m

# Verify
kubectl get pods -n financial-rag
kubectl get hpa -n financial-rag
kubectl get ingress -n financial-rag
```

Expected output:
```
NAME                                    READY   STATUS    RESTARTS
financial-rag-agent-api-7d9f8b-xxxxx    1/1     Running   0
financial-rag-agent-api-7d9f8b-yyyyy    1/1     Running   0
financial-rag-agent-agent-6c8f9a-xxxxx  1/1     Running   0
financial-rag-agent-pgvector-0          1/1     Running   0
financial-rag-agent-redis-0             1/1     Running   0
```

---

## Common Errors and Fixes — Phase 8

| Error | Cause | Fix |
|---|---|---|
| `field is immutable` on upgrade | `helm.sh/chart` in `matchLabels` | Use `selectorLabels` (stable) in `matchLabels`, `labels` in metadata only |
| `ImagePullBackOff` | ECR not authenticated | `aws ecr get-login-password \| docker login` or verify IRSA role |
| PVC stuck in `Pending` | StorageClass `gp3` not installed | Install EBS CSI driver: `eksctl utils install-ebs-csi-driver` |
| `no nodes available` | Karpenter not provisioning | Check `kubectl logs -n karpenter deployment/karpenter` |
| HPA not scaling | Metrics server not installed | `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml` |
| `terragrunt: S3 bucket not found` | State bucket not created | Apply `s3` module first with local backend |
| `dependency cycle` | Circular module dependencies | Check `dependency.config_path` — must form a DAG |
| Ingestion pod not scheduled | Missing spot capacity in region | Add more instance families to `nodepool-spot.yaml` requirements |
| `connection refused` to pgvector | Pod not yet ready | Check readinessProbe — `pg_isready` must pass before traffic |

---

## What's Next — Phase 9

Phase 9 adds GitOps and network security:
- **ArgoCD** App of Apps pattern — Git is the source of truth for all K8s state
- Sync waves to enforce deployment order (databases before API)
- **Cilium** eBPF networking replacing kube-proxy
- Deny-all NetworkPolicy with explicit allow rules per service
- L7 HTTP policy — allow only `POST /query` and `GET /health` from ingress
- Hubble for real-time network flow visibility
- Multi-cluster mesh for staging → prod promotion
