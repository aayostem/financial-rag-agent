# Commit 24 — Git Message & PR Description

---

## Git Commit Message

```
feat: add Helm chart for financial-rag-agent Kubernetes deployment

- helm/financial-rag-agent/Chart.yaml       — chart metadata, v0.1.0 / appVersion 1.0.0
- helm/financial-rag-agent/values.yaml      — base defaults (all envs inherit)
- helm/financial-rag-agent/values.prod.yaml — EKS prod overrides, large-scale

Templates:
  templates/_helpers.tpl              — name, fullname, image, labels, selectorLabels helpers
  templates/namespace.yaml            — financial-rag namespace
  templates/api-deployment.yaml       — FastAPI Deployment + ClusterIP Service + ServiceAccount
  templates/api-hpa.yaml              — HPA (autoscaling/v2), CPU + memory metrics
  templates/api-ingress.yaml          — AWS ALB Ingress, TLS termination, ip target-type
  templates/agent-deployment.yaml     — Agent Pool Deployment + Service + ServiceAccount
  templates/agent-hpa.yaml            — HPA with asymmetric scale-up/down behavior
  templates/ingestion-cronjob.yaml    — EDGAR CronJob, concurrencyPolicy: Forbid
  templates/pgvector-statefulset.yaml — StatefulSet + headless + ClusterIP services, PVC template
  templates/redis-statefulset.yaml    — StatefulSet + headless + ClusterIP services, AOF persistence
  templates/servicemonitor.yaml       — Prometheus ServiceMonitor (kube-prometheus-stack)
  templates/NOTES.txt                 — post-install instructions

WHY: Docker Compose is a local dev tool. EKS production requires:
  - HPA to absorb query load spikes on agent pods (LLM inference is CPU-spiky)
  - StatefulSets with PVC templates so pgvector HNSW index and Redis AOF survive restarts
  - Separate resource quotas: agent pods get 4–8 Gi RAM for embedding model in tmpfs
  - Rolling updates with maxUnavailable=1 so no query traffic drops during deploys
  - ALB Ingress with ip target-type to avoid double-hop through kube-proxy
  - IRSA-annotated ServiceAccounts ready for ECR pull + Secrets Manager access
  Helm packages all of this into a single versioned artifact (v0.1.0) deployable
  identically across staging and production via values file overlays.
```

---

## PR Description

### What

Adds a production-grade Helm chart under `helm/financial-rag-agent/` covering all
five system components: API, Agent Pool, Ingestion CronJob, pgvector, and Redis.

### Why Docker Compose → Helm

| Concern | Docker Compose | This Helm chart |
|---|---|---|
| Horizontal scaling | Manual `--scale` | HPA on CPU + memory, autoscaling/v2 |
| Pod restart survival | Volume bind mount | StatefulSet + PVC template (gp3/gp3-iops) |
| Rolling deploys | Recreate | `maxUnavailable: 1, maxSurge: 2` |
| TLS termination | None | AWS ALB, `target-type: ip` |
| Secrets | `.env` file | `envFrom.secretRef` → AWS Secrets Manager via IRSA |
| Observability | Docker logs | Prometheus ServiceMonitor, pod annotations |
| Multi-env parity | Multiple compose files | Single chart, values file overlays |

### Prod Scale (values.prod.yaml)

| Component | Min Replicas | Max Replicas | CPU Limit | Memory Limit |
|---|---|---|---|---|
| API | 5 | 20 | 2000m | 2 Gi |
| Agent Pool | 5 | 25 | 4000m | 8 Gi |
| Ingestion | CronJob | CronJob | 2000m | 4 Gi |
| pgvector | 1 (StatefulSet) | — | 4000m | 16 Gi |
| Redis | 1 (StatefulSet) | — | 1000m | 4 Gi |

### HPA Tuning Rationale

- **API**: Scale at 60% CPU / 70% memory. Scale-down window 5 min to avoid request drops.
- **Agent Pool**: Lower thresholds (55%/65%) — LLM + embedding inference causes CPU spikes.
  Scale-up burst: +5 pods/60s. Scale-down: only 1 pod/3 min with a 10-min stabilisation window
  to let in-flight agent loops drain cleanly.

### IRSA

All three ServiceAccounts (`-api`, `-agent`, `-ingestion`) carry the `eks.amazonaws.com/role-arn`
annotation placeholder. Wire up after creating the IAM roles in Commit 25.

### Deploy Commands

```bash
# Staging
helm upgrade --install fra-staging ./helm/financial-rag-agent \
  -f helm/financial-rag-agent/values.yaml \
  --namespace financial-rag --create-namespace

# Production
helm upgrade --install fra-prod ./helm/financial-rag-agent \
  -f helm/financial-rag-agent/values.yaml \
  -f helm/financial-rag-agent/values.prod.yaml \
  --namespace financial-rag --create-namespace

# Lint before push
helm lint ./helm/financial-rag-agent -f helm/financial-rag-agent/values.prod.yaml
```

### Out of Scope (next commits)

- Commit 25: NetworkPolicy, PodDisruptionBudget, IRSA IAM roles, gp3-iops StorageClass
- Commit 26: Karpenter node provisioner config, Cluster Autoscaler
- Commit 27: GitHub Actions `helm upgrade` deploy job