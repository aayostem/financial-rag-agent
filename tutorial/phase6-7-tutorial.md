# Financial RAG Agent — Phase 6 & 7: Vault Zero-Trust Secrets + CI/CD Pipeline

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase 6:** Vault Dynamic Secrets & Zero-Trust — Components #52–69  
> **Phase 7:** CI/CD Pipeline — Components #70–74  
> **Time:** Phase 6 ~2.5 hours | Phase 7 ~1.5 hours  
> **Prerequisite:** Phases 1–5 complete — API running, Docker services healthy

---

## What You Will Build

**Phase 6** — Zero-trust secrets management with HashiCorp Vault:
- Vault installed and initialised on Kubernetes
- Dynamic PostgreSQL credentials — rotated automatically, never stored
- PKI secrets engine — short-lived TLS certificates issued on demand
- Kubernetes auth method — pods authenticate with their service account JWT
- Vault policies with least-privilege access per workload
- Three distinct service accounts: api, agent, ingestion
- Vault Agent sidecar — secrets injected as files, never in environment variables
- Python `hvac` client for runtime secret retrieval
- Production HA with Raft storage

**Phase 7** — CI/CD pipeline with GitHub Actions:
- Secret scanning with gitleaks on every push
- Dependency vulnerability audit with pip-audit
- Lint and type checking (ruff + mypy)
- Unit tests with coverage enforcement
- Integration tests with real PostgreSQL and Redis services
- Multi-stage Docker build pushed to GitHub Container Registry
- Container image scanning with Trivy (SARIF uploaded to GitHub Security)
- OPA Rego policy to enforce least-privilege Vault policies

---

# PHASE 6 — Vault Dynamic Secrets & Zero-Trust

## The Problem Vault Solves

In Phases 1–5, secrets live in `.env` and are injected as environment
variables. This has three problems at production scale:

1. **Static credentials** — if `POSTGRES_PASSWORD` leaks, it works forever
   until someone manually rotates it.
2. **Shared credentials** — every pod uses the same password. If the
   ingestion service is compromised, the attacker has database access
   identical to the API service.
3. **Environment variable exposure** — any process in the container can
   read `os.environ`. Secrets in files with restricted permissions are
   harder to exfiltrate.

Vault solves all three:
- **Dynamic credentials** rotate automatically every hour
- **Per-role credentials** — api, agent, and ingestion each get different
  database users with different privileges
- **File-based injection** — Vault Agent writes secrets to files that only
  the application process can read

---

## New Files in This Phase

```
infrastructure/
├── vault/
│   ├── vault-bootstrap.sh          ← Component #53–63 (init + policy setup)
│   ├── vault-agent-config.hcl      ← Component #65 (sidecar template)
│   └── policies/
│       ├── api-policy.hcl
│       ├── agent-policy.hcl
│       └── ingestion-policy.hcl
├── k8s/
│   └── service-accounts.yaml       ← Component #64
└── docker/
    └── prometheus/
        └── prometheus.yml          ← referenced by docker-compose.prod.yml
src/financial_rag/security/
└── vault.py                        ← Component #67 (hvac integration)
```

---

## Step 1 — Understand the Vault Architecture

Before writing any code, understand how the components connect:

```
                    ┌─────────────────────────────────┐
                    │           HashiCorp Vault         │
                    │  ┌──────────┐  ┌──────────────┐  │
                    │  │ Database │  │     PKI      │  │
                    │  │  Engine  │  │   Engine     │  │
                    │  └──────────┘  └──────────────┘  │
                    │  ┌──────────────────────────────┐ │
                    │  │   Kubernetes Auth Method     │ │
                    │  └──────────────────────────────┘ │
                    └───────────────┬─────────────────┘
                                    │ issues dynamic creds
                    ┌───────────────▼─────────────────┐
                    │         Vault Agent Sidecar      │
                    │   reads K8s service account JWT  │
                    │   writes secrets to /vault/secrets│
                    └───────────────┬─────────────────┘
                                    │ files in shared volume
                    ┌───────────────▼─────────────────┐
                    │        Application Container     │
                    │   reads /vault/secrets/config    │
                    │   POSTGRES_PASSWORD from file    │
                    └─────────────────────────────────┘
```

The key insight: **the application never talks to Vault directly in
normal operation**. Vault Agent handles authentication and secret renewal.
The app just reads files.

---

## Step 2 — Install Vault on Kubernetes (Component #52)

```bash
# Add the HashiCorp Helm repository
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo update

# Install Vault in dev mode first (single pod, no HA, in-memory)
helm install vault hashicorp/vault \
  --namespace vault \
  --create-namespace \
  --set "server.dev.enabled=true" \
  --set "injector.enabled=true"

# Wait for Vault to be ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=vault \
  -n vault \
  --timeout=120s

# Verify
kubectl exec -n vault vault-0 -- vault status
```

Expected output:
```
Key             Value
---             -----
Seal Type       shamir
Initialized     true
Sealed          false
Version         1.15.x
```

> **Dev mode vs production:** Dev mode starts unsealed with a root token
> stored in `vault-0` logs. It loses all data on pod restart. Use it only
> to learn the workflow. Production setup (HA + Raft + KMS auto-unseal)
> is covered in Component #68 at the end of this phase.

---

## Step 3 — Vault Bootstrap Script (Components #53–63)

This script automates the entire Vault configuration. Run it once after
installation. Each command is idempotent — safe to re-run.

Create `infrastructure/vault/vault-bootstrap.sh`:

```bash
#!/usr/bin/env bash
# infrastructure/vault/vault-bootstrap.sh
# =============================================================================
# Configures Vault for the Financial RAG Agent.
# Run after: helm install vault hashicorp/vault
#
# What this script does:
#   1. Enables the database secrets engine (Component #54)
#   2. Configures PostgreSQL connection (Component #55)
#   3. Creates per-role database credentials (Component #56)
#   4. Enables the PKI secrets engine (Component #57)
#   5. Generates the root certificate (Component #58)
#   6. Creates PKI roles (Component #59)
#   7. Enables Kubernetes auth method (Component #60)
#   8. Configures Kubernetes auth (Component #61)
#   9. Creates least-privilege policies (Component #62)
#   10. Creates Kubernetes auth roles (Component #63)
# =============================================================================

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-root}"          # dev mode default
NAMESPACE="${NAMESPACE:-financial-rag}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres.financial-rag.svc.cluster.local}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_ADMIN_USER="${POSTGRES_ADMIN_USER:-finrag_admin}"
POSTGRES_ADMIN_PASSWORD="${POSTGRES_ADMIN_PASSWORD:?Required}"

export VAULT_ADDR VAULT_TOKEN

log()     { echo "[$(date +%H:%M:%S)] $*"; }
success() { echo "[$(date +%H:%M:%S)] ✅ $*"; }

# =============================================================================
# Component #54 — Database Secrets Engine
# =============================================================================
log "Enabling database secrets engine..."
vault secrets enable database 2>/dev/null || log "database engine already enabled"
success "Database secrets engine ready"

# =============================================================================
# Component #55 — PostgreSQL Connection
# =============================================================================
log "Configuring PostgreSQL connection..."
vault write database/config/financial-rag \
  plugin_name=postgresql-database-plugin \
  allowed_roles="api-role,agent-role,ingestion-role" \
  connection_url="postgresql://{{username}}:{{password}}@${POSTGRES_HOST}:${POSTGRES_PORT}/financial_rag?sslmode=require" \
  username="${POSTGRES_ADMIN_USER}" \
  password="${POSTGRES_ADMIN_PASSWORD}" \
  password_authentication="scram-sha-256"

success "PostgreSQL connection configured"

# =============================================================================
# Component #56 — Database Roles (one per service, different privileges)
# =============================================================================
log "Creating database roles..."

# API role: read-only access (SELECT on chunks + filings + analysis_history)
vault write database/roles/api-role \
  db_name=financial-rag \
  creation_statements="
    CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';
    GRANT SELECT ON filings, financial_chunks, analysis_history TO \"{{name}}\";
    GRANT INSERT ON analysis_history TO \"{{name}}\";
  " \
  default_ttl="1h" \
  max_ttl="24h"

# Agent role: read + write analysis_history
vault write database/roles/agent-role \
  db_name=financial-rag \
  creation_statements="
    CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';
    GRANT SELECT ON filings, financial_chunks TO \"{{name}}\";
    GRANT SELECT, INSERT ON analysis_history TO \"{{name}}\";
  " \
  default_ttl="1h" \
  max_ttl="24h"

# Ingestion role: write access for bulk insert
vault write database/roles/ingestion-role \
  db_name=financial-rag \
  creation_statements="
    CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';
    GRANT SELECT, INSERT, UPDATE ON filings, financial_chunks TO \"{{name}}\";
    GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO \"{{name}}\";
  " \
  default_ttl="2h" \
  max_ttl="6h"

success "Database roles created (api-role, agent-role, ingestion-role)"

# =============================================================================
# Component #57 — PKI Secrets Engine
# =============================================================================
log "Enabling PKI secrets engine..."
vault secrets enable pki 2>/dev/null || log "pki engine already enabled"
vault secrets tune -max-lease-ttl=8760h pki
success "PKI engine ready"

# =============================================================================
# Component #58 — Root Certificate
# =============================================================================
log "Generating root certificate..."
vault write -field=certificate pki/root/generate/internal \
  common_name="financial-rag-agent CA" \
  ttl=8760h \
  > /tmp/finrag-ca.crt

vault write pki/config/urls \
  issuing_certificates="${VAULT_ADDR}/v1/pki/ca" \
  crl_distribution_points="${VAULT_ADDR}/v1/pki/crl"

success "Root CA generated → /tmp/finrag-ca.crt"

# =============================================================================
# Component #59 — PKI Role
# =============================================================================
log "Creating PKI role..."
vault write pki/roles/financial-rag \
  allowed_domains="financial-rag.svc.cluster.local,financial-rag.internal" \
  allow_subdomains=true \
  max_ttl=72h \
  require_cn=false

success "PKI role created"

# =============================================================================
# Component #60–61 — Kubernetes Auth Method
# =============================================================================
log "Enabling Kubernetes auth method..."
vault auth enable kubernetes 2>/dev/null || log "kubernetes auth already enabled"

# Read the K8s cluster's service account JWT and CA cert
K8S_HOST=$(kubectl config view --raw --minify --flatten \
  -o jsonpath='{.clusters[].cluster.server}')

vault write auth/kubernetes/config \
  kubernetes_host="${K8S_HOST}" \
  kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token

success "Kubernetes auth configured → ${K8S_HOST}"

# =============================================================================
# Component #62 — Vault Policies (one per service)
# =============================================================================
log "Writing Vault policies..."

# API policy — read LLM key, Redis password, dynamic DB creds
vault policy write api-policy - <<'EOF'
path "secret/data/financial-rag/prod/llm" {
  capabilities = ["read"]
}
path "secret/data/financial-rag/prod/redis" {
  capabilities = ["read"]
}
path "secret/data/financial-rag/prod/config" {
  capabilities = ["read"]
}
path "database/creds/api-role" {
  capabilities = ["read"]
}
path "pki/issue/financial-rag" {
  capabilities = ["create", "update"]
}
EOF

# Agent policy
vault policy write agent-policy - <<'EOF'
path "secret/data/financial-rag/prod/llm" {
  capabilities = ["read"]
}
path "secret/data/financial-rag/prod/redis" {
  capabilities = ["read"]
}
path "database/creds/agent-role" {
  capabilities = ["read"]
}
EOF

# Ingestion policy
vault policy write ingestion-policy - <<'EOF'
path "secret/data/financial-rag/prod/edgar" {
  capabilities = ["read"]
}
path "secret/data/financial-rag/prod/redis" {
  capabilities = ["read"]
}
path "database/creds/ingestion-role" {
  capabilities = ["read"]
}
EOF

success "Policies written: api-policy, agent-policy, ingestion-policy"

# =============================================================================
# Component #63 — Kubernetes Auth Roles
# =============================================================================
log "Creating Kubernetes auth roles..."

vault write auth/kubernetes/role/financial-rag-api \
  bound_service_account_names="financial-rag-agent-api" \
  bound_service_account_namespaces="${NAMESPACE}" \
  policies="api-policy" \
  ttl=1h \
  max_ttl=24h

vault write auth/kubernetes/role/financial-rag-agent \
  bound_service_account_names="financial-rag-agent-agent" \
  bound_service_account_namespaces="${NAMESPACE}" \
  policies="agent-policy" \
  ttl=1h \
  max_ttl=24h

vault write auth/kubernetes/role/financial-rag-ingestion \
  bound_service_account_names="financial-rag-agent-ingestion" \
  bound_service_account_namespaces="${NAMESPACE}" \
  policies="ingestion-policy" \
  ttl=2h \
  max_ttl=6h

success "Kubernetes auth roles created"

# =============================================================================
# Seed static secrets
# =============================================================================
log "Seeding static secrets (update values before production use)..."

vault kv put secret/financial-rag/prod/llm \
  openai_api_key="REPLACE_ME" \
  llm_model="llama-3.3-70b-versatile" \
  llm_base_url="https://api.groq.com/openai/v1"

vault kv put secret/financial-rag/prod/redis \
  password="REPLACE_ME" \
  host="redis.financial-rag.svc.cluster.local" \
  port="6379"

vault kv put secret/financial-rag/prod/edgar \
  user_agent="financial-rag-agent contact@yourdomain.com"

vault kv put secret/financial-rag/prod/config \
  api_key="REPLACE_ME" \
  cors_origins="https://yourdomain.com"

success "Static secrets seeded — update values before production use"

echo ""
echo "=================================================="
echo " Vault bootstrap complete"
echo " Verify with: vault read database/creds/api-role"
echo "=================================================="
```

Make it executable:

```bash
chmod +x infrastructure/vault/vault-bootstrap.sh
```

---

## Step 4 — Vault Policies (Component #62)

The bootstrap script writes policies inline. For audit and GitOps tracking,
also store them as files. Create `infrastructure/vault/policies/`:

```hcl
# infrastructure/vault/policies/api-policy.hcl
# Least-privilege policy for the API service.
# Read-only on secrets. Dynamic DB creds (read = request new credentials).

path "secret/data/financial-rag/prod/llm" {
  capabilities = ["read"]
}

path "secret/data/financial-rag/prod/redis" {
  capabilities = ["read"]
}

path "secret/data/financial-rag/prod/config" {
  capabilities = ["read"]
}

# Dynamic database credentials — "read" here means "generate a new credential"
path "database/creds/api-role" {
  capabilities = ["read"]
}

# Issue TLS certificates for mTLS between services
path "pki/issue/financial-rag" {
  capabilities = ["create", "update"]
}
```

```hcl
# infrastructure/vault/policies/ingestion-policy.hcl
# Ingestion service only needs EDGAR config and Redis.
# No LLM access — ingestion does not call the LLM.

path "secret/data/financial-rag/prod/edgar" {
  capabilities = ["read"]
}

path "secret/data/financial-rag/prod/redis" {
  capabilities = ["read"]
}

path "database/creds/ingestion-role" {
  capabilities = ["read"]
}
```

> **Why three separate policies?**  
> The ingestion service does not need the LLM API key. If ingestion is
> compromised, the attacker cannot call the LLM. The API service does not
> need EDGAR credentials. Compartmentalisation limits blast radius.
> This is the principle of least privilege applied per-workload.

---

## Step 5 — Kubernetes Service Accounts (Component #64)

Your repo has Helm-templated service accounts. Here is the plain Kubernetes
equivalent for learners not yet using Helm. Create
`infrastructure/k8s/service-accounts.yaml`:

```yaml
# infrastructure/k8s/service-accounts.yaml
# One service account per workload — Vault auth binds to these.
# Each maps to a different Vault policy (different secret access).

apiVersion: v1
kind: ServiceAccount
metadata:
  name: financial-rag-agent-api
  namespace: financial-rag
  labels:
    app.kubernetes.io/name: financial-rag-agent
    app.kubernetes.io/component: api
  annotations:
    # Uncomment after creating IRSA role for S3/ECR access
    # eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/financial-rag-api
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: financial-rag-agent-agent
  namespace: financial-rag
  labels:
    app.kubernetes.io/name: financial-rag-agent
    app.kubernetes.io/component: agent
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: financial-rag-agent-ingestion
  namespace: financial-rag
  labels:
    app.kubernetes.io/name: financial-rag-agent
    app.kubernetes.io/component: ingestion
```

Apply:

```bash
kubectl create namespace financial-rag 2>/dev/null || true
kubectl apply -f infrastructure/k8s/service-accounts.yaml
```

---

## Step 6 — Vault Agent Sidecar Config (Component #65)

The Vault Agent sidecar runs as a second container in each pod. It:
1. Authenticates to Vault using the pod's service account JWT
2. Fetches secrets from the configured paths
3. Renders them into files using Go templates
4. Renews credentials automatically before they expire

Create `infrastructure/vault/vault-agent-config.hcl`:

```hcl
# infrastructure/vault/vault-agent-config.hcl
# Vault Agent sidecar configuration.
# This file is mounted into the vault-agent container via ConfigMap.

vault {
  address = "http://vault.vault.svc.cluster.local:8200"
}

# Authenticate using the pod's Kubernetes service account JWT
auto_auth {
  method "kubernetes" {
    mount_path = "auth/kubernetes"
    config = {
      # Role name must match what was created in vault-bootstrap.sh
      # Set via downward API: VAULT_ROLE env var on the sidecar
      role = "financial-rag-api"
    }
  }

  # Write the Vault token to a file for the app container (optional)
  sink "file" {
    config = {
      path = "/vault/secrets/.vault-token"
      mode = 0600
    }
  }
}

# Render database credentials to a file
template {
  destination = "/vault/secrets/database.env"
  # Go template syntax — Vault Agent fills in the values
  contents = <<-EOT
    {{- with secret "database/creds/api-role" -}}
    POSTGRES_USER={{ .Data.username }}
    POSTGRES_PASSWORD={{ .Data.password }}
    {{- end }}
  EOT
  # Restart the app when credentials rotate
  command = "/bin/sh -c 'kill -HUP $(pgrep -f uvicorn) 2>/dev/null || true'"
}

# Render LLM API key to a file
template {
  destination = "/vault/secrets/llm.env"
  contents = <<-EOT
    {{- with secret "secret/data/financial-rag/prod/llm" -}}
    OPENAI_API_KEY={{ .Data.data.openai_api_key }}
    LLM_MODEL={{ .Data.data.llm_model }}
    LLM_BASE_URL={{ .Data.data.llm_base_url }}
    {{- end }}
  EOT
}

# Render Redis credentials
template {
  destination = "/vault/secrets/redis.env"
  contents = <<-EOT
    {{- with secret "secret/data/financial-rag/prod/redis" -}}
    REDIS_HOST={{ .Data.data.host }}
    REDIS_PORT={{ .Data.data.port }}
    REDIS_PASSWORD={{ .Data.data.password }}
    {{- end }}
  EOT
}
```

### How the sidecar injects secrets into the app pod

```yaml
# Snippet of a Kubernetes deployment using Vault Agent injection
# (annotations trigger the Vault mutating webhook)
spec:
  serviceAccountName: financial-rag-agent-api
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "financial-rag-api"
    vault.hashicorp.com/agent-inject-secret-database: "database/creds/api-role"
    vault.hashicorp.com/agent-inject-template-database: |
      {{- with secret "database/creds/api-role" -}}
      POSTGRES_USER={{ .Data.username }}
      POSTGRES_PASSWORD={{ .Data.password }}
      {{- end }}
```

The annotation approach uses Vault's mutating admission webhook — it
automatically injects the Vault Agent sidecar without changing the
Deployment spec. This is the recommended production pattern.

---

## Step 7 — Deploy and Test the Sidecar (Component #66)

```bash
# Test pod that uses the API service account
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vault-test
  namespace: financial-rag
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "financial-rag-api"
    vault.hashicorp.com/agent-inject-secret-database: "database/creds/api-role"
spec:
  serviceAccountName: financial-rag-agent-api
  containers:
  - name: app
    image: busybox
    command: ["sleep", "3600"]
  restartPolicy: Never
EOF

# Wait for it to start
kubectl wait --for=condition=ready pod/vault-test -n financial-rag --timeout=60s

# Verify secrets were injected as files
kubectl exec -n financial-rag vault-test -- cat /vault/secrets/database.env
```

Expected output:
```
POSTGRES_USER=v-k8s-api-role-AbCdEf123456
POSTGRES_PASSWORD=A1b2C3d4-E5f6-G7h8-I9j0-K1L2M3N4O5P6
```

The username and password are dynamically generated by Vault — unique per
pod, automatically rotated after `default_ttl` (1 hour for api-role).

```bash
# Clean up test pod
kubectl delete pod vault-test -n financial-rag
```

---

## Step 8 — Python Vault Client (Component #67)

Install hvac:

```bash
pip install hvac
```

Create `src/financial_rag/security/vault.py`:

```python
# src/financial_rag/security/vault.py
# =============================================================================
# Vault client for runtime secret retrieval.
#
# Used when Vault Agent sidecar is NOT available (local development).
# In production, the sidecar writes secrets to files — this client
# reads those files directly. Direct Vault API calls are a fallback.
#
# Priority:
#   1. /vault/secrets/*.env files (sidecar-injected, production)
#   2. Direct Vault API via hvac (fallback, development)
#   3. Settings from .env (fallback, local dev)
# =============================================================================

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Vault Agent writes secrets here in Kubernetes
_VAULT_SECRETS_DIR = Path("/vault/secrets")


class VaultSecretReader:
    """
    Reads secrets from Vault Agent-injected files or directly via hvac.

    In production (Kubernetes + Vault Agent sidecar):
        Secrets are written to /vault/secrets/*.env as KEY=VALUE pairs.
        This class reads those files and returns values as a dict.

    In development (no sidecar):
        Falls back to direct Vault API via hvac if VAULT_ADDR is set.
        Falls back to environment variables if Vault is not available.

    Usage:
        reader = VaultSecretReader()
        db_creds = reader.read("database")
        # Returns {"POSTGRES_USER": "v-k8s-...", "POSTGRES_PASSWORD": "..."}
    """

    def read(self, secret_name: str) -> dict[str, str]:
        """
        Read a secret by name.

        Args:
            secret_name: Name without extension (e.g. "database", "llm")
                        Maps to /vault/secrets/{secret_name}.env

        Returns:
            dict of KEY=VALUE pairs from the secret file.
            Empty dict if secret not found (caller uses fallback).
        """
        # Try file-based injection first (production path)
        file_path = _VAULT_SECRETS_DIR / f"{secret_name}.env"
        if file_path.exists():
            logger.debug("Reading secret '%s' from Vault sidecar file", secret_name)
            return self._parse_env_file(file_path)

        # Try direct Vault API (development fallback)
        vault_addr = os.environ.get("VAULT_ADDR")
        vault_token = os.environ.get("VAULT_TOKEN")
        if vault_addr and vault_token:
            return self._read_from_vault_api(secret_name, vault_addr, vault_token)

        logger.debug(
            "Vault not available for secret '%s' — using environment variables",
            secret_name,
        )
        return {}

    def read_database_credentials(self) -> dict[str, str]:
        """
        Read dynamic database credentials.

        Returns POSTGRES_USER and POSTGRES_PASSWORD from the database
        secret (rotated by Vault, TTL 1h).
        """
        creds = self.read("database")
        if creds:
            logger.info(
                "Database credentials loaded from Vault (user=%s)",
                creds.get("POSTGRES_USER", "unknown")[:20] + "...",
            )
        return creds

    def _parse_env_file(self, path: Path) -> dict[str, str]:
        """Parse a KEY=VALUE env file into a dict."""
        result: dict[str, str] = {}
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        except OSError as exc:
            logger.warning("Failed to read Vault secret file %s: %s", path, exc)
        return result

    def _read_from_vault_api(
        self,
        secret_name: str,
        vault_addr: str,
        vault_token: str,
    ) -> dict[str, str]:
        """
        Read a secret directly from Vault via hvac.
        Used in development when running Vault locally.
        """
        try:
            import hvac

            client = hvac.Client(url=vault_addr, token=vault_token)

            if not client.is_authenticated():
                logger.warning("Vault token is invalid or expired")
                return {}

            # Try KV v2 path first
            path = f"financial-rag/prod/{secret_name}"
            try:
                response = client.secrets.kv.v2.read_secret_version(path=path)
                return {
                    k: str(v)
                    for k, v in response["data"]["data"].items()
                }
            except Exception:
                pass

            # Try dynamic database creds
            if secret_name == "database":
                try:
                    response = client.secrets.database.generate_credentials(
                        name="api-role"
                    )
                    return {
                        "POSTGRES_USER": response["data"]["username"],
                        "POSTGRES_PASSWORD": response["data"]["password"],
                    }
                except Exception:
                    pass

            return {}

        except ImportError:
            logger.debug("hvac not installed — skipping Vault API fallback")
            return {}
        except Exception as exc:
            logger.warning("Vault API read failed for '%s': %s", secret_name, exc)
            return {}


# =============================================================================
# Module-level singleton
# =============================================================================

_vault_reader: VaultSecretReader | None = None


def get_vault_reader() -> VaultSecretReader:
    """Return the application-level VaultSecretReader singleton."""
    global _vault_reader
    if _vault_reader is None:
        _vault_reader = VaultSecretReader()
    return _vault_reader
```

Update `src/financial_rag/security/__init__.py`:

```python
from financial_rag.security.vault import VaultSecretReader, get_vault_reader

__all__ = ["VaultSecretReader", "get_vault_reader"]
```

---

## Step 9 — OPA Rego Policy for Vault (Component #74 preview)

Your repo has `policies/rego/vault_policy.rego`. This validates Vault
policy HCL files before they are applied — ensuring no policy grants
wildcards, delete capabilities, or sudo to application roles.

Use the file from your repo as-is. Test it with conftest:

```bash
pip install conftest  # or brew install conftest

# Write a test input
cat > /tmp/test-policy.json <<'EOF'
{
  "policies": [
    {
      "name": "bad-policy",
      "rules": [
        {
          "path": "secret/*",
          "capabilities": ["write", "delete"]
        }
      ]
    }
  ]
}
EOF

conftest test /tmp/test-policy.json \
  --policy policies/rego/vault_policy.rego
```

Expected output:
```
FAIL - /tmp/test-policy.json - financial_rag.vault - Vault policy 'bad-policy'
       grants write on wildcard secret path 'secret/*' — scope to exact paths.
FAIL - /tmp/test-policy.json - financial_rag.vault - Vault policy 'bad-policy'
       grants delete on path 'secret/*' — application roles must not delete secrets.
```

Three rules enforced by the Rego policy:

| Rule | What it catches |
|---|---|
| No wildcard write on `secret/` | Prevents `secret/*` with write |
| No delete on secret paths | Apps should never delete secrets |
| No `sudo` capability | Reserved for admin policies only |
| No `sys/` access from non-admin | Prevents vault management by apps |

---

## Step 10 — Production Vault HA (Component #68)

```bash
# Production Helm values for HA Vault with Raft storage
helm upgrade vault hashicorp/vault \
  --namespace vault \
  --set "server.dev.enabled=false" \
  --set "server.ha.enabled=true" \
  --set "server.ha.replicas=3" \
  --set "server.ha.raft.enabled=true" \
  --set "server.ha.raft.setNodeId=true" \
  --set "server.dataStorage.enabled=true" \
  --set "server.dataStorage.size=10Gi" \
  --set "ui.enabled=true"
```

### AWS KMS Auto-unseal

In production, Vault must be unsealed after every restart. Manual unsealing
with 3 of 5 key shares is operationally painful. AWS KMS auto-unseal
means Vault unseals itself using a KMS key you control:

```hcl
# Add to Helm values as extraConfig
seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
}
```

### Raft Snapshot Backup (Component #69)

```bash
# Manual snapshot (run before upgrades)
vault operator raft snapshot save /backup/vault-$(date +%Y%m%d-%H%M%S).snap

# Automated via Kubernetes CronJob
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vault-backup
  namespace: vault
spec:
  schedule: "0 2 * * *"    # 2am daily
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: vault-backup
          containers:
          - name: backup
            image: hashicorp/vault:latest
            command:
            - /bin/sh
            - -c
            - |
              vault operator raft snapshot save /tmp/vault.snap
              aws s3 cp /tmp/vault.snap \
                s3://your-backup-bucket/vault/$(date +%Y%m%d-%H%M%S).snap
            env:
            - name: VAULT_ADDR
              value: "http://vault.vault.svc.cluster.local:8200"
          restartPolicy: OnFailure
EOF
```

---

## Step 11 — Phase 6 Verification

```bash
# Verify dynamic DB credentials are issued
vault read database/creds/api-role

# Expected output:
# Key                Value
# lease_id           database/creds/api-role/AbCdEfGhIjKlMn
# lease_duration     1h
# username           v-k8s-api-role-AbCdEf123456
# password           A1b2C3d4-...

# Verify the credential actually works in PostgreSQL
PGPASSWORD="<password from above>" psql \
  -h localhost -U "v-k8s-api-role-AbCdEf123456" \
  -d financial_rag -c "SELECT count(*) FROM financial_chunks;"

# Verify policy denies what it should
vault token create -policy=api-policy
# Then try to access a path not in the policy:
vault kv get secret/financial-rag/prod/edgar
# Expected: permission denied
```

---

# PHASE 7 — CI/CD Pipeline

## New Files in This Phase

```
.github/
└── workflows/
    └── ci.yml                     ← Component #70 (GitHub Actions)
infrastructure/
└── docker/
    └── Dockerfile                 ← Component #71 (multi-stage build)
policies/
└── rego/
    └── vault_policy.rego          ← Component #74 (OPA Rego)
```

---

## Step 12 — The CI Pipeline Architecture

Your pipeline has six sequential jobs with a clear dependency graph:

```
secrets-scan ──┐
               ├──→ lint ──→ unit-tests ──→ integration-tests
dependency-scan┘                    │
                                    └──→ build ──→ image-scan
```

Key design decisions:

**`secrets-scan` runs first, blocks everything.** If a secret is committed,
no code runs. The `fetch-depth: 0` checkout gives gitleaks the full git
history — not just the latest commit — so it catches secrets added and
"deleted" in separate commits.

**`concurrency` with `cancel-in-progress: true`.** If you push twice in
quick succession, the first run is cancelled when the second starts. This
prevents queue buildup and wasted minutes.

**Integration tests only on push, not PRs.** PRs run unit tests only
(fast feedback). Full integration tests run on push to `main`/`develop`
(where real services are available).

**`build` only on `main`.** Docker images are only published from the
main branch. Feature branches get lint and unit tests but never push images.

Use `ci.yml` from your repo. Two issues to fix:

### Issue 1 — Redis password in integration tests

The integration test step sets `REDIS_PASSWORD: ""` (empty string) but the
Redis service in the CI workflow has no password configured either. However,
your `Settings` class has `REDIS_PASSWORD: SecretStr = Field(...)` — required
with no default — so an empty string will fail Pydantic validation.

Fix: either add `REDIS_PASSWORD: test-redis-password` to the env block,
or create a `conftest.py` that sets it before settings are loaded:

```yaml
# In integration-tests job env block, change:
REDIS_PASSWORD: ""
# To:
REDIS_PASSWORD: "test-redis-password-ci"
```

And remove `--requirepass` from the Redis service health check command since
CI Redis has no password:

```yaml
# In the redis service, the health check uses:
redis-cli ping
# This works because CI Redis has no password — correct as-is
# But your app will try to auth with the password above — Redis will reject it
# Solution: add requirepass to the CI Redis service too:
services:
  redis:
    image: redis:7-alpine
    options: >-
      --health-cmd "redis-cli -a test-redis-password-ci ping"
    # Add password via command override:
    # Unfortunately GitHub Actions services don't support command override easily
    # Simplest fix: use REDIS_PASSWORD="" and handle in Settings
```

The cleanest fix for the tutorial: add `REDIS_PASSWORD` to the `Settings`
validator so empty string is treated as "no auth" in testing:

```python
# In settings.py _apply_all_defaults_and_validate:
# Add this before the issues check:
if is_test and not self.REDIS_PASSWORD.get_secret_value():
    # Allow empty Redis password in testing (CI uses passwordless Redis)
    pass  # Settings validator already passed since Field(...) was satisfied
```

Actually the cleanest fix is simpler — just set a real password in both
places in the CI YAML:

```yaml
# redis service: add command
services:
  redis:
    image: redis:7-alpine
    options: >-
      --health-cmd "redis-cli -a test-redis-password-ci ping"
    # GitHub Actions doesn't support service command override
    # Use environment variable approach instead
env:
  REDIS_PASSWORD: "test-redis-password-ci"
```

Since GitHub Actions service containers don't support custom commands, the
simplest fix is to accept passwordless Redis in CI and set `REDIS_PASSWORD`
to any non-empty string that the app sends (Redis will ignore it without
`requirepass`):

```yaml
REDIS_PASSWORD: "not-used-in-ci"
```

### Issue 2 — Schema file path mismatch

The integration test applies the schema:

```yaml
- name: Apply schema
  run: |
    psql -h localhost -U finrag -d financial_rag_test \
      -f infrastructure/docker/init/create_schema.sql
```

But in Phase 1 we named the file `01_create_schema.sql`. Fix:

```yaml
-f infrastructure/docker/init/01_create_schema.sql
```

The corrected integration test env block:

```yaml
integration-tests:
  services:
    postgres:
      image: pgvector/pgvector:pg16
      env:
        POSTGRES_USER:     finrag
        POSTGRES_PASSWORD: test-pg-password-32-chars-minimum
        POSTGRES_DB:       financial_rag_test
      ports: ["5432:5432"]
      options: >-
        --health-cmd "pg_isready -U finrag -d financial_rag_test"
        --health-interval 10s
        --health-timeout 5s
        --health-retries 5
    redis:
      image: redis:7-alpine
      ports: ["6379:6379"]
      options: >-
        --health-cmd "redis-cli ping"
        --health-interval 10s
        --health-timeout 5s
        --health-retries 5
  steps:
    - name: Apply schema
      env:
        PGPASSWORD: test-pg-password-32-chars-minimum
      run: |
        psql -h localhost -U finrag -d financial_rag_test \
          -f infrastructure/docker/init/01_create_schema.sql
    - name: Run tests
      env:
        APP_ENV: testing
        POSTGRES_HOST: localhost
        POSTGRES_USER: finrag
        POSTGRES_PASSWORD: test-pg-password-32-chars-minimum
        POSTGRES_DB: financial_rag_test
        REDIS_HOST: localhost
        REDIS_PASSWORD: "not-used-in-ci"
      run: pytest tests/ -v --tb=short -x -m "not slow"
```

---

## Step 13 — Multi-Stage Dockerfile (Component #71)

Use the `Dockerfile` from your repo as-is. Three patterns worth teaching:

### Pattern 1: Layer caching order

```dockerfile
# Copy dependency files FIRST (changes rarely)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies (cached until pyproject.toml changes)
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install -e "." --no-cache-dir

# THEN copy application code (changes frequently)
# If only src/ changes, pip install is skipped from cache
```

In this `Dockerfile`, `src/` is copied before `pip install`. This means
every code change invalidates the pip cache. The optimal order is:

```dockerfile
# Better layer ordering:
COPY pyproject.toml README.md ./
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install -e "." --no-cache-dir
COPY src/ ./src/
```

However, since the package is installed in editable mode (`-e .`),
the source must exist when pip runs. The workaround is to create a
stub `src/financial_rag/__init__.py` before the pip install, then
copy the real source after. For a tutorial, the current approach is
acceptable — just note it as a production optimisation.

### Pattern 2: Non-root user

```dockerfile
RUN groupadd -r finrag && useradd -r -g finrag finrag
RUN chown -R finrag:finrag /app
USER finrag
```

Containers run as root by default. If an attacker escapes the container,
they have root on the host. Running as a non-root user limits the damage.
Most Kubernetes `PodSecurityAdmission` policies require this.

### Pattern 3: HEALTHCHECK in the image

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

This is redundant with the Docker Compose and Kubernetes healthchecks, but
having it in the image means any deployment that forgets to configure a
healthcheck still gets one. Defense in depth.

---

## Step 14 — Container Scanning with Trivy (Component #72)

Trivy scans the built image for CVEs in:
- OS packages (`apt` installed `libpq5`, `curl`)
- Python packages (everything in `/app/venv`)

The workflow uploads results in SARIF format to GitHub Security tab:

```yaml
- name: Upload to GitHub Security
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: trivy-results.sarif
```

`exit-code: "0"` means Trivy reports findings but does not fail the build.
Change to `"1"` to block deployments with CRITICAL vulnerabilities.

Test Trivy locally:

```bash
# Install Trivy
brew install aquasecurity/trivy/trivy  # macOS
# or
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh

# Scan your image
trivy image financial-rag-agent:latest \
  --severity CRITICAL,HIGH \
  --format table
```

---

## Step 15 — Secret Scanning with gitleaks (Component #73)

gitleaks scans your git history for secrets — API keys, passwords,
tokens — that were accidentally committed.

The workflow uses the official gitleaks GitHub Action:

```yaml
- name: Run gitleaks
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

`fetch-depth: 0` is critical — without it, Actions only fetches the latest
commit. gitleaks needs full history to catch secrets that were added in
commit A and "deleted" in commit B (the secret is still in the git object
store and anyone who clones the repo can see it).

Test locally before pushing:

```bash
# Install gitleaks
brew install gitleaks  # macOS

# Scan the current repo
gitleaks detect --source . --verbose

# Scan a specific commit range
gitleaks detect --source . --log-opts="HEAD~10..HEAD"
```

Add a `.gitleaks.toml` to allowlist false positives:

```toml
# .gitleaks.toml
[allowlist]
description = "Global allowlist"
regexes = [
  '''test-pg-password-32-chars-minimum''',  # CI test passwords in YAML
  '''test-redis-password''',
  '''REPLACE_ME''',                          # Vault seed placeholders
]
paths = [
  '''.env.example''',                        # Example file with placeholders
]
```

---

## Step 16 — OPA Rego Policy (Component #74)

Use `policies/rego/vault_policy.rego` from your repo as-is.

The four rules it enforces are already documented in Step 9. One
additional note: the `warn` rule for missing TTL:

```rego
warn contains msg if {
    policy := input.policies[_]
    not policy.token_ttl
    not policy.token_max_ttl
    msg := sprintf("Vault policy '%v' has no token_ttl...", [policy.name])
}
```

`warn` produces output but does not fail `conftest test`. `deny` fails the
check. Use `warn` for things that are bad practice but not blocking;
use `deny` for things that must never reach production.

Run the full policy suite in CI by adding this step to your workflow:

```yaml
# Add to lint job after mypy:
- name: OPA policy check
  run: |
    # Install conftest
    curl -L https://github.com/open-policy-agent/conftest/releases/download/v0.50.0/conftest_0.50.0_Linux_x86_64.tar.gz \
      | tar xz -C /usr/local/bin
    # Run against any policy test fixtures
    conftest test policies/test_fixtures/ \
      --policy policies/rego/ || true
```

---

## Step 17 — Production Docker Compose

Your repo has `infrastructure/docker/docker-compose.prod.yml`. Create
the missing Prometheus config it references:

```yaml
# infrastructure/docker/prometheus/prometheus.yml
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "financial-rag-api"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
```

Start the full production stack:

```bash
# Create .env.prod
cp .env.example .env.prod
# Edit .env.prod with production values

# Start
docker compose -f infrastructure/docker/docker-compose.prod.yml \
  --env-file .env.prod up -d

# Verify all services
docker compose -f infrastructure/docker/docker-compose.prod.yml ps
```

---

## Step 18 — Utility Scripts

The three shell scripts from your repo are already complete. Wire them in:

```bash
# Make executable
chmod +x scripts/start.sh scripts/stop.sh scripts/cleanup.sh

# Development workflow
./scripts/start.sh      # starts postgres + redis, waits for healthy
./scripts/stop.sh       # stops all services
./scripts/cleanup.sh    # stops + removes volumes (destructive)
```

---

## Final File Tree — Phases 6 & 7

```
financial-rag-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                 ← Phase 7 (with fixes)
├── .gitleaks.toml                 ← Phase 7
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile             ← Phase 7
│   │   ├── docker-compose.prod.yml← Phase 7
│   │   ├── init/
│   │   │   └── 01_create_schema.sql
│   │   └── prometheus/
│   │       └── prometheus.yml     ← Phase 7 (new)
│   ├── vault/
│   │   ├── vault-bootstrap.sh     ← Phase 6
│   │   ├── vault-agent-config.hcl ← Phase 6
│   │   └── policies/
│   │       ├── api-policy.hcl     ← Phase 6
│   │       ├── agent-policy.hcl   ← Phase 6
│   │       └── ingestion-policy.hcl ← Phase 6
│   └── k8s/
│       └── service-accounts.yaml  ← Phase 6
├── policies/
│   └── rego/
│       └── vault_policy.rego      ← Phase 7
├── scripts/
│   ├── start.sh
│   ├── stop.sh
│   └── cleanup.sh
└── src/financial_rag/security/
    ├── __init__.py
    └── vault.py                   ← Phase 6
```

---

## Common Errors and Fixes — Phases 6 & 7

| Error | Cause | Fix |
|---|---|---|
| `permission denied` on vault-bootstrap.sh | File not executable | `chmod +x infrastructure/vault/vault-bootstrap.sh` |
| `Error initializing listener: listen tcp: bind: permission denied` | Port 8200 in use | `lsof -i :8200` and kill the process |
| `connection refused` to Vault in CI | Vault not installed in CI | Vault config is for K8s only; CI uses `.env` secrets |
| `POSTGRES_PASSWORD required` in unit tests | Settings requires password | Set `POSTGRES_PASSWORD=test-pg-password` in CI env |
| gitleaks fails on `.env.example` | Placeholder looks like real key | Add to `[allowlist]` in `.gitleaks.toml` |
| Trivy not finding the image | Image not built locally | `docker build -f infrastructure/docker/Dockerfile -t financial-rag-agent .` first |
| `conftest: command not found` | Not installed | `brew install conftest` or download binary |
| Schema not found in CI | Path mismatch | Change to `01_create_schema.sql` in ci.yml |
| Dynamic creds expire mid-request | TTL too short or renewal missed | Vault Agent handles renewal — ensure sidecar is running |

---

## What's Next — Phase 8

Phase 8 moves to full Kubernetes deployment:
- Helm chart structure with `_helpers.tpl`
- StatefulSets for PostgreSQL and Redis with persistent volumes
- HorizontalPodAutoscaler for the API (CPU + custom metrics)
- Karpenter for node provisioning — spot instances for ingestion workloads
- AWS ALB Ingress with SSL termination
- Terragrunt modules for VPC, EKS, RDS, ElastiCache
- Multi-environment config: dev, staging, production
