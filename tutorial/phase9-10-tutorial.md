# Financial RAG Agent — Phase 9 & 10: GitOps, eBPF Networking & Runtime Security

> **Series:** Financial RAG Agent (freeCodeCamp)  
> **Phase 9:** ArgoCD GitOps & Cilium eBPF Networking — Components #101–119  
> **Phase 10:** Falco Runtime Security & Threat Detection — Components #120–150  
> **Time:** Phase 9 ~3 hours | Phase 10 ~2 hours  
> **Prerequisite:** Phase 8 complete — EKS cluster running, Helm chart deployed

---

## What You Will Build

**Phase 9** — GitOps and zero-trust networking:
- ArgoCD installed with bootstrap script
- App of Apps pattern — one root ApplicationSet manages all environments
- Three-environment promotion pipeline: dev → staging → prod
- ArgoCD Project with RBAC roles and sync windows (no prod deploys during business hours)
- Cilium replacing kube-proxy with eBPF
- Deny-all NetworkPolicy with DNS egress exception
- L7 HTTP policies — Cilium enforces allowed HTTP methods and paths at the kernel level
- Istio service mesh for mTLS between services
- Hubble and Cilium ServiceMonitors for Prometheus

**Phase 10** — Runtime threat detection:
- Falco installed with eBPF driver (no kernel module needed on EKS)
- Eight custom rules covering the financial RAG threat model
- Falcosidekick routing alerts to Slack, PagerDuty, and CloudWatch
- Prometheus metrics for Falco event rates
- ConfigMap-based rule deployment (hot-reloadable without pod restart)
- PrometheusRule alerting on Falco critical events

---

# PHASE 9 — ArgoCD GitOps & Cilium eBPF Networking

## New Files in This Phase

```
argocd/
├── bootstrap/
│   └── bootstrap.sh               ← Component #108–110
├── projects/
│   └── financial-rag-project.yaml ← ArgoCD Project (RBAC + sync windows)
├── appsets/
│   ├── apps-appset.yaml           ← Component #111 (App of Apps)
│   └── env-appset.yaml            ← environment config sync
cilium/
├── network-policies/
│   ├── default-deny-all.yaml      ← Component #104
│   ├── api-l7-policy.yaml         ← Component #107
│   └── agent-l7-policy.yaml
├── ebpf/
│   └── cilium-servicemonitor.yaml ← Component #118
istio-mesh/
├── authorization-policies.yaml
├── destination-rules.yaml
├── gateway.yaml
└── kustomization.yaml
```

---

## Step 1 — Understand the GitOps Architecture

Before writing any files, understand the flow:

```
Developer pushes to GitHub
        ↓
ArgoCD detects diff (polls every 3 minutes or webhook)
        ↓
ApplicationSet generates Application per environment
        ↓
dev: auto-sync immediately
staging: auto-sync immediately
prod: manual sync only, allowed window: 02:00–06:00 UTC
        ↓
Helm renders templates with environment-specific values
        ↓
kubectl apply (server-side apply) → K8s API
        ↓
Slack notification: sync succeeded / failed
```

**Why GitOps?** Without GitOps, deployments happen via `helm upgrade` from
someone's laptop or a CI runner. The cluster state is not reproducible from
Git alone — if the cluster is destroyed, you don't know exactly what was
running. With ArgoCD, the cluster continuously reconciles toward the Git
state. The Git repo IS the source of truth.

---

## Step 2 — Install Cilium (Component #101)

Cilium must be installed **before** ArgoCD because ArgoCD pods need
networking to start. Install it directly with Helm, not through ArgoCD
(bootstrapping problem — you need the network to sync the network).

```bash
helm repo add cilium https://helm.cilium.io/
helm repo update

# Install Cilium with kube-proxy replacement
# This replaces kube-proxy entirely — do this on a fresh cluster
helm install cilium cilium/cilium \
  --version 1.15.5 \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=<YOUR_EKS_API_ENDPOINT> \
  --set k8sServicePort=443 \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true \
  --set hubble.metrics.enableOpenMetrics=true \
  --set hubble.metrics.enabled="{dns,drop,tcp,flow,port-distribution,icmp,httpV2:exemplars=true;labelsContext=source_ip\,source_namespace\,source_workload\,destination_ip\,destination_namespace\,destination_workload\,traffic_direction}" \
  --set ipam.mode=eni \
  --set eni.enabled=true \
  --set operator.replicas=2

# Verify installation
cilium status --wait

# Verify Hubble
cilium hubble port-forward &
hubble status
```

> **`kubeProxyReplacement=true`** replaces kube-proxy with Cilium's eBPF
> datapath. This gives you L4 load balancing in the kernel — no userspace
> iptables chains. On EKS, use `ipam.mode=eni` so Cilium manages pod IPs
> directly on the EC2 ENI — no overlay network, native VPC routing.

---

## Step 3 — Verify Cilium (Component #102)

```bash
# Check all Cilium agents are running
kubectl get pods -n kube-system -l app.kubernetes.io/name=cilium-agent

# Run the connectivity test (deploys test pods and verifies L3/L4/L7)
cilium connectivity test

# Hubble CLI — observe live network flows
hubble observe --namespace financial-rag --follow

# Hubble UI (browser)
cilium hubble ui
```

---

## Step 4 — Deny-All Network Policy (Component #104)

Apply the baseline policy first — before the allow rules. This ensures
no traffic flows until explicitly permitted.

Use `cilium/network-policies/default-deny-all.yaml` from your repo as-is.

```yaml
# cilium/network-policies/default-deny-all.yaml
apiVersion: "cilium.io/v2"
kind: CiliumNetworkPolicy
metadata:
  name: default-deny-all
  namespace: financial-rag
spec:
  endpointSelector: {}    # matches ALL pods in the namespace
  ingress: []             # deny all ingress
  egress:
    # Allow DNS only — pods need to resolve service names
    - toEndpoints:
        - matchLabels:
            k8s:io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP
            - port: "53"
              protocol: TCP
```

> **Why allow DNS in the deny-all?** Without DNS, pods cannot resolve
> `financial-rag-agent-pgvector.financial-rag.svc.cluster.local` and
> will fail to connect to any service. DNS is the one egress that must
> be permitted before any other allow rules are applied.

```bash
kubectl apply -f cilium/network-policies/default-deny-all.yaml

# Verify — all traffic should now be dropped except DNS
hubble observe --namespace financial-rag --verdict DROPPED
```

---

## Step 5 — L7 HTTP Policies (Components #105–107)

Cilium's L7 policies are enforced at the eBPF level — no sidecar proxy
required. The kernel itself inspects HTTP method and path and drops
non-matching requests before they reach the application.

Use `cilium/network-policies/api-l7-policy.yaml` and
`agent-l7-policy.yaml` from your repo as-is.

Key points for your audience:

### What L7 enforcement means in practice

Without L7 policy:
```
curl -X DELETE http://api/filings/abc123   # reaches FastAPI, returns 405
curl -X POST http://api/admin/reset        # reaches FastAPI, your app must deny
```

With L7 policy (`method: POST path: /query` only):
```
curl -X DELETE http://api/filings/abc123   # dropped at kernel, never reaches app
curl -X POST http://api/admin/reset        # dropped at kernel, never reaches app
```

The application never sees the forbidden request. Even if FastAPI had a bug
that incorrectly handled the route, the kernel already dropped it.

### The API L7 policy allows:

| Source | Method | Path |
|---|---|---|
| world (ALB) | GET | /health |
| world (ALB) | POST | /query, /v1/query |
| world (ALB) | GET | /metrics |
| monitoring namespace | GET | /metrics |

Everything else is dropped. No `DELETE`, no `PUT`, no `/docs` in production
(matches your `server.py` which disables `/docs` when `DEBUG=False`).

Apply:

```bash
kubectl apply -f cilium/network-policies/api-l7-policy.yaml
kubectl apply -f cilium/network-policies/agent-l7-policy.yaml
```

Verify with Hubble:

```bash
# Test that /query works
curl -X POST https://api.financial-rag.cloudfrugal.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Test that DELETE is blocked (should get connection reset, not 405)
curl -X DELETE https://api.financial-rag.cloudfrugal.com/query
# Expected: connection reset (Cilium drops at kernel, no HTTP response)

# Watch Hubble
hubble observe --namespace financial-rag --verdict DROPPED --follow
```

---

## Step 6 — Istio Service Mesh

Your repo includes Istio mTLS configuration alongside Cilium. This is a
layered security model:

```
Cilium L3/L4/L7  →  enforces which pods can talk to which pods, HTTP method/path
Istio mTLS       →  encrypts all pod-to-pod traffic, enforces service identity
```

Install Istio:

```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

kubectl create namespace istio-system

# Base CRDs
helm install istio-base istio/base -n istio-system --set defaultRevision=default

# Control plane
helm install istiod istio/istiod -n istio-system --wait

# Enable sidecar injection for the financial-rag namespace
kubectl label namespace financial-rag istio-injection=enabled
```

Apply the Istio resources from your repo:

```bash
# PeerAuthentication — enforce strict mTLS (reject plaintext)
kubectl apply -f istio-mesh/peer-auth-strict-mtls.yaml

# AuthorizationPolicies — deny all, then allow specific principals
kubectl apply -f istio-mesh/authorization-policies.yaml

# DestinationRules — connection pool, outlier detection, load balancing
kubectl apply -f istio-mesh/destination-rules.yaml

# Gateway + VirtualService — TLS termination at the mesh edge
kubectl apply -f istio-mesh/gateway.yaml
```

The `authorization-policies.yaml` from your repo implements four policies:

```yaml
# 1. deny-all — namespace-wide baseline
spec: {}   # empty spec = deny everything

# 2. allow-gateway-to-api — only istio-ingressgateway can reach the API
source.principals: ["cluster.local/ns/istio-system/sa/istio-ingressgateway"]

# 3. allow-api-to-agent — only the API service account can call the agent
source.principals: ["cluster.local/ns/financial-rag/sa/financial-rag-prod-api"]

# 4. allow-prometheus-scrape — monitoring namespace can GET /metrics
source.principals: ["cluster.local/ns/monitoring/sa/kube-prometheus-stack-prometheus"]
```

The `destination-rules.yaml` sets connection pool limits and outlier
detection per service — if a pgvector pod starts returning errors,
Istio ejects it from the load balancer for 30 seconds.

---

## Step 7 — Install ArgoCD (Components #108–110)

Use the bootstrap script from your repo. It automates:
1. Helm install of ArgoCD
2. Wait for rollout
3. Retrieve initial admin password
4. Port-forward and CLI login
5. Add the Git repository
6. Apply RBAC config
7. Apply notification secrets
8. Apply all ApplicationSets

```bash
chmod +x argocd/bootstrap/bootstrap.sh

# Set required environment variables
export GITHUB_TOKEN="ghp_..."           # PAT with repo read access
export SLACK_TOKEN="xoxb-..."           # optional
export PAGERDUTY_PROD_KEY="..."         # optional

./argocd/bootstrap/bootstrap.sh
```

After bootstrap, verify:

```bash
kubectl get applicationsets -n argocd
kubectl get applications -n argocd
```

Expected:
```
NAME                          READY
apps-appset                   True
env-appset                    True

NAME                          HEALTH    SYNC
financial-rag-dev             Healthy   Synced
financial-rag-staging         Healthy   Synced
financial-rag-prod            Healthy   OutOfSync   ← manual sync required
```

---

## Step 8 — ArgoCD Project (RBAC + Sync Windows)

Use `argocd/projects/financial-rag-project.yaml` from your repo as-is.

Three things worth teaching from this file:

### Sync Windows

```yaml
syncWindows:
  # Dev and staging: auto-sync anytime
  - kind: allow
    schedule: "* * * * *"
    duration: 24h
    applications: ["*-dev", "*-staging"]

  # Prod: sync only 02:00–06:00 UTC
  - kind: allow
    schedule: "0 2 * * *"
    duration: 4h
    applications: ["*-prod"]
    manualSync: true

  # Prod: DENY during business hours (08:00–18:00 UTC Mon–Fri)
  - kind: deny
    schedule: "0 8 * * 1-5"
    duration: 10h
    applications: ["*-prod"]
    manualSync: false
```

The deny window during business hours prevents someone accidentally triggering
a prod deployment during peak traffic. Even manual syncs are blocked. The
only window for prod changes is 02:00–06:00 UTC — low-traffic, pre-market.

### Three-tier RBAC

```yaml
roles:
  - name: platform-engineer    # full sync, all environments
  - name: developer            # get + sync dev/staging only, no prod
  - name: ci-bot               # get + sync all (used by GitHub Actions)
```

The CI bot can sync all environments but cannot modify the project
configuration, RBAC policies, or sync windows.

### App of Apps: the `apps-appset.yaml`

```yaml
generators:
  - list:
      elements:
        - environment: dev
          autoSync: "true"
          imageTag: "latest"
        - environment: staging
          autoSync: "true"
          imageTag: "latest"
        - environment: prod
          autoSync: "false"    # manual only
          imageTag: "1.0.0"   # pinned tag
```

One ApplicationSet generates three Applications — one per environment.
The `template` section is identical for all three; only the generator
variables differ. This is the DRY principle applied to GitOps.

---

## Step 9 — Promote to Production

With ArgoCD configured:

```bash
# Check what's out of sync in prod
argocd app diff financial-rag-prod

# Sync prod manually (within the allowed window: 02:00–06:00 UTC)
argocd app sync financial-rag-prod --prune

# Watch the rollout
argocd app wait financial-rag-prod --health

# Rollback if needed
argocd app rollback financial-rag-prod 1  # roll back to revision 1
```

---

## Step 10 — Cilium ServiceMonitor (Component #118)

Use `cilium/ebpf/cilium-servicemonitor.yaml` from your repo as-is.

After applying, verify metrics are flowing:

```bash
kubectl apply -f cilium/ebpf/cilium-servicemonitor.yaml

# Check Prometheus is scraping Hubble
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090 &
# Open http://localhost:9090/targets — look for cilium-agent and cilium-operator
```

---

## Phase 9 Verification

```bash
# All Cilium agents healthy
kubectl get pods -n kube-system -l app.kubernetes.io/name=cilium-agent

# Network connectivity test
cilium connectivity test --test-namespace financial-rag

# ArgoCD all apps synced
argocd app list

# Hubble flows (no unexpected drops)
hubble observe --namespace financial-rag --verdict DROPPED --last 100

# Istio mTLS — all traffic encrypted
istioctl x describe service financial-rag-agent-api.financial-rag
```

---

# PHASE 10 — Falco Runtime Security & Threat Detection

## New Files in This Phase

```
falco/
├── config/
│   ├── falco.yaml                 ← Falco daemon config
│   ├── falco-helm-values.yaml     ← Helm values (Components #121)
│   └── falco-financial-rag-configmap.yaml ← Component #132 (rules ConfigMap)
├── rules/
│   └── financial_rag_rules.yaml  ← Components #125–131
└── sidekick/
    └── falcosidekick-values.yaml  ← Components #136–138
```

---

## Step 11 — The Falco Threat Model

Before installing anything, understand what Falco detects and why it
matters for a financial RAG agent specifically.

Your application runs LLM inference, reads sensitive financial data from
SEC filings, and holds database credentials in Vault-injected files. The
realistic attack scenarios are:

| Attack | Detection |
|---|---|
| Attacker gets RCE via prompt injection, spawns a shell | Shell Spawned in Financial RAG Pod |
| Malware reads `/vault/secrets/database.env` | Vault Secret Read by Unexpected Process |
| Compromised agent pod connects to attacker's C2 server | Unexpected Outbound Connection |
| Attacker uses `kubectl exec` to enter a pod | kubectl exec in Financial RAG Namespace |
| Cryptojacking — attacker runs mining software | Crypto Mining in Financial RAG |
| Container escape — writes to read-only filesystem | Write Outside Allowed Paths |
| Privilege escalation via setuid binary | Privilege Escalation in Financial RAG Pod |
| Credential harvesting via `/proc/*/environ` | Read Process Environment |

Falco detects these at the **syscall layer** — below the application,
below the container runtime. Even if the attacker has full control of
the Python process, Falco sees the underlying `open()`, `connect()`,
and `execve()` syscalls.

---

## Step 12 — Install Falco (Component #121)

```bash
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm repo update

# Create namespace
kubectl create namespace falco

# Install with eBPF driver and Falcosidekick
helm upgrade --install falco falcosecurity/falco \
  --namespace kube-system \
  -f falco/config/falco-helm-values.yaml \
  --set falcosidekick.config.slack.webhookurl="${SLACK_SECURITY_WEBHOOK}" \
  --set falcosidekick.config.pagerduty.routingkey="${PAGERDUTY_KEY}"
```

> **Why `kube-system` not `falco` namespace?** Falco runs as a DaemonSet
> that must access the host's eBPF subsystem. Running it in `kube-system`
> ensures it starts early and has the required privileges. The Helm chart
> creates its own service account with the minimum permissions needed.

Verify:

```bash
# Check Falco pods are running on all nodes
kubectl get pods -n kube-system -l app.kubernetes.io/name=falco

# Check eBPF probe loaded
kubectl exec -n kube-system \
  $(kubectl get pods -n kube-system -l app.kubernetes.io/name=falco -o name | head -1) \
  -- ls -la /sys/fs/bpf/
# Expected: falco directory exists in BPF filesystem

# List loaded rules
kubectl exec -n kube-system \
  $(kubectl get pods -n kube-system -l app.kubernetes.io/name=falco -o name | head -1) \
  -- falco -l 2>/dev/null | grep "financial-rag"
```

---

## Step 13 — Custom Rules (Components #125–131)

Use `falco/rules/financial_rag_rules.yaml` from your repo as-is.

Eight rules covering the financial RAG threat model. Walk through each
with your freeCodeCamp audience:

### Rule anatomy

Every Falco rule has five required fields:

```yaml
- rule: Rule Name
  desc: Human-readable description
  condition: >         # boolean expression using Falco fields
    spawned_process and
    financial_rag_namespace and
    shell_procs
  output: >            # what to log when condition is true
    Shell spawned (user=%user.name command=%proc.cmdline pod=%k8s.pod.name)
  priority: CRITICAL   # DEBUG, INFO, NOTICE, WARNING, ERROR, CRITICAL, ALERT, EMERGENCY
  tags: [financial-rag, shell, T1059]  # MITRE ATT&CK tag
```

### Rule 1 — Shell Spawned (Component #125)

```yaml
condition: >
  spawned_process and
  financial_rag_namespace and
  shell_procs and
  not vault_agent_container and
  not istio_proxy_container
```

`shell_procs` is a built-in Falco macro that matches `bash`, `sh`, `zsh`,
`dash`, `fish`. The `not` clauses prevent false positives from legitimate
containers — Vault Agent and Istio proxy do occasionally spawn subprocesses.

### Rule 2 — Unexpected Outbound Connection (Component #126)

```yaml
condition: >
  outbound and
  agent_container and
  not fd.sport in (5432, 6379, 443, 15001, 15006) and
  not fd.sip = "127.0.0.1"
```

Ports `15001` and `15006` are Istio's iptables intercept ports — traffic
the Envoy sidecar handles internally. Without this exception, every
legitimate network call would trigger the rule.

This rule catches **Cilium policy bypass attempts**. If an attacker finds
a way to bypass the L3/L4 NetworkPolicy, they still can't make outbound
connections without Falco alerting.

### Rule 4 — Vault Secret Read by Unexpected Process (Component #127)

```yaml
condition: >
  open_read and
  financial_rag_namespace and
  fd.name startswith "/vault/secrets/" and
  not proc.name in (python3, python, uvicorn, gunicorn, vault) and
  not vault_agent_container
```

This rule implements the detection side of your Vault security model.
If an attacker compromises the container and runs a shell to `cat
/vault/secrets/database.env`, Falco fires before they can use the credential.

### Rule 7 — kubectl exec (Component #125 audit)

```yaml
condition: >
  ka.verb = "create" and
  ka.target.resource = "pods/exec" and
  ka.target.namespace = "financial-rag"
priority: WARNING  # not CRITICAL — exec is allowed, just logged
```

Priority is `WARNING` not `CRITICAL` — kubectl exec is permitted via
approved runbook with MFA. The rule creates an audit trail for SOC2
compliance (CC6.1 requires logging all privileged access). Every exec
session is logged with the user, pod, and command.

---

## Step 14 — Deploy Rules as ConfigMap (Component #132)

```bash
# Create the ConfigMap from the rules file
kubectl create configmap falco-financial-rag-rules \
  --namespace kube-system \
  --from-file=financial_rag_rules.yaml=falco/rules/financial_rag_rules.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

The Helm values mount this ConfigMap into Falco's rules directory:

```yaml
# From falco-helm-values.yaml
extraVolumes:
  - name: financial-rag-rules
    configMap:
      name: falco-financial-rag-rules
extraVolumeMounts:
  - mountPath: /etc/falco/rules.d
    name: financial-rag-rules
```

To update rules without restarting Falco (hot reload):

```bash
# Update the ConfigMap
kubectl create configmap falco-financial-rag-rules \
  --namespace kube-system \
  --from-file=financial_rag_rules.yaml=falco/rules/financial_rag_rules.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# Send SIGHUP to Falco to reload rules
kubectl exec -n kube-system \
  $(kubectl get pods -n kube-system -l app.kubernetes.io/name=falco -o name | head -1) \
  -- kill -1 1
```

---

## Step 15 — Test Rules (Component #134)

Trigger each rule deliberately in a non-production namespace to confirm
they fire correctly:

```bash
# Test 1: Shell spawn detection
kubectl run test-shell --image=alpine --namespace=financial-rag \
  --labels="app.kubernetes.io/component=api" \
  --restart=Never -- sleep 600 &

kubectl exec -n financial-rag test-shell -- /bin/sh -c "echo test"

# Expected Falco alert: Shell Spawned in Financial RAG Pod (CRITICAL)
kubectl logs -n kube-system \
  $(kubectl get pods -n kube-system -l app.kubernetes.io/name=falco -o name | head -1) \
  | grep "Shell Spawned"

# Test 2: Vault secret read
kubectl exec -n financial-rag test-shell -- \
  cat /vault/secrets/database.env 2>/dev/null || true

# Expected: Vault Secret Read by Unexpected Process (CRITICAL)

# Test 3: Outbound connection to unexpected port
kubectl exec -n financial-rag test-shell -- \
  wget -q --timeout=3 http://example.com:8080 || true

# Expected: Unexpected Outbound Connection (HIGH)

# Clean up
kubectl delete pod test-shell -n financial-rag
```

---

## Step 16 — Falcosidekick (Components #136–138)

Falcosidekick routes Falco alerts to external destinations.
Use `falco/sidekick/falcosidekick-values.yaml` from your repo as-is.

Three destinations configured:

**Slack** (WARNING and above):
```yaml
slack:
  webhookurl: ""          # injected via --set at install time
  minimumpriority: "warning"
  messageformat: |
    *{{ .Rule }}* | Priority: {{ .Priority }} | Pod: {{ .OutputFields.k8s_pod_name }}
```

**PagerDuty** (CRITICAL only — wakes someone up):
```yaml
pagerduty:
  routingkey: ""          # injected via --set
  minimumpriority: "critical"
```

**CloudWatch** (all events — SOC2 audit trail):
```yaml
aws:
  cloudwatchlogs:
    loggroup: "/aws/falco/financial-rag"
    minimumpriority: "warning"
```

Two replicas with pod anti-affinity ensures alert delivery even if one node fails:

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: kubernetes.io/hostname
```

Install Falcosidekick:

```bash
helm upgrade --install falco falcosecurity/falco \
  --namespace kube-system \
  -f falco/config/falco-helm-values.yaml \
  -f falco/sidekick/falcosidekick-values.yaml \
  --set falcosidekick.config.slack.webhookurl="${SLACK_SECURITY_WEBHOOK}" \
  --set falcosidekick.config.pagerduty.routingkey="${PAGERDUTY_KEY}"
```

Verify Falcosidekick is receiving events:

```bash
kubectl port-forward svc/falco-falcosidekick-ui -n kube-system 2802:2802 &
# Open http://localhost:2802
```

---

## Step 17 — Prometheus Integration (Components #144–146)

Falco exposes metrics at `/metrics` which the Helm values enable via
`serviceMonitor.enabled: true`. The PrometheusRule from your repo adds
alerting on top of these metrics.

Apply the Prometheus alerts:

```bash
kubectl apply -f observability/prometheus-rules/financial-rag-alerts.yaml
```

Three security alerts in the PrometheusRule:

```yaml
# Alert immediately on any CRITICAL Falco event
- alert: FalcoCriticalEventDetected
  expr: sum(rate(falco_events_total{namespace="financial-rag",priority="Critical"}[5m])) > 0
  for: 0m    # fire immediately — no waiting period for critical security events

# Alert on high Cilium drop rate (possible lateral movement attempt)
- alert: CiliumDropRateHigh
  expr: sum(rate(hubble_drop_total{namespace="financial-rag"}[5m])) > 10
  for: 2m

# Alert on Vault token renewal failure
- alert: VaultTokenRenewalFailure
  expr: sum(rate(financial_rag_vault_token_renewal_errors_total[5m])) > 0
  for: 1m    # if token renewal fails for 1 minute, secrets will expire
```

Note `for: 0m` on the Falco critical alert — this fires immediately with
zero stabilisation period. Security events at CRITICAL priority (shell
spawned, vault credential stolen, crypto miner) require immediate response,
not a 5-minute wait to confirm the alert.

---

## Step 18 — Gitleaks Standalone Workflow

Your repo also has a standalone `gitleaks.yml` workflow that runs separately
from the main CI pipeline. Place it at `.github/workflows/gitleaks.yml`.

Two important features beyond the basic gitleaks check in `ci.yml`:

**Weekly full-history scan:**
```yaml
schedule:
  - cron: "0 3 * * 1"    # Monday 03:00 UTC
```
The scheduled run uses `fetch-depth: 0` (full history) to catch secrets
buried in old commits — not just the latest 50. This is how you find
secrets that were "deleted" six months ago but are still in the git object
store.

**PR comment on detection:**
```yaml
- name: Post PR comment on secret detection
  if: failure() && github.event_name == 'pull_request'
```
When a secret is found in a PR, a comment is posted with exact remediation
steps: rotate the credential, use `git-filter-repo` to remove from history.
Developers get actionable instructions in the PR, not just a failed check.

**Slack alert:**
```yaml
- name: Alert security channel on secret detection
  if: failure()
  run: curl -s -X POST $SLACK_WEBHOOK ...
```
The security channel gets notified immediately — not just the developer
who pushed. This ensures the security team sees every detection even if
the developer dismisses the PR check.

---

## Final File Tree — Phases 9 & 10

```
financial-rag-agent/
├── .github/
│   └── workflows/
│       └── gitleaks.yml               ← Phase 10
├── argocd/
│   ├── bootstrap/
│   │   └── bootstrap.sh               ← Phase 9
│   ├── projects/
│   │   └── financial-rag-project.yaml ← Phase 9
│   └── appsets/
│       ├── apps-appset.yaml           ← Phase 9
│       └── env-appset.yaml            ← Phase 9
├── cilium/
│   ├── network-policies/
│   │   ├── default-deny-all.yaml      ← Phase 9
│   │   ├── api-l7-policy.yaml         ← Phase 9
│   │   └── agent-l7-policy.yaml       ← Phase 9
│   └── ebpf/
│       └── cilium-servicemonitor.yaml ← Phase 9
├── istio-mesh/
│   ├── authorization-policies.yaml    ← Phase 9
│   ├── destination-rules.yaml         ← Phase 9
│   ├── gateway.yaml                   ← Phase 9
│   └── kustomization.yaml             ← Phase 9
├── falco/
│   ├── config/
│   │   ├── falco.yaml                 ← Phase 10
│   │   ├── falco-helm-values.yaml     ← Phase 10
│   │   └── falco-financial-rag-configmap.yaml ← Phase 10
│   ├── rules/
│   │   └── financial_rag_rules.yaml  ← Phase 10
│   └── sidekick/
│       └── falcosidekick-values.yaml  ← Phase 10
└── observability/
    └── prometheus-rules/
        └── financial-rag-alerts.yaml  ← Phase 10
```

---

## Common Errors and Fixes — Phases 9 & 10

| Error | Cause | Fix |
|---|---|---|
| `cilium connectivity test` fails | kube-proxy still running | Delete kube-proxy DaemonSet: `kubectl delete ds kube-proxy -n kube-system` |
| ArgoCD pods stuck `Pending` | Cilium not ready when ArgoCD installs | Apply `default-deny-all.yaml` after ArgoCD is running, not before |
| `OutOfSync` never resolves | Ignored differences not set | Add HPA `/spec/replicas` to `ignoreDifferences` in ApplicationSet |
| Falco `permission denied` on eBPF | EKS managed node missing BPF capability | Use `driver.kind: module` on older EKS AMIs, or upgrade to AL2023 |
| Rules not loading | ConfigMap not mounted | Check `extraVolumeMounts` path matches `rules_file` in `falco.yaml` |
| Falcosidekick not routing to Slack | Webhook URL not set | `--set falcosidekick.config.slack.webhookurl=$SLACK_WEBHOOK` |
| L7 policy blocks health check | Path mismatch | Verify `/health` in `api-l7-policy.yaml` ingress rules |
| ArgoCD sync window blocks prod | Correct behaviour | Use `argocd app sync --force` only during allowed window |
| `istioctl` shows plaintext traffic | PeerAuthentication not applied | Check `peer-auth-strict-mtls.yaml` is applied in `financial-rag` namespace |
| Gitleaks false positive | Test passwords in YAML | Add pattern to `[allowlist]` in `.gitleaks.toml` |

---

## What's Next — Phase 11

Phase 11 builds the complete LGTM observability stack:
- **Loki** for log aggregation (S3 backend)
- **Tempo** for distributed tracing
- **Mimir** for long-term metrics storage
- **Grafana** dashboards: RAG performance, LLM cost, error budgets
- OpenTelemetry auto-instrumentation for FastAPI, SQLAlchemy, Redis
- Custom spans for embedding, search, and LLM calls
- SLO alerting with error budget burn rate rules
- Trace-to-log correlation in Grafana
