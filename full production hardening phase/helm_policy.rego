# =============================================================================
# policies/rego/helm_policy.rego
# Conftest policies for Helm-rendered Kubernetes manifests.
# Run in CI: helm template . -f values.yaml -f values.prod.yaml | conftest test -
# =============================================================================
package financial_rag.helm

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# Deny: Deprecated Kubernetes APIs (EKS 1.29 removals)
# ---------------------------------------------------------------------------
removed_apis := {
    "flowcontrol.apiserver.k8s.io/v1beta1",
    "flowcontrol.apiserver.k8s.io/v1beta2",
    "autoscaling/v2beta1",
    "autoscaling/v2beta2",
    "batch/v1beta1",
}

deny contains msg if {
    input.apiVersion
    removed_apis[input.apiVersion]
    msg := sprintf("Resource '%v' uses removed API version '%v' — not supported in Kubernetes 1.29+.", [input.metadata.name, input.apiVersion])
}

# ---------------------------------------------------------------------------
# Deny: HPA using autoscaling/v1 (must use autoscaling/v2 for multi-metric)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "HorizontalPodAutoscaler"
    input.apiVersion == "autoscaling/v1"
    msg := sprintf("HPA '%v' uses autoscaling/v1 — must use autoscaling/v2 for CPU+memory dual-metric scaling.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: Service type LoadBalancer without annotations (bypasses Istio gateway)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "Service"
    input.spec.type == "LoadBalancer"
    not input.metadata.annotations["service.beta.kubernetes.io/aws-load-balancer-type"]
    not input.metadata.annotations["service.beta.kubernetes.io/aws-load-balancer-nlb-target-type"]
    msg := sprintf("Service '%v' of type LoadBalancer missing ALB/NLB annotations — must route through Istio gateway.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: Ingress without Istio gateway class or ALB class
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "Ingress"
    not input.spec.ingressClassName
    not input.metadata.annotations["kubernetes.io/ingress.class"]
    msg := sprintf("Ingress '%v' has no ingressClassName — must specify 'alb' or 'istio'.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: PodDisruptionBudget with minAvailable=0 (allows full disruption)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "PodDisruptionBudget"
    input.spec.minAvailable == 0
    msg := sprintf("PodDisruptionBudget '%v' has minAvailable=0 — allows complete pod disruption.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: StatefulSet without a serviceName (headless service required)
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "StatefulSet"
    not input.spec.serviceName
    msg := sprintf("StatefulSet '%v' missing spec.serviceName — headless service required for stable DNS.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Deny: CronJob without concurrencyPolicy
# ---------------------------------------------------------------------------
deny contains msg if {
    input.kind == "CronJob"
    not input.spec.concurrencyPolicy
    msg := sprintf("CronJob '%v' missing spec.concurrencyPolicy — set to 'Forbid' for ingestion jobs.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Warn: Deployment without podAntiAffinity (single-node HA risk)
# ---------------------------------------------------------------------------
warn contains msg if {
    input.kind == "Deployment"
    input.spec.replicas > 1
    not input.spec.template.spec.affinity.podAntiAffinity
    msg := sprintf("Deployment '%v' has replicas > 1 but no podAntiAffinity — all pods may schedule on one node.", [input.metadata.name])
}

# ---------------------------------------------------------------------------
# Warn: No topologySpreadConstraints on multi-replica workloads
# ---------------------------------------------------------------------------
warn contains msg if {
    input.kind == "Deployment"
    input.spec.replicas >= 3
    not input.spec.template.spec.topologySpreadConstraints
    msg := sprintf("Deployment '%v' has 3+ replicas but no topologySpreadConstraints — consider spreading across AZs.", [input.metadata.name])
}
