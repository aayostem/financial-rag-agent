# =============================================================================
# policies/rego/k8s_admission.rego
# OPA Gatekeeper / Conftest admission policies for financial-rag-agent.
# Enforced at:
#   1. CI (conftest) — every PR touching Helm or raw manifests
#   2. Kubernetes admission (Gatekeeper ConstraintTemplate) — every pod create
#
# Rules:
#   - No :latest image tags in production
#   - All containers must declare resource requests AND limits
#   - readOnlyRootFilesystem must be true
#   - allowPrivilegeEscalation must be false
#   - Must not run as root (runAsNonRoot or runAsUser > 0)
#   - No host network, host PID, host IPC
#   - ALL capabilities must be dropped
#   - No privileged containers
#   - Required labels: app.kubernetes.io/name, app.kubernetes.io/component
# =============================================================================

package financial_rag.k8s.admission

import future.keywords.if
import future.keywords.in
import future.keywords.contains

# ---------------------------------------------------------------------------
# Deny: :latest image tag in any container
# Mutable tags make deployments non-reproducible and bypass supply chain controls
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    endswith(container.image, ":latest")
    msg := sprintf(
        "Container '%v' uses ':latest' tag. Pin to an immutable SHA or semver tag. Image: %v",
        [container.name, container.image]
    )
}

deny contains msg if {
    container := input.review.object.spec.initContainers[_]
    endswith(container.image, ":latest")
    msg := sprintf(
        "InitContainer '%v' uses ':latest' tag. Image: %v",
        [container.name, container.image]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not contains(container.image, ":")
    msg := sprintf(
        "Container '%v' has no image tag at all. All images must use pinned tags. Image: %v",
        [container.name, container.image]
    )
}

# ---------------------------------------------------------------------------
# Deny: Missing resource requests or limits
# HPA requires requests; limits prevent noisy-neighbour OOM on shared nodes
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.requests.cpu
    msg := sprintf(
        "Container '%v' missing resources.requests.cpu. HPA cannot function without CPU requests.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.requests.memory
    msg := sprintf(
        "Container '%v' missing resources.requests.memory.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.limits.cpu
    msg := sprintf(
        "Container '%v' missing resources.limits.cpu. Unbounded CPU causes node starvation.",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.resources.limits.memory
    msg := sprintf(
        "Container '%v' missing resources.limits.memory. Unbounded memory causes OOM evictions.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: readOnlyRootFilesystem not set to true
# Writable root filesystem allows post-exploit persistence
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem == true
    msg := sprintf(
        "Container '%v': securityContext.readOnlyRootFilesystem must be true. Use emptyDir for writable paths.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: allowPrivilegeEscalation not set to false
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.allowPrivilegeEscalation == false
    msg := sprintf(
        "Container '%v': securityContext.allowPrivilegeEscalation must be false.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: Running as root
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    container.securityContext.runAsUser == 0
    msg := sprintf(
        "Container '%v' runs as UID 0 (root). Set runAsUser to a non-zero UID.",
        [container.name]
    )
}

deny contains msg if {
    not input.review.object.spec.securityContext.runAsNonRoot == true
    not_all_containers_have_user := {c.name | c := input.review.object.spec.containers[_]; not c.securityContext.runAsUser}
    count(not_all_containers_have_user) > 0
    msg := "Pod must set securityContext.runAsNonRoot: true at pod level, or runAsUser at container level."
}

# ---------------------------------------------------------------------------
# Deny: Privileged containers
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf(
        "Container '%v' is privileged. Privileged containers have full host access.",
        [container.name]
    )
}

# ---------------------------------------------------------------------------
# Deny: Capabilities not fully dropped
# ---------------------------------------------------------------------------
deny contains msg if {
    container := input.review.object.spec.containers[_]
    not container.securityContext.capabilities.drop
    msg := sprintf(
        "Container '%v': securityContext.capabilities.drop must be set. Require drop: [ALL].",
        [container.name]
    )
}

deny contains msg if {
    container := input.review.object.spec.containers[_]
    caps := container.securityContext.capabilities.drop
    not "ALL" in caps
    msg := sprintf(
        "Container '%v': capabilities.drop must include 'ALL'. Current drop: %v",
        [container.name, caps]
    )
}

# ---------------------------------------------------------------------------
# Deny: Host namespaces
# ---------------------------------------------------------------------------
deny contains msg if {
    input.review.object.spec.hostNetwork == true
    msg := "Pod must not use hostNetwork. This grants access to the node's network namespace."
}

deny contains msg if {
    input.review.object.spec.hostPID == true
    msg := "Pod must not use hostPID. This grants visibility into all host processes."
}

deny contains msg if {
    input.review.object.spec.hostIPC == true
    msg := "Pod must not use hostIPC. This allows IPC namespace sharing with the host."
}

# ---------------------------------------------------------------------------
# Deny: Missing required Kubernetes labels
# Required for: Cilium policy selectors, Istio AuthorizationPolicy, cost attribution
# ---------------------------------------------------------------------------
deny contains msg if {
    not input.review.object.metadata.labels["app.kubernetes.io/name"]
    msg := "Pod missing required label: app.kubernetes.io/name. Required for service mesh routing and cost attribution."
}

deny contains msg if {
    not input.review.object.metadata.labels["app.kubernetes.io/component"]
    msg := "Pod missing required label: app.kubernetes.io/component. Required for Cilium network policy selectors."
}

# ---------------------------------------------------------------------------
# Warn: No liveness or readiness probe
# Not blocking (warn only) — some jobs legitimately skip probes
# ---------------------------------------------------------------------------
warn contains msg if {
    container := input.review.object.spec.containers[_]
    not container.livenessProbe
    msg := sprintf(
        "Container '%v' has no livenessProbe. Unhealthy pods will not be restarted automatically.",
        [container.name]
    )
}

warn contains msg if {
    container := input.review.object.spec.containers[_]
    not container.readinessProbe
    msg := sprintf(
        "Container '%v' has no readinessProbe. Pods will receive traffic before they are ready.",
        [container.name]
    )
}
