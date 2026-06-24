# =============================================================================
# policies/tests/k8s_admission_test.rego
# Unit tests for the k8s_admission policies.
# Run: conftest verify --policy policies/rego/
# =============================================================================
package financial_rag.k8s.admission_test

import data.financial_rag.k8s.admission

# ---------------------------------------------------------------------------
# Test: latest tag is denied
# ---------------------------------------------------------------------------
test_deny_latest_tag if {
    result := admission.deny with input as {
        "review": {"object": {
            "metadata": {"labels": {
                "app.kubernetes.io/name": "api",
                "app.kubernetes.io/component": "api"
            }},
            "spec": {
                "containers": [{"name": "api", "image": "my-image:latest",
                    "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"memory": "256Mi"}},
                    "securityContext": {
                        "readOnlyRootFilesystem": true,
                        "allowPrivilegeEscalation": false,
                        "capabilities": {"drop": ["ALL"]}
                    }
                }]
            }
        }}
    }
    count(result) > 0
    some msg in result
    contains(msg, ":latest")
}

# ---------------------------------------------------------------------------
# Test: image without tag is denied
# ---------------------------------------------------------------------------
test_deny_no_tag if {
    result := admission.deny with input as {
        "review": {"object": {
            "metadata": {"labels": {
                "app.kubernetes.io/name": "api",
                "app.kubernetes.io/component": "api"
            }},
            "spec": {
                "containers": [{"name": "api", "image": "my-image",
                    "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"memory": "256Mi"}},
                    "securityContext": {
                        "readOnlyRootFilesystem": true,
                        "allowPrivilegeEscalation": false,
                        "capabilities": {"drop": ["ALL"]}
                    }
                }]
            }
        }}
    }
    count(result) > 0
}

# ---------------------------------------------------------------------------
# Test: compliant pod passes all checks
# ---------------------------------------------------------------------------
test_compliant_pod_passes if {
    result := admission.deny with input as {
        "review": {"object": {
            "metadata": {"labels": {
                "app.kubernetes.io/name": "financial-rag-agent",
                "app.kubernetes.io/component": "api"
            }},
            "spec": {
                "securityContext": {"runAsNonRoot": true},
                "hostNetwork": false,
                "hostPID": false,
                "containers": [{"name": "api",
                    "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/financial-rag-agent/api:sha-abc123",
                    "resources": {
                        "requests": {"cpu": "500m", "memory": "512Mi"},
                        "limits": {"cpu": "1000m", "memory": "1Gi"}
                    },
                    "securityContext": {
                        "readOnlyRootFilesystem": true,
                        "allowPrivilegeEscalation": false,
                        "runAsUser": 1000,
                        "capabilities": {"drop": ["ALL"]}
                    },
                    "livenessProbe": {"httpGet": {"path": "/health", "port": 8000}},
                    "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}}
                }]
            }
        }}
    }
    count(result) == 0
}

# ---------------------------------------------------------------------------
# Test: missing CPU request is denied
# ---------------------------------------------------------------------------
test_deny_missing_cpu_request if {
    result := admission.deny with input as {
        "review": {"object": {
            "metadata": {"labels": {
                "app.kubernetes.io/name": "api",
                "app.kubernetes.io/component": "api"
            }},
            "spec": {
                "containers": [{"name": "api",
                    "image": "my-image:sha-abc123",
                    "resources": {"requests": {"memory": "128Mi"}, "limits": {"memory": "256Mi"}},
                    "securityContext": {
                        "readOnlyRootFilesystem": true,
                        "allowPrivilegeEscalation": false,
                        "capabilities": {"drop": ["ALL"]}
                    }
                }]
            }
        }}
    }
    count(result) > 0
    some msg in result
    contains(msg, "cpu")
}

# ---------------------------------------------------------------------------
# Test: privileged container is denied
# ---------------------------------------------------------------------------
test_deny_privileged if {
    result := admission.deny with input as {
        "review": {"object": {
            "metadata": {"labels": {
                "app.kubernetes.io/name": "api",
                "app.kubernetes.io/component": "api"
            }},
            "spec": {
                "containers": [{"name": "api",
                    "image": "my-image:sha-abc123",
                    "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"memory": "256Mi"}},
                    "securityContext": {
                        "privileged": true,
                        "readOnlyRootFilesystem": true,
                        "allowPrivilegeEscalation": false,
                        "capabilities": {"drop": ["ALL"]}
                    }
                }]
            }
        }}
    }
    count(result) > 0
    some msg in result
    contains(msg, "privileged")
}
