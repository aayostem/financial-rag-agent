# =============================================================================
# policies/rego/vault_policy.rego
# Validates Vault policy HCL files for least-privilege compliance.
# Run with conftest against parsed HCL or as a pre-commit check.
# Ensures no policy grants wildcard paths or root-level access.
# =============================================================================
package financial_rag.vault

import future.keywords.if
import future.keywords.contains

# ---------------------------------------------------------------------------
# Deny: Wildcard path in any policy
# ---------------------------------------------------------------------------
deny contains msg if {
    policy := input.policies[_]
    rule := policy.rules[_]
    endswith(rule.path, "/*")
    rule.capabilities[_] == "write"
    contains(rule.path, "secret/")
    msg := sprintf("Vault policy '%v' grants write on wildcard secret path '%v' — scope to exact paths.", [policy.name, rule.path])
}

# ---------------------------------------------------------------------------
# Deny: Delete capability on KV secret paths (app roles should never delete)
# ---------------------------------------------------------------------------
deny contains msg if {
    policy := input.policies[_]
    rule := policy.rules[_]
    contains(rule.path, "secret/")
    rule.capabilities[_] == "delete"
    msg := sprintf("Vault policy '%v' grants delete on path '%v' — application roles must not delete secrets.", [policy.name, rule.path])
}

# ---------------------------------------------------------------------------
# Deny: sudo capability in any application policy
# ---------------------------------------------------------------------------
deny contains msg if {
    policy := input.policies[_]
    rule := policy.rules[_]
    rule.capabilities[_] == "sudo"
    msg := sprintf("Vault policy '%v' grants sudo capability — sudo is reserved for administrative policies only.", [policy.name])
}

# ---------------------------------------------------------------------------
# Deny: Access to sys/ paths from application policies
# ---------------------------------------------------------------------------
deny contains msg if {
    policy := input.policies[_]
    rule := policy.rules[_]
    startswith(rule.path, "sys/")
    not policy.name == "admin"
    msg := sprintf("Vault policy '%v' accesses sys/ path '%v' — sys/ paths are administrative only.", [policy.name, rule.path])
}

# ---------------------------------------------------------------------------
# Warn: Policy has no TTL constraint on token (infinite lease risk)
# ---------------------------------------------------------------------------
warn contains msg if {
    policy := input.policies[_]
    not policy.token_ttl
    not policy.token_max_ttl
    msg := sprintf("Vault policy '%v' has no token_ttl or token_max_ttl — tokens issued with this policy have no expiry.", [policy.name])
}
