#!/bin/bash
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────
log()     { echo "  $*"; }
success() { echo "✅ $*"; }
error()   { echo "❌ $*" >&2; exit 1; }

wait_healthy() {
  local container=$1
  local retries=30
  log "Waiting for $container to be healthy..."
  until [ "$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)" == "healthy" ]; do
    ((retries--)) || error "$container did not become healthy in time"
    sleep 2
  done
  success "$container is healthy"
}

# ── Preflight ─────────────────────────────────────────────────────────────────
[ -f .env ] || error ".env file not found. Copy .env.example and configure it."

set -a; source .env; set +a

command -v docker &>/dev/null || error "Docker is not installed or not in PATH"
docker info &>/dev/null       || error "Docker daemon is not running"

# ── Core services ─────────────────────────────────────────────────────────────
log "Starting PostgreSQL and Redis..."
docker compose up -d postgres redis

wait_healthy "${POSTGRES_CONTAINER:-finrag-postgres}"
wait_healthy "${REDIS_CONTAINER:-finrag-redis}"

docker compose ps

# ── Optional tools ────────────────────────────────────────────────────────────
read -r -p "Start management tools (pgAdmin, Redis Commander)? (y/n) " start_tools
echo

if [[ $start_tools =~ ^[Yy]$ ]]; then
  log "Starting management tools..."
  docker compose --profile tools up -d
  success "Management tools started"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
success "Development environment ready!"
echo "   PostgreSQL  →  localhost:${POSTGRES_PORT:-5432}"
echo "   Redis       →  localhost:${REDIS_PORT:-6379}"

if [[ $start_tools =~ ^[Yy]$ ]]; then
  echo "   pgAdmin     →  http://localhost:5050  (admin@finrag.local / admin)"
  echo "   Redis UI    →  http://localhost:8081"
fi