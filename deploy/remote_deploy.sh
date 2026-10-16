#!/usr/bin/env bash
# Runs ON the server. The GitHub Actions deploy job pipes this script over SSH.
# Required env: IMAGE (full image reference), REGISTRY_USER, REGISTRY_TOKEN.
set -euo pipefail

: "${IMAGE:?IMAGE is required}"
: "${REGISTRY_USER:?REGISTRY_USER is required}"
: "${REGISTRY_TOKEN:?REGISTRY_TOKEN is required}"

ENV_FILE=/opt/rag/.env
if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Create it first (see DEPLOY.md)." >&2
  exit 1
fi

trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT

printf '%s' "$REGISTRY_TOKEN" | docker login ghcr.io -u "$REGISTRY_USER" --password-stdin
docker pull "$IMAGE"

docker rm -f rag >/dev/null 2>&1 || true
docker run -d --name rag --restart unless-stopped -p 80:8000 --env-file "$ENV_FILE" "$IMAGE"

for _ in $(seq 1 30); do
  if curl -fsS http://localhost/health >/dev/null 2>&1; then
    echo "Deployed $IMAGE and the health check passed."
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 5
done

echo "Health check failed after 150 seconds. Last container logs:" >&2
docker logs --tail 50 rag >&2
exit 1
