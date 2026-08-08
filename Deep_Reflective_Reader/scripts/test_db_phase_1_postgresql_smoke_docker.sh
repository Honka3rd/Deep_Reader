#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yml"
SERVICE_NAME="postgres"
DB_NAME="${POSTGRES_DB:-deep_reader}"
DB_USER="${POSTGRES_USER:-deep_reader}"

cd "${REPO_ROOT}"

docker compose -f "${COMPOSE_FILE}" up -d "${SERVICE_NAME}"

echo "Waiting for PostgreSQL container health..."
for _ in $(seq 1 30); do
  container_id="$(docker compose -f "${COMPOSE_FILE}" ps -q "${SERVICE_NAME}")"
  status="$(docker inspect --format '{{.State.Health.Status}}' "${container_id}" 2>/dev/null || true)"
  if [[ "${status}" == "healthy" ]]; then
    break
  fi
  sleep 1
done

container_id="$(docker compose -f "${COMPOSE_FILE}" ps -q "${SERVICE_NAME}")"
status="$(docker inspect --format '{{.State.Health.Status}}' "${container_id}")"
if [[ "${status}" != "healthy" ]]; then
  docker compose -f "${COMPOSE_FILE}" logs "${SERVICE_NAME}"
  echo "PostgreSQL container did not become healthy." >&2
  exit 1
fi

tmp_sql="$(mktemp)"
trap 'rm -f "${tmp_sql}"' EXIT

python3 - <<'PY' > "${tmp_sql}"
from pathlib import Path

script_path = Path("Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py")
namespace = {
    "__file__": str(script_path),
    "__name__": "deep_reader_smoke_sql_loader",
}
exec(script_path.read_text(encoding="utf-8"), namespace)
print(namespace["SMOKE_SQL"].format(
    migration_sql=namespace["MIGRATION_PATH"].read_text(encoding="utf-8"),
))
PY

docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE_NAME}" \
  psql -U "${DB_USER}" -d "${DB_NAME}" --set ON_ERROR_STOP=1 < "${tmp_sql}"

echo '{"status":"ok","tests":["db_phase_1_postgresql_relational_consistency_smoke_docker"]}'
