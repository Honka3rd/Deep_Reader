#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
UI_DIR="${ROOT_DIR}/Deep_Reader_UI"
ENV_LOADER="${ROOT_DIR}/scripts/load_local_env.sh"
BACKEND_ONLY=false

if [[ "$#" -gt 1 ]]; then
  printf 'Usage: %s [--backend-only]\n' "$0" >&2
  exit 2
fi
case "${1:-}" in
  --backend-only) BACKEND_ONLY=true ;;
  --help|-h) printf 'Usage: %s [--backend-only]\n' "$0"; exit 0 ;;
  '') ;;
  *) printf 'Usage: %s [--backend-only]\n' "$0" >&2; exit 2 ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker is required but was not found.\n' >&2
  exit 1
fi

if [[ "${BACKEND_ONLY}" == false ]] && ! command -v npm >/dev/null 2>&1; then
  printf 'npm is required but was not found.\n' >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_LOADER}" "${ENV_FILE}"

: "${DEEP_READER_UI_PORT:?DEEP_READER_UI_PORT is required in .env.local}"

if [[ "${BACKEND_ONLY}" == false ]] && command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${DEEP_READER_UI_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  printf 'UI port %s is already in use. Stop that process or change DEEP_READER_UI_PORT in %s.\n' \
    "${DEEP_READER_UI_PORT}" "${ENV_FILE}" >&2
  exit 1
fi

printf 'Starting PostgreSQL and API with %s...\n' "${ENV_FILE}"
if [[ -n "${DEEP_READER_CONTAINER_PROXY:-}" ]]; then
  printf 'API outbound proxy: configured from .env.local.\n'
else
  printf 'API outbound proxy: direct connection.\n'
fi
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --wait postgres
printf 'Applying interaction version guards before API startup...\n'
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
  sh -c 'exec psql -v ON_ERROR_STOP=1 --single-transaction -f - -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < "${ROOT_DIR}/Deep_Reflective_Reader/db/migrations/003_interaction_version_guards.sql"
printf 'Building verified tokenizer assets; API startup will verify them offline...\n'
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build --wait api

printf 'Checking model API connectivity from the API container...\n'
if ! docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T api \
  python -m scripts.check_llm_connectivity; then
  printf 'Warning: Model API connectivity could not be confirmed. Continuing local startup; model-dependent actions may fail.\n' >&2
  printf 'Review the diagnostic above and DEEP_READER_CONTAINER_PROXY in .env.local. Network settings are unchanged; no route fallback was attempted.\n' >&2
fi

printf '\nBackend services:\n'
printf '  API:      http://127.0.0.1:%s\n' "${DEEP_READER_API_PORT}"
printf '  Postgres: 127.0.0.1:%s\n' "${DEEP_READER_POSTGRES_PORT}"
if [[ "${BACKEND_ONLY}" == true ]]; then
  printf 'Backend ready; existing UI processes were left unchanged.\n'
  exit 0
fi
printf 'Starting UI at http://127.0.0.1:%s/\n' "${DEEP_READER_UI_PORT}"

cd "${UI_DIR}"
exec npm run dev -- --host 127.0.0.1
