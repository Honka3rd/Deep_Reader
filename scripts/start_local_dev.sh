#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
UI_DIR="${ROOT_DIR}/Deep_Reader_UI"
ENV_LOADER="${ROOT_DIR}/scripts/load_local_env.sh"

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker is required but was not found.\n' >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  printf 'npm is required but was not found.\n' >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_LOADER}" "${ENV_FILE}"

: "${DEEP_READER_UI_PORT:?DEEP_READER_UI_PORT is required in .env.local}"

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${DEEP_READER_UI_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  printf 'UI port %s is already in use. Stop that process or change DEEP_READER_UI_PORT in %s.\n' \
    "${DEEP_READER_UI_PORT}" "${ENV_FILE}" >&2
  exit 1
fi

printf 'Starting PostgreSQL and API with %s...\n' "${ENV_FILE}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --wait postgres
printf 'Applying interaction version guards before API startup...\n'
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
  sh -c 'exec psql -v ON_ERROR_STOP=1 --single-transaction -f - -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < "${ROOT_DIR}/Deep_Reflective_Reader/db/migrations/003_interaction_version_guards.sql"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build api

printf '\nBackend services:\n'
printf '  API:      http://127.0.0.1:%s\n' "${DEEP_READER_API_PORT}"
printf '  Postgres: 127.0.0.1:%s\n' "${DEEP_READER_POSTGRES_PORT}"
printf 'Starting UI at http://127.0.0.1:%s/\n' "${DEEP_READER_UI_PORT}"

cd "${UI_DIR}"
exec npm run dev -- --host 127.0.0.1
