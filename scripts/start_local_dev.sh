#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
UI_DIR="${ROOT_DIR}/Deep_Reader_UI"

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Missing local environment file: %s\n' "${ENV_FILE}" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker is required but was not found.\n' >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  printf 'npm is required but was not found.\n' >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${DEEP_READER_UI_PORT:?DEEP_READER_UI_PORT is required in .env.local}"

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${DEEP_READER_UI_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  printf 'UI port %s is already in use. Stop that process or change DEEP_READER_UI_PORT in %s.\n' \
    "${DEEP_READER_UI_PORT}" "${ENV_FILE}" >&2
  exit 1
fi

printf 'Starting PostgreSQL and API with %s...\n' "${ENV_FILE}"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build postgres api

printf '\nBackend services:\n'
printf '  API:      http://127.0.0.1:%s\n' "${DEEP_READER_API_PORT}"
printf '  Postgres: 127.0.0.1:%s\n' "${DEEP_READER_POSTGRES_PORT}"
printf 'Starting UI at http://127.0.0.1:%s/\n' "${DEEP_READER_UI_PORT}"

cd "${UI_DIR}"
exec npm run dev -- --host 127.0.0.1
