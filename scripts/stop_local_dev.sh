#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env.local"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DEEP_READER_UI_PORT="${DEEP_READER_UI_PORT:-5173}"

stop_ui_processes() {
  if ! command -v lsof >/dev/null 2>&1; then
    printf 'lsof is not available; skipping UI port cleanup for %s.\n' \
      "${DEEP_READER_UI_PORT}" >&2
    return
  fi

  pids="$(lsof -nP -tiTCP:"${DEEP_READER_UI_PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true)"

  if [[ -z "${pids}" ]]; then
    printf 'No UI process is listening on port %s.\n' "${DEEP_READER_UI_PORT}"
    return
  fi

  for pid in ${pids}; do
    command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    if [[ ! "${command_line}" =~ (Deep_Reader_UI|vite|npm|node) ]]; then
      printf 'Skipping pid %s on port %s; command does not look like the local UI: %s\n' \
        "${pid}" "${DEEP_READER_UI_PORT}" "${command_line}" >&2
      continue
    fi

    printf 'Stopping UI process pid %s on port %s: %s\n' \
      "${pid}" "${DEEP_READER_UI_PORT}" "${command_line}"
    kill "${pid}" 2>/dev/null || true
  done

  sleep 2

  for pid in ${pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      command_line="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
      if [[ "${command_line}" =~ (Deep_Reader_UI|vite|npm|node) ]]; then
        printf 'Force stopping UI process pid %s.\n' "${pid}"
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
  done
}

stop_compose_services() {
  if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is not available; skipping backend service cleanup.\n' >&2
    return
  fi

  if [[ -f "${ENV_FILE}" ]]; then
    printf 'Stopping Docker Compose services with %s...\n' "${ENV_FILE}"
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" down
  else
    printf 'Stopping Docker Compose services without .env.local...\n'
    docker compose -f "${COMPOSE_FILE}" down
  fi
}

stop_ui_processes
stop_compose_services

printf 'Local Deep Reader development services stopped.\n'
