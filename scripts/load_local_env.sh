#!/usr/bin/env bash

# Source this file from local development scripts to load .env.local into the
# current shell before starting Docker Compose or other child processes.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'This script must be sourced, not executed directly.\n' >&2
  printf 'Usage: source scripts/load_local_env.sh [path/to/.env.local]\n' >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-${ROOT_DIR}/.env.local}"
ENV_TEMPLATE="${ROOT_DIR}/.env.local.example"

if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'Missing local environment file: %s\n' "${ENV_FILE}" >&2
  if [[ -f "${ENV_TEMPLATE}" ]]; then
    printf 'Create one with: cp %s %s\n' "${ENV_TEMPLATE}" "${ENV_FILE}" >&2
  fi
  return 1
fi

EXISTING_DEEP_READER_LLM_API_KEY="${DEEP_READER_LLM_API_KEY:-}"
EXISTING_OPENAI_API_KEY="${OPENAI_API_KEY:-}"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

if [[ -n "${EXISTING_DEEP_READER_LLM_API_KEY}" && -z "${DEEP_READER_LLM_API_KEY:-}" ]]; then
  export DEEP_READER_LLM_API_KEY="${EXISTING_DEEP_READER_LLM_API_KEY}"
fi

if [[ -n "${EXISTING_OPENAI_API_KEY}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${EXISTING_OPENAI_API_KEY}"
fi

# Keep the developer-facing private key name provider-neutral while the current
# backend still consumes the OpenAI-compatible environment variable.
if [[ -n "${DEEP_READER_LLM_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${DEEP_READER_LLM_API_KEY}"
fi

if [[ -n "${OPENAI_API_KEY:-}" && -z "${DEEP_READER_LLM_API_KEY:-}" ]]; then
  export DEEP_READER_LLM_API_KEY="${OPENAI_API_KEY}"
fi

export DEEP_READER_ENV_FILE="${ENV_FILE}"
