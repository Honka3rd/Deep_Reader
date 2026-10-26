#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/Deep_Reflective_Reader"
PYTHON="${DEEP_READER_PYTHON:-${BACKEND_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Install backend requirements in Deep_Reflective_Reader/.venv first, or set DEEP_READER_PYTHON to an executable path.\n' >&2
  exit 1
fi

cd "${BACKEND_DIR}"
exec "${PYTHON}" -m scripts.prepare_tokenizer_assets "$@"
