#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
DOC_NAME="${DOC_NAME:-Madame Bovary}"
TOP_K="${TOP_K:-3}"
LOG_FILE="${LOG_FILE:-/tmp/deep_reader_rest_dynamic_context.log}"
RESP_DIR="${RESP_DIR:-/tmp/deep_reader_rest_dynamic_context_responses}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-30}"

if ! command -v curl >/dev/null 2>&1; then
  echo "FAIL: curl not found."
  exit 2
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "FAIL: rg (ripgrep) not found."
  exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FAIL: OPENAI_API_KEY is not set."
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

mkdir -p "$RESP_DIR"
rm -f "$LOG_FILE"
rm -f "$RESP_DIR"/ask*.json "$RESP_DIR"/health.json

echo "Starting server: ${PYTHON_BIN} -m uvicorn main:app --host ${HOST} --port ${PORT}"
"$PYTHON_BIN" -m uvicorn main:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Waiting for /health ..."
max_tries=$(( START_TIMEOUT_SECONDS * 2 ))
for _ in $(seq 1 "$max_tries"); do
  if curl -sS "http://${HOST}:${PORT}/health" >"$RESP_DIR/health.json" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if [[ ! -s "$RESP_DIR/health.json" ]]; then
  echo "FAIL: server health check did not pass within ${START_TIMEOUT_SECONDS}s."
  echo "Server log tail:"
  tail -n 60 "$LOG_FILE" || true
  exit 1
fi

SESSION_ID="rest-dynamic-test-$(date +%s)"

post_ask() {
  local index="$1"
  local query="$2"
  local out_file="$RESP_DIR/ask${index}.json"
  local payload
  local http_code

  payload=$(printf '{"doc_name":"%s","query":"%s","top_k":%s,"session_id":"%s"}' \
    "$DOC_NAME" "$query" "$TOP_K" "$SESSION_ID")

  http_code="$(curl -sS -o "$out_file" -w "%{http_code}" \
    -X POST "http://${HOST}:${PORT}/documents/ask" \
    -H "Content-Type: application/json" \
    --data "$payload" || true)"

  echo "ASK${index} HTTP ${http_code}"
  cat "$out_file" || true
  echo
}

post_ask 1 "Who is the new fellow in the first chapter?"
post_ask 2 "Describe the cap of the new fellow in detail."
post_ask 3 "How did the class react when his name sounded like Charbovari?"

sleep 1

local_mode_hit="$(rg -n 'FaissIndexBundle#context_mode: local_window_mode' "$LOG_FILE" -S || true)"
radius_hit="$(rg -n 'FaissIndexBundle#context_mode: local_window_mode .*radius=[0-9]+' "$LOG_FILE" -S || true)"
budget_hit="$(rg -n 'FaissIndexBundle#context_mode: local_window_mode .*budget=[0-9]+' "$LOG_FILE" -S || true)"
retrieval_budget_hit="$(rg -n 'FaissIndexBundle#context_budget: retrieval_mode' "$LOG_FILE" -S || true)"

echo
echo "=== Matched Log Lines ==="
echo "${local_mode_hit}"
echo "${retrieval_budget_hit}"

if [[ -n "$local_mode_hit" && -n "$radius_hit" && -n "$budget_hit" && -n "$retrieval_budget_hit" ]]; then
  echo
  echo "PASS: dynamic radius + dynamic token budget logs were triggered via REST API."
  echo "Log file: $LOG_FILE"
  exit 0
fi

echo
echo "FAIL: expected log patterns were not fully matched."
echo "Expected:"
echo "1) local_window_mode"
echo "2) radius=<int> in local_window_mode"
echo "3) budget=<int> in local_window_mode"
echo "4) context_budget: retrieval_mode"
echo
echo "Server log tail:"
tail -n 120 "$LOG_FILE" || true
exit 1
