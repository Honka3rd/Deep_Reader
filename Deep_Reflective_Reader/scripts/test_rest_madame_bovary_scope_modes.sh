#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
DOC_NAME="${DOC_NAME:-Madame Bovary}"
TOP_K="${TOP_K:-5}"
LOG_FILE="${LOG_FILE:-/tmp/deep_reader_rest_madame_bovary_scope_modes.log}"
RESP_DIR="${RESP_DIR:-/tmp/deep_reader_rest_madame_bovary_scope_modes_responses}"
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
rm -f "$RESP_DIR"/health.json "$RESP_DIR"/case*.json

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
  tail -n 60 "$LOG_FILE" || true
  exit 1
fi

post_ask() {
  local case_id="$1"
  local session_id="$2"
  local query="$3"
  local out_file="$RESP_DIR/case${case_id}.json"
  local payload
  local http_code

  payload=$(printf '{"doc_name":"%s","query":"%s","top_k":%s,"session_id":"%s"}' \
    "$DOC_NAME" "$query" "$TOP_K" "$session_id")

  http_code="$(curl -sS -o "$out_file" -w "%{http_code}" \
    -X POST "http://${HOST}:${PORT}/documents/ask" \
    -H "Content-Type: application/json" \
    --data "$payload" || true)"

  echo "CASE${case_id} HTTP ${http_code}"
  cat "$out_file" || true
  echo

  if [[ "$http_code" != "200" ]]; then
    echo "FAIL: CASE${case_id} request failed."
    exit 1
  fi
}

# Case 1: local-scope factual question, first turn in session => should stay retrieval_mode.
SESSION_RETRIEVAL="case-retrieval-$(date +%s)"
post_ask 1 "$SESSION_RETRIEVAL" "Who is Charles Bovary?"

# Case 2: global-scope aggregation question.
SESSION_GLOBAL="case-global-$(date +%s)-$RANDOM"
post_ask 2 "$SESSION_GLOBAL" "Emma有哪些主要的情感關係？"

sleep 1

case1_scope_local="$(rg -n "ContextOrchestrator#scope: scope=local.*requested_top_k=${TOP_K}" "$LOG_FILE" -S || true)"
case1_retrieval_mode="$(rg -n "FaissIndexBundle#context_mode: retrieval_mode \\(active=None, best=.*threshold=.*\\)" "$LOG_FILE" -S || true)"

case2_scope_global="$(rg -n "ContextOrchestrator#scope: scope=global.*requested_top_k=${TOP_K}" "$LOG_FILE" -S || true)"
case2_global_coverage="$(rg -n "ContextOrchestrator#global_coverage:" "$LOG_FILE" -S || true)"
case2_full_text_mode="$(rg -n "ContextOrchestrator#context_mode: full_text_mode" "$LOG_FILE" -S || true)"

echo
echo "=== Matched Log Lines ==="
echo "[Case1] scope local:"
echo "${case1_scope_local}"
echo "[Case1] retrieval mode:"
echo "${case1_retrieval_mode}"
echo
echo "[Case2] scope global:"
echo "${case2_scope_global}"
echo "[Case2] global coverage:"
echo "${case2_global_coverage}"
echo "[Case2] full text mode:"
echo "${case2_full_text_mode}"

if [[ -z "$case1_scope_local" || -z "$case1_retrieval_mode" ]]; then
  echo
  echo "FAIL: Case1 did not trigger expected retrieval path."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

if [[ -z "$case2_scope_global" ]]; then
  echo
  echo "FAIL: Case2 did not trigger global scope."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

if [[ -z "$case2_global_coverage" && -z "$case2_full_text_mode" ]]; then
  echo
  echo "FAIL: Case2 is global but did not hit full_text_mode nor global_coverage fallback."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

echo
echo "PASS: Madame Bovary scope-mode tests passed."
echo "Case1: retrieval path triggered."
if [[ -n "$case2_full_text_mode" ]]; then
  echo "Case2: global full_text_mode triggered."
else
  echo "Case2: global retrieval/coverage path triggered."
fi
echo "Log file: $LOG_FILE"
