#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
DOC_NAME="${DOC_NAME:-APPLE}"
TOP_K="${TOP_K:-5}"
LOG_FILE="${LOG_FILE:-/tmp/deep_reader_scope_signal_quality.log}"
RESP_DIR="${RESP_DIR:-/tmp/deep_reader_scope_signal_quality_responses}"
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

if [[ ! -f "data/raw/APPLE.pdf" ]]; then
  echo "FAIL: data/raw/APPLE.pdf not found."
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

mkdir -p "$RESP_DIR"
rm -f "$LOG_FILE"
rm -f "$RESP_DIR"/*.json "$RESP_DIR"/*.code

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
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

post_ask() {
  local case_id="$1"
  local session_id="$2"
  local query="$3"
  local out_file="$RESP_DIR/case${case_id}.json"
  local code_file="$RESP_DIR/case${case_id}.code"
  local payload
  local http_code

  payload=$(printf '{"doc_name":"%s","query":"%s","top_k":%s,"session_id":"%s"}' \
    "$DOC_NAME" "$query" "$TOP_K" "$session_id")

  http_code="$(curl -sS -o "$out_file" -w "%{http_code}" \
    -X POST "http://${HOST}:${PORT}/documents/ask" \
    -H "Content-Type: application/json" \
    --data "$payload" || true)"

  echo "$http_code" >"$code_file"
  echo "CASE${case_id} HTTP ${http_code}"
  cat "$out_file" || true
  echo

  if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
    echo "FAIL: CASE${case_id} request failed (expected 200 or 201)."
    exit 1
  fi
}

SESSION_ID="scope-quality-$(date +%s)-$RANDOM"

# Warm-up turn to ensure session_active_chunk_index exists.
post_ask 0 "$SESSION_ID" "What SEC form is this Apple filing?"

# Strong local signals (lexical, language-specific).
post_ask 1 "$SESSION_ID" "This paragraph is mainly about what?"
post_ask 2 "$SESSION_ID" "這一段在講什麼？"

# Weak local anchors (session-dependent lexical anchors).
post_ask 3 "$SESSION_ID" "What is explained here?"
post_ask 4 "$SESSION_ID" "這裡在講什麼？"

sleep 1

scope_local_logs="$(rg -n "ContextOrchestrator#scope: scope=local" "$LOG_FILE" -S || true)"
strong_logs="$(rg -n "QuestionScopeResolver#local_signal: quality=strong" "$LOG_FILE" -S || true)"
weak_logs="$(rg -n "QuestionScopeResolver#local_signal: quality=weak" "$LOG_FILE" -S || true)"

echo "=== Matched Log Lines ==="
echo "[Scope local]"
echo "${scope_local_logs}"
echo
echo "[Local signal strong]"
echo "${strong_logs}"
echo
echo "[Local signal weak]"
echo "${weak_logs}"
echo

if [[ -z "$scope_local_logs" ]]; then
  echo "FAIL: no local scope logs found."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

if [[ -z "$strong_logs" ]]; then
  echo "FAIL: strong local signal quality log not found."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

if [[ -z "$weak_logs" ]]; then
  echo "FAIL: weak local signal quality log not found."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY' "$RESP_DIR"/case*.json
import json
import sys

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    answer = payload.get("answer", "")
    if not isinstance(answer, str) or not answer.strip():
        raise SystemExit(f"FAIL: {path} has empty answer")
print("Response payload validation passed.")
PY

echo "PASS: scope signal quality test passed (strong + weak logs both present)."
echo "Log file: $LOG_FILE"
