#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
TOP_K="${TOP_K:-5}"
DOC_NAME_BASE="${DOC_NAME_BASE:-APPLE}"
DOC_NAME_WITH_EXT="${DOC_NAME_WITH_EXT:-APPLE.pdf}"
LOG_FILE="${LOG_FILE:-/tmp/deep_reader_rest_apple_pdf_ask.log}"
RESP_DIR="${RESP_DIR:-/tmp/deep_reader_rest_apple_pdf_ask_responses}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-30}"
CURL_MAX_TIME_SECONDS="${CURL_MAX_TIME_SECONDS:-1800}"

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
rm -f "$RESP_DIR"/health.json "$RESP_DIR"/ask*.json "$RESP_DIR"/ask*.code

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
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

SESSION_ID_BASE="rest-apple-base-$(date +%s)"
SESSION_ID_EXT="rest-apple-ext-$(date +%s)"
QUERY="What SEC form is this Apple filing?"

post_ask() {
  local index="$1"
  local doc_name="$2"
  local session_id="$3"
  local out_file="$RESP_DIR/ask${index}.json"
  local code_file="$RESP_DIR/ask${index}.code"
  local payload
  local http_code

  payload=$(printf '{"doc_name":"%s","query":"%s","top_k":%s,"session_id":"%s"}' \
    "$doc_name" "$QUERY" "$TOP_K" "$session_id")

  http_code="$(curl -sS --max-time "$CURL_MAX_TIME_SECONDS" \
    -o "$out_file" -w "%{http_code}" \
    -X POST "http://${HOST}:${PORT}/documents/ask" \
    -H "Content-Type: application/json" \
    --data "$payload" || true)"

  echo "$http_code" >"$code_file"
  echo "ASK${index} HTTP ${http_code} (doc_name=${doc_name})"
  cat "$out_file" || true
  echo
}

post_ask 1 "$DOC_NAME_BASE" "$SESSION_ID_BASE"
post_ask 2 "$DOC_NAME_WITH_EXT" "$SESSION_ID_EXT"

sleep 1

ask1_code="$(cat "$RESP_DIR/ask1.code" 2>/dev/null || true)"
ask2_code="$(cat "$RESP_DIR/ask2.code" 2>/dev/null || true)"

bad_txt_not_found="$(rg -n 'APPLE\\.txt not found' "$LOG_FILE" -S || true)"
bad_double_pdf="$(rg -n 'APPLE\\.pdf\\.pdf not found' "$LOG_FILE" -S || true)"
bad_filenotfound="$(rg -n 'FileNotFoundError' "$LOG_FILE" -S || true)"

echo "=== HTTP Status ==="
echo "ASK1=${ask1_code}"
echo "ASK2=${ask2_code}"
echo
echo "=== Loader Error Signals ==="
echo "APPLE.txt not found:"
echo "${bad_txt_not_found}"
echo
echo "APPLE.pdf.pdf not found:"
echo "${bad_double_pdf}"
echo
echo "FileNotFoundError:"
echo "${bad_filenotfound}"
echo

if [[ "$ask1_code" != "200" || "$ask2_code" != "200" ]]; then
  echo "FAIL: expected both requests to return HTTP 200."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

if [[ -n "$bad_txt_not_found" || -n "$bad_double_pdf" ]]; then
  echo "FAIL: detected loader extension mismatch in server logs."
  tail -n 120 "$LOG_FILE" || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY' "$RESP_DIR/ask1.json" "$RESP_DIR/ask2.json"
import json
import sys

paths = sys.argv[1:]
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "answer" not in payload:
        raise SystemExit(f"FAIL: {path} missing answer field")
    if not isinstance(payload["answer"], str) or not payload["answer"].strip():
        raise SystemExit(f"FAIL: {path} answer is empty")
print("Response payload validation passed.")
PY

echo "PASS: REST API accepted both APPLE and APPLE.pdf on /documents/ask."
echo "Log file: $LOG_FILE"
