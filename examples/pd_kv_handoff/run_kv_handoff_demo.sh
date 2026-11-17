#!/usr/bin/env bash
# RFC-38 KV handoff passthrough demo.
#
# Starts one mock prefill worker, one mock decode worker and a PD router using
# the nixl passthrough branch. The prefill worker returns the RFC-38 §7.3
# kv_transfer_params object; the decode worker verifies that the router
# forwarded it unchanged.
#
# Usage:
#   ./examples/pd_kv_handoff/run_kv_handoff_demo.sh
#
# Requires: curl, Python 3 with fastapi/uvicorn, and a built router binary.
# Build the binary with:
#   cargo build --release --bin vllm-router
# Or point ROUTER_BIN at an existing binary.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PYTHON=${PYTHON:-python3}
ROUTER_BIN=${ROUTER_BIN:-$REPO_ROOT/target/release/vllm-router}
if [[ ! -x "$ROUTER_BIN" ]]; then
    ROUTER_BIN=$(command -v vllm-router || true)
fi
if [[ -z "$ROUTER_BIN" || ! -x "$ROUTER_BIN" ]]; then
    echo "router binary not found." >&2
    echo "Build it first: cargo build --release --bin vllm-router" >&2
    echo "Or set ROUTER_BIN=/path/to/vllm-router" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
    echo "$PYTHON is missing fastapi/uvicorn." >&2
    echo "Install the project dev dependencies first: pip install -e '.[dev]'" >&2
    echo "Or set PYTHON=/path/to/python with fastapi/uvicorn installed." >&2
    exit 1
fi

PREFILL_PORT=${PREFILL_PORT:-18181}
DECODE_PORT=${DECODE_PORT:-18182}
ROUTER_PORT=${ROUTER_PORT:-18190}
METRICS_PORT=${METRICS_PORT:-19190}
MODEL=${MODEL:-mock-pd-model}

LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/vllm-router-kv-handoff.XXXXXX)}
CAPTURE_DIR=${CAPTURE_DIR:-$LOG_DIR/capture}
mkdir -p "$LOG_DIR" "$CAPTURE_DIR"

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "logs:     $LOG_DIR"
    echo "captures: $CAPTURE_DIR"
}
trap cleanup EXIT INT TERM

wait_http() {
    local url=$1
    local name=$2
    local attempts=${3:-200}
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "timed out waiting for $name at $url" >&2
    return 1
}

start_worker() {
    local role=$1
    local port=$2
    local worker_id=$3
    echo "starting $role worker $worker_id on 127.0.0.1:$port"
    "$PYTHON" "$SCRIPT_DIR/mock_kv_worker.py" \
        --host 127.0.0.1 \
        --port "$port" \
        --role "$role" \
        --worker-id "$worker_id" \
        --model "$MODEL" \
        --capture-dir "$CAPTURE_DIR" \
        >"$LOG_DIR/$worker_id.log" 2>&1 &
    PIDS+=("$!")
}

echo "=== starting mock P/D workers ==="
start_worker prefill "$PREFILL_PORT" "prefill-1"
start_worker decode "$DECODE_PORT" "decode-1"

wait_http "http://127.0.0.1:$PREFILL_PORT/health" "prefill worker"
wait_http "http://127.0.0.1:$DECODE_PORT/health" "decode worker"

echo "=== starting PD router on 127.0.0.1:$ROUTER_PORT ==="
ROUTER_ARGS=(
    --host 127.0.0.1
    --port "$ROUTER_PORT"
    --policy round_robin
    --vllm-pd-disaggregation
    --kv-connector nixl
    --prefill "http://127.0.0.1:$PREFILL_PORT" none
    --decode "http://127.0.0.1:$DECODE_PORT"
    --prefill-policy round_robin
    --decode-policy round_robin
    --worker-startup-timeout-secs 30
    --worker-startup-check-interval 1
    --prometheus-host 127.0.0.1
    --prometheus-port "$METRICS_PORT"
    --log-level debug
)
"$ROUTER_BIN" "${ROUTER_ARGS[@]}" >"$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("$ROUTER_PID")

if ! wait_http "http://127.0.0.1:$ROUTER_PORT/health" "pd router" 300; then
    echo "--- router log ---" >&2
    tail -80 "$LOG_DIR/router.log" >&2 || true
    exit 1
fi

echo "=== sending request through the router ==="
HTTP_CODE=$(curl -sS \
    -o "$LOG_DIR/client_response.json" \
    -w '%{http_code}' \
    "http://127.0.0.1:$ROUTER_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"verify RFC-38 KV handoff passthrough\"}],
        \"max_tokens\": 8,
        \"stream\": false
    }" || true)

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "router returned HTTP $HTTP_CODE" >&2
    echo "--- client response ---" >&2
    cat "$LOG_DIR/client_response.json" >&2 || true
    echo "--- router log ---" >&2
    tail -80 "$LOG_DIR/router.log" >&2 || true
    echo "--- prefill log ---" >&2
    tail -40 "$LOG_DIR/prefill-1.log" >&2 || true
    echo "--- decode log ---" >&2
    tail -40 "$LOG_DIR/decode-1.log" >&2 || true
    exit 1
fi

echo "=== verifying kv_transfer_params passthrough ==="
"$PYTHON" "$SCRIPT_DIR/verify_handoff.py" \
    --capture-dir "$CAPTURE_DIR" \
    --response "$LOG_DIR/client_response.json"

echo
echo "RFC-38 KV handoff demo passed."
