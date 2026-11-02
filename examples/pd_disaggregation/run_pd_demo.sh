#!/usr/bin/env bash
# Local prefill/decode disaggregation demo.
#
# Starts 2 mock prefill workers, 3 mock decode workers and one PD router,
# sends requests through the router, then optionally keeps the router running.
#
# Usage:
#   ./run_pd_demo.sh                  # run checks, keep router alive
#   KEEP_RUNNING=0 ./run_pd_demo.sh   # run checks and exit
#
# Requires: curl, Python 3 with fastapi/uvicorn, and a built router binary.

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
    exit 1
fi

read -r -a PREFILL_PORTS <<< "${PREFILL_PORTS:-18081 18082}"
read -r -a DECODE_PORTS <<< "${DECODE_PORTS:-18083 18084 18085}"

ROUTER_PORT=${ROUTER_PORT:-18090}
METRICS_PORT=${METRICS_PORT:-19090}
POLICY=${POLICY:-round_robin}
KEEP_RUNNING=${KEEP_RUNNING:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/vllm-router-pd-demo.XXXXXX)}
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "logs: $LOG_DIR"
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
    "$PYTHON" "$SCRIPT_DIR/mock_worker.py" \
        --host 127.0.0.1 \
        --port "$port" \
        --role "$role" \
        --worker-id "$worker_id" \
        >"$LOG_DIR/$worker_id.log" 2>&1 &
    PIDS+=("$!")
}

echo "=== starting mock workers ==="
for i in "${!PREFILL_PORTS[@]}"; do
    start_worker prefill "${PREFILL_PORTS[$i]}" "prefill-$((i + 1))"
done
for i in "${!DECODE_PORTS[@]}"; do
    start_worker decode "${DECODE_PORTS[$i]}" "decode-$((i + 1))"
done

for port in "${PREFILL_PORTS[@]}"; do
    wait_http "http://127.0.0.1:$port/health" "prefill worker on $port"
done
for port in "${DECODE_PORTS[@]}"; do
    wait_http "http://127.0.0.1:$port/health" "decode worker on $port"
done

ROUTER_ARGS=(
    --host 127.0.0.1
    --port "$ROUTER_PORT"
    --policy "$POLICY"
    --vllm-pd-disaggregation
    --worker-startup-timeout-secs 30
    --worker-startup-check-interval 1
    --prometheus-host 127.0.0.1
    --prometheus-port "$METRICS_PORT"
    --log-level "$LOG_LEVEL"
)
for port in "${PREFILL_PORTS[@]}"; do
    ROUTER_ARGS+=(--prefill "http://127.0.0.1:$port" none)
done
for port in "${DECODE_PORTS[@]}"; do
    ROUTER_ARGS+=(--decode "http://127.0.0.1:$port")
done

echo "=== starting PD router on 127.0.0.1:$ROUTER_PORT ==="
echo "prefill: ${PREFILL_PORTS[*]}"
echo "decode:  ${DECODE_PORTS[*]}"
"$ROUTER_BIN" "${ROUTER_ARGS[@]}" >"$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("$ROUTER_PID")

if ! wait_http "http://127.0.0.1:$ROUTER_PORT/health" "pd router" 300; then
    echo "--- router log ---" >&2
    tail -50 "$LOG_DIR/router.log" >&2 || true
    exit 1
fi

echo "=== sending requests through the router ==="
"$PYTHON" - "$ROUTER_PORT" <<'PY'
import collections
import sys

import requests

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/completions"
roles = collections.Counter()
workers = collections.Counter()

for i in range(12):
    response = requests.post(
        url,
        json={
            "model": "mock-pd-model",
            "prompt": f"pd demo request {i}",
            "max_tokens": 4,
            "stream": False,
        },
        timeout=10,
    )
    response.raise_for_status()
    body = response.json()
    role = body.get("worker_role") or response.headers.get("X-Worker-Role")
    worker = body.get("worker_id") or response.headers.get("X-Worker-Id")
    roles[role] += 1
    workers[worker] += 1
    text = body["choices"][0].get("text", "")
    print(f"request {i + 1:02d}: status={response.status_code} role={role} worker={worker} text={text!r}")

if roles != {"decode": 12}:
    raise SystemExit(f"expected all responses from decode workers, got: {dict(roles)}")
if len(workers) < 2:
    raise SystemExit(f"expected at least two decode workers, got: {dict(workers)}")

print()
print("decode worker distribution:", dict(workers))
print("PD routing checks passed.")
PY

if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo
    echo "Router is running at http://127.0.0.1:$ROUTER_PORT"
    echo "Metrics: http://127.0.0.1:$METRICS_PORT/metrics"
    echo "Try:"
    echo "  curl -s http://127.0.0.1:$ROUTER_PORT/v1/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"mock-pd-model\",\"prompt\":\"hello\",\"max_tokens\":4}'"
    echo "Press Ctrl+C to stop."
    wait "$ROUTER_PID" || true
else
    echo "KEEP_RUNNING=0, shutting down."
fi
