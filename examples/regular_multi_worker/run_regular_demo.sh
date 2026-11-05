#!/usr/bin/env bash
# Local regular (non-PD) multi-worker demo.
#
# Starts three mock vLLM workers that each run prefill and decode in the same
# instance, plus one regular router that spreads requests across them.
#
# Usage:
#   ./run_regular_demo.sh                  # run checks, keep router alive
#   KEEP_RUNNING=0 ./run_regular_demo.sh   # run checks and exit
#
# Requires: curl, Python 3 with fastapi/uvicorn/requests, and a built router binary.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MOCK_WORKER="$REPO_ROOT/examples/pd_disaggregation/mock_worker.py"

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

if ! "$PYTHON" -c 'import fastapi, uvicorn, requests' >/dev/null 2>&1; then
    echo "$PYTHON is missing fastapi/uvicorn/requests." >&2
    echo "Install the project dev dependencies first: pip install -e '.[dev]'" >&2
    exit 1
fi

read -r -a WORKER_PORTS <<< "${WORKER_PORTS:-18081 18082 18083}"

ROUTER_PORT=${ROUTER_PORT:-18090}
METRICS_PORT=${METRICS_PORT:-19090}
POLICY=${POLICY:-round_robin}
MODEL=${MODEL:-mock-regular-model}
WORKER_STARTUP_TIMEOUT=${WORKER_STARTUP_TIMEOUT:-30}
KEEP_RUNNING=${KEEP_RUNNING:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

LOG_DIR=${LOG_DIR:-$(mktemp -d /tmp/vllm-router-regular-demo.XXXXXX)}
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
    local port=$1
    local worker_id=$2
    echo "starting regular worker $worker_id on 127.0.0.1:$port"
    "$PYTHON" "$MOCK_WORKER" \
        --host 127.0.0.1 \
        --port "$port" \
        --role regular \
        --worker-id "$worker_id" \
        --model "$MODEL" \
        >"$LOG_DIR/$worker_id.log" 2>&1 &
    PIDS+=("$!")
}

echo "=== starting mock workers ==="
for i in "${!WORKER_PORTS[@]}"; do
    start_worker "${WORKER_PORTS[$i]}" "regular-$((i + 1))"
done

for port in "${WORKER_PORTS[@]}"; do
    wait_http "http://127.0.0.1:$port/health" "worker on $port"
done

ROUTER_ARGS=(
    --host 127.0.0.1
    --port "$ROUTER_PORT"
    --policy "$POLICY"
    --worker-startup-timeout-secs "$WORKER_STARTUP_TIMEOUT"
    --worker-startup-check-interval 1
    --prometheus-host 127.0.0.1
    --prometheus-port "$METRICS_PORT"
    --log-level "$LOG_LEVEL"
    --worker-urls
)
for port in "${WORKER_PORTS[@]}"; do
    ROUTER_ARGS+=("http://127.0.0.1:$port")
done

echo "=== starting regular router on 127.0.0.1:$ROUTER_PORT ==="
echo "workers: ${WORKER_PORTS[*]}"
echo "policy:  $POLICY"
"$ROUTER_BIN" "${ROUTER_ARGS[@]}" >"$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("$ROUTER_PID")

if ! wait_http "http://127.0.0.1:$ROUTER_PORT/health" "regular router" 300; then
    echo "--- router log ---" >&2
    tail -50 "$LOG_DIR/router.log" >&2 || true
    exit 1
fi

echo "=== sending requests through the router ==="
"$PYTHON" - "$ROUTER_PORT" "$MODEL" <<'PY'
import collections
import sys

import requests

port = int(sys.argv[1])
model = sys.argv[2]
base = f"http://127.0.0.1:{port}"
roles = collections.Counter()
workers = collections.Counter()

for i in range(12):
    response = requests.post(
        f"{base}/v1/completions",
        json={
            "model": model,
            "prompt": f"regular demo request {i}",
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

# A streaming request confirms the same regular worker path also handles SSE.
with requests.post(
    f"{base}/v1/chat/completions",
    json={
        "model": model,
        "messages": [{"role": "user", "content": "stream hello"}],
        "max_tokens": 4,
        "stream": True,
    },
    timeout=10,
    stream=True,
) as stream_response:
    stream_response.raise_for_status()
    chunks = [
        line
        for line in stream_response.iter_lines(decode_unicode=True)
        if line and line.startswith("data: ")
    ]

if not chunks or chunks[-1] != "data: [DONE]":
    raise SystemExit(f"expected a terminated SSE stream, got: {chunks[-3:]!r}")

if roles != {"regular": 12}:
    raise SystemExit(f"expected all responses from regular workers, got: {dict(roles)}")

unique_workers = len(workers)
print()
print("regular worker distribution:", dict(workers))
print(f"unique workers hit: {unique_workers}")
if unique_workers < 2:
    raise SystemExit(f"expected at least two regular workers, got: {dict(workers)}")

print("Regular multi-worker routing checks passed.")
PY

if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo
    echo "Router is running at http://127.0.0.1:$ROUTER_PORT"
    echo "Metrics: http://127.0.0.1:$METRICS_PORT/metrics"
    echo "Try:"
    echo "  curl -s http://127.0.0.1:$ROUTER_PORT/v1/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"mock-regular-model\",\"prompt\":\"hello\",\"max_tokens\":4}'"
    echo "Press Ctrl+C to stop."
    wait "$ROUTER_PID" || true
else
    echo "KEEP_RUNNING=0, shutting down."
fi
