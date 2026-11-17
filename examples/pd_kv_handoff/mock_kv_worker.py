#!/usr/bin/env python3
"""Mock prefill/decode workers for the RFC-38 KV handoff passthrough demo.

The prefill worker returns the ``kv_transfer_params`` object from
``kv_transfer_params.rfc38.json`` in its non-streaming response. The decode
worker checks that the router forwarded that object unchanged, then returns a
deterministic chat completion.

The workers do not run a model and do not move any KV bytes. The demo only
exercises the router's control-plane handoff:

    client -> vllm-router (PD, --kv-connector nixl)
                |-> prefill mock: returns kv_transfer_params
                `-> decode mock: verifies the same object arrived

Run through ``run_kv_handoff_demo.sh``; ``--help`` lists the individual
arguments if you want to start the workers manually.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_HANDOFF_TEMPLATE = EXAMPLES_DIR / "kv_transfer_params.rfc38.json"

# Fields that RFC-38 §7.3 requires in the object handed from P to D.
REQUIRED_FIELDS = (
    "do_remote_decode",
    "do_remote_prefill",
    "transfer_id",
    "remote_request_id",
    "remote_engine_id",
    "remote_block_ids",
    "num_prompt_blocks",
    "remote_state_id",
    "state_mode",
    "state_bytes",
    "num_computed_tokens",
    "swa_valid_start",
    "first_sampled_token",
    "agent_metadata",
)

REQUIRED_AGENT_FIELDS = (
    "engine_id",
    "host",
    "te_rpc_port",
    "segment",
    "blocks_base",
    "states_base",
    "block_bytes",
    "state_bytes",
    "num_blocks",
    "layout_version",
    "registration_generation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock P/D worker for KV handoff")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", default="mock-pd-model")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Directory for request/response JSON captures (default: ./.kv_handoff_capture)",
    )
    parser.add_argument(
        "--handoff-template",
        type=Path,
        default=DEFAULT_HANDOFF_TEMPLATE,
        help="RFC-38 kv_transfer_params template returned by the prefill worker",
    )
    parser.add_argument("--latency-ms", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so the verifier never reads a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_headers(request: Request) -> Dict[str, str]:
    """Capture correlation headers without recording Authorization."""
    keys = ("content-type", "x-request-id", "x-data-parallel-rank")
    captured: Dict[str, str] = {}
    for key in keys:
        value = request.headers.get(key)
        if value is not None:
            captured[key] = value
    return captured


def request_id_from(request: Request, body: Dict[str, Any]) -> str:
    return (
        request.headers.get("x-request-id")
        or str(body.get("request_id") or "")
        or "mock-request"
    )


def chat_response(
    *,
    response_id: str,
    model: str,
    content: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 1,
            "total_tokens": 33,
        },
        **extra,
    }


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    worker_id = args.worker_id or f"{args.role}-{args.port}"
    capture_dir = args.capture_dir or Path.cwd() / ".kv_handoff_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    def capture(name: str, request: Request, body: Dict[str, Any]) -> None:
        write_json(
            capture_dir / name,
            {
                "worker_id": worker_id,
                "worker_role": args.role,
                "path": request.url.path,
                "headers": selected_headers(request),
                "body": body,
            },
        )

    async def prefill(request: Request) -> JSONResponse:
        body = await request.json()
        problems: List[str] = []

        handoff = body.get("kv_transfer_params")
        if not isinstance(handoff, dict):
            problems.append("prefill request is missing kv_transfer_params object")
        else:
            if handoff.get("do_remote_decode") is not True:
                problems.append("prefill request must set do_remote_decode=true")
            if handoff.get("do_remote_prefill") is not False:
                problems.append("prefill request must set do_remote_prefill=false")

        # The router rewrites these fields before calling P.
        if body.get("max_tokens") != 1:
            problems.append("router must force max_tokens=1 for prefill")
        if body.get("min_tokens") != 1:
            problems.append("router must force min_tokens=1 for prefill")
        if body.get("stream") is not False:
            problems.append("router must force stream=false for prefill")
        if "stream_options" in body:
            problems.append("router must remove stream_options for prefill")

        if problems:
            capture("prefill_request.json", request, body)
            return JSONResponse(
                {"error": "invalid prefill request", "problems": problems}, 400
            )

        request_id = request_id_from(request, body)
        handoff = read_json(args.handoff_template)
        # Keep the RFC-38 object shape, but bind the per-request correlation
        # fields so the demo can prove the router did not rewrite them.
        handoff["transfer_id"] = f"transfer-{request_id}"
        handoff["remote_request_id"] = request_id

        response = chat_response(
            response_id=f"chatcmpl-prefill-{request_id}",
            model=args.model,
            content="[prefill] kv handoff published",
            extra={
                "worker_id": worker_id,
                "worker_role": "prefill",
                "kv_transfer_params": handoff,
            },
        )

        capture("prefill_request.json", request, body)
        write_json(capture_dir / "prefill_response.json", {"body": response})
        return JSONResponse(response)

    async def decode(request: Request) -> JSONResponse:
        body = await request.json()
        problems: List[str] = []

        handoff = body.get("kv_transfer_params")
        if not isinstance(handoff, dict):
            problems.append("decode request is missing kv_transfer_params object")
            handoff = {}

        expected_path = capture_dir / "prefill_response.json"
        expected: Optional[Dict[str, Any]] = None
        if not expected_path.exists():
            problems.append(f"prefill response capture not found: {expected_path}")
        else:
            expected = (
                read_json(expected_path).get("body", {}).get("kv_transfer_params")
            )
            if not isinstance(expected, dict):
                problems.append(
                    "prefill response capture has no kv_transfer_params object"
                )

        if expected is not None and handoff != expected:
            problems.append(
                "router did not forward kv_transfer_params unchanged: "
                "decode request differs from prefill response"
            )

        for field in REQUIRED_FIELDS:
            if field not in handoff:
                problems.append(f"kv_transfer_params missing field: {field}")

        if handoff.get("do_remote_decode") is not False:
            problems.append("decode request must set do_remote_decode=false")
        if handoff.get("do_remote_prefill") is not True:
            problems.append("decode request must set do_remote_prefill=true")
        if handoff.get("state_mode") not in (None, "raw_slot"):
            problems.append("decode request must use state_mode=raw_slot")

        agent = handoff.get("agent_metadata")
        if not isinstance(agent, dict):
            problems.append("agent_metadata must be an object")
            agent = {}
        for field in REQUIRED_AGENT_FIELDS:
            if field not in agent:
                problems.append(f"agent_metadata missing field: {field}")

        request_id = request_id_from(request, body)
        if handoff.get("remote_request_id") not in (None, request_id):
            problems.append(
                "remote_request_id does not match X-Request-Id: "
                f"{handoff.get('remote_request_id')!r} != {request_id!r}"
            )

        capture("decode_request.json", request, body)
        if problems:
            return JSONResponse(
                {"error": "invalid decode request", "problems": problems}, 400
            )

        response = chat_response(
            response_id=f"chatcmpl-decode-{request_id}",
            model=args.model,
            content="[decode] kv handoff verified",
            extra={
                "worker_id": worker_id,
                "worker_role": "decode",
                "kv_handoff_ok": True,
                "verified_transfer_id": handoff.get("transfer_id"),
                "verified_block_ids": handoff.get("remote_block_ids"),
            },
        )
        write_json(capture_dir / "decode_response.json", {"body": response})
        return JSONResponse(response)

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/get_server_info")
    async def server_info() -> JSONResponse:
        return JSONResponse(
            {
                "worker_id": worker_id,
                "worker_role": args.role,
                "model": args.model,
                "dp_size": 1,
            }
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": args.model, "object": "model"}],
            }
        )

    @app.get("/get_load")
    async def get_load() -> JSONResponse:
        return JSONResponse({"load": 0})

    @app.post("/flush_cache")
    async def flush_cache() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        if args.latency_ms > 0:
            import asyncio

            await asyncio.sleep(args.latency_ms / 1000.0)
        if args.role == "prefill":
            return await prefill(request)
        return await decode(request)

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(
        create_app(args),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
