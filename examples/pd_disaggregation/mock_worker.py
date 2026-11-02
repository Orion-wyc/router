#!/usr/bin/env python3
"""Minimal OpenAI-compatible worker used by the local PD demo.

This worker does not run a model. It only returns deterministic responses so
that the router's prefill/decode routing can be exercised on a CPU-only host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock prefill/decode worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--model", default="mock-pd-model")
    parser.add_argument("--latency-ms", type=int, default=0)
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    worker_id = args.worker_id or f"{args.role}-{args.port}"
    inflight = {"count": 0}

    def response_headers() -> dict[str, str]:
        return {
            "X-Worker-Id": worker_id,
            "X-Worker-Role": args.role,
        }

    async def maybe_delay() -> None:
        if args.latency_ms > 0:
            await asyncio.sleep(args.latency_ms / 1000.0)

    def completion_body(path: str, request_data: dict) -> dict:
        now = int(time.time())
        text = f"[{args.role}] ok"
        if path.endswith("/chat/completions"):
            choices = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
            obj = "chat.completion"
        else:
            choices = [{"index": 0, "text": text, "finish_reason": "stop"}]
            obj = "text_completion"
        return {
            "id": f"cmpl-{now}-{worker_id}",
            "object": obj,
            "created": now,
            "model": args.model,
            "choices": choices,
            "worker_id": worker_id,
            "worker_role": args.role,
            "echo": request_data,
        }

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/health_generate")
    async def health_generate() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.post("/flush_cache")
    async def flush_cache() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": args.model, "object": "model"}],
                "worker_id": worker_id,
                "worker_role": args.role,
            },
            headers=response_headers(),
        )

    @app.get("/get_server_info")
    async def server_info() -> JSONResponse:
        return JSONResponse(
            {
                "worker_id": worker_id,
                "worker_role": args.role,
                "model": args.model,
                "dp_size": 1,
                "load_in_flight": inflight["count"],
            },
            headers=response_headers(),
        )

    @app.get("/get_load")
    async def get_load() -> JSONResponse:
        return JSONResponse(
            {"load": inflight["count"], "worker_id": worker_id},
            headers=response_headers(),
        )

    async def handle_inference(request: Request):
        inflight["count"] += 1
        try:
            await maybe_delay()
            try:
                data = await request.json()
            except (json.JSONDecodeError, ValueError):
                data = {}

            if data.get("stream"):
                return await stream_response(request.url.path, data)

            body = completion_body(request.url.path, data)
            return JSONResponse(body, headers=response_headers())
        finally:
            inflight["count"] -= 1

    async def stream_response(path: str, request_data: dict) -> StreamingResponse:
        is_chat = path.endswith("/chat/completions")

        async def events():
            for token in ("[", args.role, "] ok"):
                if is_chat:
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": token}}],
                        "worker_id": worker_id,
                        "worker_role": args.role,
                    }
                else:
                    chunk = {
                        "choices": [{"index": 0, "text": token}],
                        "worker_id": worker_id,
                        "worker_role": args.role,
                    }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers=response_headers(),
        )

    @app.post("/generate")
    async def generate(request: Request):
        return await handle_inference(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle_inference(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle_inference(request)

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
