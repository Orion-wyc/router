#!/usr/bin/env python3
"""Verify that VllmPDRouter forwarded the RFC-38 KV handoff object unchanged.

Reads the JSON captures written by ``mock_kv_worker.py`` and the client
response produced by ``run_kv_handoff_demo.sh``.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    parser = argparse.ArgumentParser(description="Verify the KV handoff router demo")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--response", type=Path, required=True, help="Client response JSON"
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    capture = args.capture_dir

    required_captures = (
        "prefill_request.json",
        "prefill_response.json",
        "decode_request.json",
        "decode_response.json",
    )
    missing = [name for name in required_captures if not (capture / name).exists()]
    if missing:
        print(f"missing captures: {', '.join(missing)}", file=sys.stderr)
        return 1

    prefill_request = load_json(capture / "prefill_request.json")
    prefill_response = load_json(capture / "prefill_response.json")
    decode_request = load_json(capture / "decode_request.json")
    decode_response = load_json(capture / "decode_response.json")
    client_response = load_json(args.response)

    prefill_body = prefill_request.get("body") or {}
    decode_body = decode_request.get("body") or {}
    prefill_handoff = prefill_body.get("kv_transfer_params") or {}
    decode_handoff = decode_body.get("kv_transfer_params") or {}
    returned_handoff = (prefill_response.get("body") or {}).get(
        "kv_transfer_params"
    ) or {}

    errors: List[str] = []
    checks: List[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            checks.append(message)
        else:
            errors.append(message)

    # 1. Router -> P request shape. The router must force a single-token,
    # non-streaming prefill and set the P-side flags.
    check(
        prefill_handoff.get("do_remote_decode") is True,
        "prefill request: do_remote_decode=true",
    )
    check(
        prefill_handoff.get("do_remote_prefill") is False,
        "prefill request: do_remote_prefill=false",
    )
    check(prefill_body.get("max_tokens") == 1, "prefill request: max_tokens=1")
    check(prefill_body.get("min_tokens") == 1, "prefill request: min_tokens=1")
    check(prefill_body.get("stream") is False, "prefill request: stream=false")
    check(
        "stream_options" not in prefill_body, "prefill request: stream_options removed"
    )

    # 2. P response carries the RFC-38 §7.3 object.
    for field in REQUIRED_FIELDS:
        check(
            field in returned_handoff,
            f"prefill response: kv_transfer_params.{field} present",
        )

    agent = returned_handoff.get("agent_metadata")
    if not isinstance(agent, dict):
        errors.append("prefill response: agent_metadata is an object")
        agent = {}
    for field in REQUIRED_AGENT_FIELDS:
        check(
            field in agent,
            f"prefill response: agent_metadata.{field} present",
        )

    check(
        returned_handoff.get("do_remote_decode") is False,
        "prefill response: do_remote_decode=false (D-side value)",
    )
    check(
        returned_handoff.get("do_remote_prefill") is True,
        "prefill response: do_remote_prefill=true (D-side value)",
    )
    check(
        returned_handoff.get("state_mode") == "raw_slot",
        "prefill response: state_mode=raw_slot",
    )
    check(
        returned_handoff.get("remote_block_ids") == [17, 18],
        "prefill response: remote_block_ids=[17, 18]",
    )
    check(
        returned_handoff.get("num_prompt_blocks") == 2,
        "prefill response: num_prompt_blocks=2",
    )
    check(
        agent.get("layout_version") == "example-layout-v1",
        "prefill response: layout_version=example-layout-v1",
    )

    prefill_request_id = (prefill_request.get("headers") or {}).get("x-request-id")
    if prefill_request_id:
        check(
            returned_handoff.get("remote_request_id") == prefill_request_id,
            "prefill response: remote_request_id matches X-Request-Id",
        )
        check(
            returned_handoff.get("transfer_id") == f"transfer-{prefill_request_id}",
            "prefill response: transfer_id is bound to the request",
        )

    # 3. Router -> D request must carry the same object, bit-for-bit at the
    # JSON-value level. This is the passthrough assertion.
    check(
        decode_handoff == returned_handoff,
        "decode request: kv_transfer_params equals prefill response object",
    )
    check(
        decode_handoff.get("do_remote_decode") is False,
        "decode request: do_remote_decode=false",
    )
    check(
        decode_handoff.get("do_remote_prefill") is True,
        "decode request: do_remote_prefill=true",
    )
    decode_request_id = (decode_request.get("headers") or {}).get("x-request-id")
    if decode_request_id:
        check(
            decode_handoff.get("remote_request_id") == decode_request_id,
            "decode request: remote_request_id matches X-Request-Id",
        )

    # 4. D accepted the handoff and returned its marker.
    decode_response_body = decode_response.get("body") or {}
    check(
        decode_response_body.get("kv_handoff_ok") is True,
        "decode response: kv_handoff_ok=true",
    )
    check(
        decode_response_body.get("verified_transfer_id")
        == returned_handoff.get("transfer_id"),
        "decode response: verified_transfer_id matches P transfer_id",
    )
    check(
        decode_response_body.get("verified_block_ids") == [17, 18],
        "decode response: verified_block_ids=[17, 18]",
    )
    check(
        decode_response_body.get("worker_role") == "decode",
        "decode response: worker_role=decode",
    )

    # 5. The client sees the decode worker's response, not the prefill one.
    check(
        client_response.get("worker_role") == "decode",
        "client response: worker_role=decode",
    )
    check(
        client_response.get("kv_handoff_ok") is True,
        "client response: kv_handoff_ok=true",
    )
    check(
        client_response.get("verified_transfer_id")
        == returned_handoff.get("transfer_id"),
        "client response: verified_transfer_id matches P transfer_id",
    )
    choices = client_response.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        content = (choices[0].get("message") or {}).get("content") or ""
    check("[decode]" in content, "client response: content comes from the decode mock")

    print("KV handoff passthrough checks")
    print("=" * 32)
    for message in checks:
        print(f"[PASS] {message}")
    if errors:
        print()
        for message in errors:
            print(f"[FAIL] {message}", file=sys.stderr)
        print(f"\n{len(errors)} check(s) failed.", file=sys.stderr)
        return 1

    print()
    print(f"{len(checks)} checks passed.")
    print(f"transfer_id: {returned_handoff.get('transfer_id')}")
    print(f"block ids:   {returned_handoff.get('remote_block_ids')}")
    print(f"segment:     {agent.get('segment')}")
    print("Router forwarded the RFC-38 kv_transfer_params object unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
