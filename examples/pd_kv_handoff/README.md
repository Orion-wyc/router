# RFC-38 KV handoff passthrough demo

这个示例用两个 mock worker 验证 `VllmPDRouter` 的 P/D 交接控制面：

```text
client
  -> vllm-router (PD, --kv-connector nixl)
       -> prefill mock: 返回 RFC-38 §7.3 的 kv_transfer_params
       -> decode mock:  校验 Router 是否原样转发了这份对象
```

它不跑模型，也不搬 KV 字节。验证点是：

- Router 发给 P 的请求被改写成 `max_tokens=1`、`min_tokens=1`、`stream=false`，并带上 P 侧的 `do_remote_decode=true`、`do_remote_prefill=false`。
- P 响应里的 `kv_transfer_params` 是 RFC-38 §7.3 的完整对象（含 `agent_metadata`）。
- Router 把它原样放进 D 请求的同名字段，字段不丢、不改名、不改值。
- D 校验通过后返回响应，客户端能看到 D 的标记。

## 为什么用 `--kv-connector nixl`

vllm-router 的 `--kv-connector` 是**控制面交接形态**的选择，不是数据面传输库。`nixl` 分支的行为是“P 响应携带 `kv_transfer_params`，Router 原样透传给 D”，正好对应 RFC-38 的 pull 交接；它不会加载或调用 NIXL。

在 Ascend + DeepInfer + Mooncake TE 的场景里，实际 KV 传输由 DeepInfer 的 `MooncakeConnector` 完成，Router 只负责这份参数。`mooncake` 分支要求 P 暴露 bootstrap `/query`，并按 vLLM Mooncake push 协议生成 5 个字段，和 RFC-38 的对象不是一套；`moriio` 分支要求 ZMQ 发现。

## 文件

| 文件 | 作用 |
|---|---|
| `kv_transfer_params.rfc38.json` | RFC-38 §7.3 的 `kv_transfer_params` 样例，P mock 以它为模板 |
| `mock_kv_worker.py` | P/D mock worker；P 返回模板对象，D 校验 Router 透传是否一致 |
| `verify_handoff.py` | 读取请求/响应捕获，逐项断言 |
| `run_kv_handoff_demo.sh` | 启动 P、D、Router，发一条请求并执行校验 |

## 运行

准备 Python 依赖和 Router 二进制：

```bash
source /mnt/workspace/Envs/vllm_dev/env.sh   # 或任意带 fastapi/uvicorn 的环境
cargo build --release --bin vllm-router
```

执行：

```bash
./examples/pd_kv_handoff/run_kv_handoff_demo.sh
```

可用环境变量覆盖端口和二进制：

```bash
PREFILL_PORT=18181 \
DECODE_PORT=18182 \
ROUTER_PORT=18190 \
ROUTER_BIN=/path/to/vllm-router \
./examples/pd_kv_handoff/run_kv_handoff_demo.sh
```

脚本默认使用 `target/release/vllm-router`，找不到时回落到 `PATH` 里的 `vllm-router`。

## 预期输出

```text
=== starting mock P/D workers ===
starting prefill worker prefill-1 on 127.0.0.1:18181
starting decode worker decode-1 on 127.0.0.1:18182
=== starting PD router on 127.0.0.1:18190 ===
=== sending request through the router ===
=== verifying kv_transfer_params passthrough ===
KV handoff passthrough checks
================================
[PASS] prefill request: do_remote_decode=true
[PASS] prefill request: do_remote_prefill=false
[PASS] prefill request: max_tokens=1
[PASS] prefill request: min_tokens=1
[PASS] prefill request: stream=false
[PASS] prefill request: stream_options removed
...
[PASS] decode request: kv_transfer_params equals prefill response object
...
51 checks passed.
transfer_id: transfer-...
block ids:   [17, 18]
segment:     192.0.2.10:19001
Router forwarded the RFC-38 kv_transfer_params object unchanged.

RFC-38 KV handoff demo passed.
```

脚本结束时打印 `logs:` 和 `captures:` 路径，可直接查看：

- `capture/prefill_request.json`：Router 发给 P 的请求。
- `capture/prefill_response.json`：P 返回的完整响应。
- `capture/decode_request.json`：Router 发给 D 的请求。
- `capture/decode_response.json`：D 的原始响应。
- `router.log`：Router 的 debug 日志，包含两个阶段的 payload。

## RFC-38 对象

模板取自 RFC-38 §7.3，字段含义：

| 字段 | 本示例的值 | 说明 |
|---|---|---|
| `do_remote_decode` / `do_remote_prefill` | `false` / `true` | P 返回给 D 的最终形态；Router 沿用 P 的值 |
| `transfer_id` | `transfer-<X-Request-Id>` | 一次交接的幂等键，P 生成 |
| `remote_request_id` | `<X-Request-Id>` | P 请求 ID，用于关联 |
| `remote_engine_id` | `prefill-0` | P 实例 ID |
| `source_rank` | `0` | P rank，诊断用 |
| `remote_block_ids` / `num_prompt_blocks` | `[17,18]` / `2` | P 侧 pinned 的源块 |
| `remote_state_id` / `state_mode` / `state_bytes` | `3` / `raw_slot` / `2048` | P 侧 state slot |
| `num_computed_tokens` / `swa_valid_start` / `first_sampled_token` | `32` / `0` / `1234` | 续算位置、SWA 起点、首 token |
| `agent_metadata` | 见模板 | `segment`、`blocks_base`、`states_base`、`block_bytes`、`layout_version`、`registration_generation` 等 |

P mock 只覆盖 `transfer_id` 和 `remote_request_id`，其余字段保持模板值。D mock 用深比较确认收到的对象和 P 返回的完全一致。

## 不覆盖的内容

- 真实 KV/state 传输：D mock 不会发起 TE Read，只校验参数。
- D 的本地 block/state 分配、`WAITING_FOR_REMOTE_KV` 状态机。
- Router 的 P 参数校验和 D prompt 构造：当前 0.1.17 的 `nixl` 分支只做透传，不校验必填字段，也不补 `expanded_prompt_token_ids + [first_sampled_token]`。这些属于 RFC-31 V20 的改造范围。
- `transfer_mode=push`：示例不设置这个字段，避免 Router 进入 NIXL push 身份缓存和并发快路径。

要接真实 DeepInfer，把 P mock 的返回值换成 DeepInfer 引擎生成的 `kv_transfer_params`，D 侧改为 `MooncakeConnector` 的 Pull 入口即可；Router 侧仍可用 `--kv-connector nixl` 验证控制面透传。
