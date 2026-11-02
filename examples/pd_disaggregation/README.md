# PD 分离多 worker 本地 demo

这个 demo 用 mock worker 在 CPU 环境跑通 **Prefill / Decode 分离** 路由：

```
客户端 -> vllm-router (PD 模式)
              |-> prefill-1, prefill-2
              `-> decode-1, decode-2, decode-3
```

它验证的是 router 的 PD 调度、worker 注册、健康检查和响应转发，不涉及真实模型推理和 KV 传输。

## 已有真实 PD 示例

仓库里已有需要 GPU 和真实 vLLM 的 PD 配置，本 demo 不替代它们：

- `scripts/llama3.1/start_prefill.sh`、`start_decode.sh`、`start_router.sh`：单机 NIXL PD 示例。
- `scripts/k8s/pd_disagg/`：Kubernetes 下的 PD router 部署。
- `tests/python/e2e/test_pd_router.py`：需要 CUDA 和 vLLM 的端到端测试。

本 demo 的目标是：没有 GPU 也能快速确认 PD 路由链路是否工作。

## 快速开始

先准备环境：

```bash
source /mnt/workspace/Envs/vllm_dev/env.sh
```

或使用任意已安装项目依赖的 Python 环境：

```bash
pip install -e '.[dev]'
cargo build --release --bin vllm-router
```

运行：

```bash
./examples/pd_disaggregation/run_pd_demo.sh
```

脚本会：

1. 启动 2 个 prefill mock worker（18081、18082）。
2. 启动 3 个 decode mock worker（18083、18084、18085）。
3. 启动 PD router（18090）。
4. 发送 12 个 `/v1/completions` 请求。
5. 校验响应都来自 decode worker，且至少命中两个 decode worker。
6. 默认保持运行，按 `Ctrl+C` 停止。

只做一次检查并退出：

```bash
KEEP_RUNNING=0 ./examples/pd_disaggregation/run_pd_demo.sh
```

## 预期输出

```text
=== starting mock workers ===
starting prefill worker prefill-1 on 127.0.0.1:18081
starting prefill worker prefill-2 on 127.0.0.1:18082
starting decode worker decode-1 on 127.0.0.1:18083
starting decode worker decode-2 on 127.0.0.1:18084
starting decode worker decode-3 on 127.0.0.1:18085
=== starting PD router on 127.0.0.1:18090 ===
prefill: 18081 18082
decode:  18083 18084 18085
=== sending requests through the router ===
request 01: status=200 role=decode worker=decode-1 text='[decode] ok'
...
decode worker distribution: {'decode-1': 4, 'decode-2': 4, 'decode-3': 4}
PD routing checks passed.
Router is running at http://127.0.0.1:18090
```

手动请求：

```bash
curl -s http://127.0.0.1:18090/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-pd-model","prompt":"hello","max_tokens":4}' | python3 -m json.tool
```

响应中的 `worker_role` 应为 `decode`。

## 可配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `PREFILL_PORTS` | `18081 18082` | prefill worker 端口列表 |
| `DECODE_PORTS` | `18083 18084 18085` | decode worker 端口列表 |
| `ROUTER_PORT` | `18090` | router 监听端口 |
| `METRICS_PORT` | `19090` | Prometheus 端口 |
| `POLICY` | `round_robin` | 路由策略，可选 `random`、`power_of_two`、`cache_aware`、`consistent_hash` |
| `PYTHON` | `python3` | 启动 mock worker 的 Python |
| `ROUTER_BIN` | `target/release/vllm-router` | router 二进制路径 |
| `KEEP_RUNNING` | `1` | `0` 表示检查完自动退出 |
| `LOG_DIR` | 自动创建 | worker/router 日志目录 |

示例：

```bash
POLICY=power_of_two KEEP_RUNNING=0 ./examples/pd_disaggregation/run_pd_demo.sh
```

## 文件说明

- `mock_worker.py`：最小 OpenAI 兼容 mock worker，支持 `/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`、流式响应。
- `run_pd_demo.sh`：启动多 worker 和 PD router，完成健康检查、请求验证和清理。

## 替换为真实 vLLM PD

把 `run_pd_demo.sh` 里的 mock worker 启动部分替换为真实 vLLM：

1. 按 `scripts/llama3.1/start_prefill.sh` 启动 prefill worker，记录 bootstrap port。
2. 按 `scripts/llama3.1/start_decode.sh` 启动 decode worker。
3. router 参数改成：

```bash
vllm-router \
  --policy round_robin \
  --vllm-pd-disaggregation \
  --prefill http://127.0.0.1:8081 9000 \
  --prefill http://127.0.0.1:8082 9001 \
  --decode http://127.0.0.1:8083 \
  --decode http://127.0.0.1:8084 \
  --host 127.0.0.1 \
  --port 8090
```

真实 PD 需要 vLLM 配置 NIXL/Mooncake 等 KV connector，并依赖 GPU、RDMA/NCCL 环境；本 demo 不做 KV 传输，不能用于性能评估。
