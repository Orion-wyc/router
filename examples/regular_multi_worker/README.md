# 混步（非 PD 分离）多 worker 本地 demo

这个 demo 用 mock worker 在 CPU 环境跑通 **Regular 模式**的多 worker 路由。Regular 模式下不开启 `--vllm-pd-disaggregation`，每个 worker 是一个独立的 vLLM 实例，prefill 和 decode 在同一个实例内完成（即“混步”），router 只负责把完整请求分发给其中一个实例。

```
客户端 -> vllm-router (Regular 模式)
              |-> regular-1 (prefill + decode)
              |-> regular-2 (prefill + decode)
              `-> regular-3 (prefill + decode)
```

它验证的是 router 的多 worker 注册、健康检查、请求分发和流式转发，不涉及真实模型推理。

## 和 PD demo 的区别

| | Regular（本 demo） | PD 分离（`../pd_disaggregation`） |
|---|---|---|
| router 开关 | 无，默认模式 | `--vllm-pd-disaggregation` |
| worker 参数 | `--worker-urls URL...` | `--prefill URL [PORT]` / `--decode URL` |
| 请求路径 | router 选一个 worker，请求整体转发 | router 分别调度 prefill 和 decode，并传 KV |
| 实例职责 | 实例内部完成 prefill + decode | prefill 实例和 decode 实例职责分离 |

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
./examples/regular_multi_worker/run_regular_demo.sh
```

脚本会：

1. 启动 3 个 regular mock worker（18081、18082、18083），每个都同时处理 prefill 和 decode。
2. 启动 Regular router（18090），通过 `--worker-urls` 注册这 3 个 worker。
3. 发送 12 个 `/v1/completions` 请求。
4. 发送 1 个流式 `/v1/chat/completions` 请求，校验 SSE 以 `[DONE]` 结束。
5. 校验所有响应都来自 regular worker，且至少命中两个 worker。
6. 默认保持运行，按 `Ctrl+C` 停止。

只做一次检查并退出：

```bash
KEEP_RUNNING=0 ./examples/regular_multi_worker/run_regular_demo.sh
```

## 预期输出

```text
=== starting mock workers ===
starting regular worker regular-1 on 127.0.0.1:18081
starting regular worker regular-2 on 127.0.0.1:18082
starting regular worker regular-3 on 127.0.0.1:18083
=== starting regular router on 127.0.0.1:18090 ===
workers: 18081 18082 18083
policy:  round_robin
=== sending requests through the router ===
request 01: status=200 role=regular worker=regular-1 text='[regular] ok'
...
regular worker distribution: {'regular-1': 4, 'regular-2': 4, 'regular-3': 4}
unique workers hit: 3
Regular multi-worker routing checks passed.
Router is running at http://127.0.0.1:18090
```

手动请求：

```bash
curl -s http://127.0.0.1:18090/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-regular-model","prompt":"hello","max_tokens":4}' | python3 -m json.tool
```

响应中的 `worker_role` 应为 `regular`，`worker_id` 会随负载均衡策略变化。

## 可配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `WORKER_PORTS` | `18081 18082 18083` | regular worker 端口列表，数量即 worker 实例数 |
| `ROUTER_PORT` | `18090` | router 监听端口 |
| `METRICS_PORT` | `19090` | Prometheus 端口 |
| `POLICY` | `round_robin` | 路由策略，可选 `random`、`power_of_two`、`cache_aware`、`consistent_hash`、`kv_aware` |
| `MODEL` | `mock-regular-model` | mock worker 对外声明的模型名 |
| `WORKER_STARTUP_TIMEOUT` | `30` | 等待 worker 就绪的超时秒数 |
| `PYTHON` | `python3` | 启动 mock worker 的 Python |
| `ROUTER_BIN` | `target/release/vllm-router` | router 二进制路径 |
| `KEEP_RUNNING` | `1` | `0` 表示检查完自动退出 |
| `LOG_DIR` | 自动创建 | worker/router 日志目录 |

示例：

```bash
WORKER_PORTS="18081 18082 18083 18084" \
POLICY=power_of_two \
KEEP_RUNNING=0 \
./examples/regular_multi_worker/run_regular_demo.sh
```

`power_of_two` 至少需要 2 个 worker。

## 接入真实 vLLM 混步实例

每个 worker 都是普通 vLLM OpenAI Server，不需要 KV connector。例如启动 3 个实例（不同 GPU / 端口）：

```bash
# 终端 1
vllm serve <model> --port 8001 --tensor-parallel-size 1

# 终端 2
vllm serve <model> --port 8002 --tensor-parallel-size 1

# 终端 3
vllm serve <model> --port 8003 --tensor-parallel-size 1
```

router 侧：

```bash
vllm-router \
  --host 0.0.0.0 \
  --port 8090 \
  --policy cache_aware \
  --worker-urls http://127.0.0.1:8001 http://127.0.0.1:8002 http://127.0.0.1:8003
```

如果单个实例内部还开了数据并行（一个 URL 背后有 N 个 DP rank），再加：

```bash
--intra-node-data-parallel-size N
```

注意该参数描述的是每个 worker URL 背后的内部 DP rank 数，不是 worker 实例数；多个实例请继续用 `--worker-urls` 列出。

## 文件说明

- `run_regular_demo.sh`：启动多 worker 和 Regular router，完成健康检查、请求验证和清理。
- `../pd_disaggregation/mock_worker.py`：复用的最小 OpenAI 兼容 mock worker，`--role regular` 时模拟混步实例，支持 `/health`、`/v1/models`、`/v1/completions`、`/v1/chat/completions`、流式响应。
