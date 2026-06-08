# Token Monitor

OpenAI 兼容接口的 token 用量监控网关。部署在 Coder 与上游大模型服务之间，透明代理所有请求并精确统计 token 用量。

## 架构

```
Coder ──► Monitor (localhost:9100) ──► 上游服务商 (OpenAI API)
                 │
                 ▼
          token_usage.jsonl
```

Monitor 作为本地 HTTP 反向代理运行，支持流式（SSE）和非流式两种响应模式。每次请求完成后，将 token 用量（prompt / completion / reasoning / cache hit）追加写入 JSONL 日志。

## 启动

```bash
python monitor --provider kimi --api-key <KEY>
python monitor --provider zhipu --api-key <KEY> --model glm-5.1
python monitor --port 9100 --provider deepseek --api-key <KEY>
python monitor --provider minimax --api-key <KEY> --model MiniMax-M3
python monitor --provider minimax --api-key <KEY> --model MiniMax-M2.7
python monitor --summary token_usage.jsonl
```

MiniMax models must use official full model IDs, for example `MiniMax-M3` or `MiniMax-M2.7`.

启动后将 Coder 连接到本地网关：

```
/connect custom
BaseURL: http://127.0.0.1:9100/v1/chat/completions
API Key: any（网关自行使用上游 Key）
```

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--provider` | 上游服务商 ID | `kimi` |
| `--api-key` | 上游 API Key | 非 `--summary` 模式必填 |
| `--model` | 覆盖请求中的模型名称 | 不覆盖 |
| `--endpoint` | 自定义上游 endpoint（覆盖 provider） | 按 provider 解析 |
| `--host` | 监听地址 | `127.0.0.1` |
| `--port` | 监听端口 | `9100` |
| `--log-file` | token 用量日志路径 | `token_usage.jsonl` |
| `--summary` | 读取 JSONL 日志并输出汇总报告后退出 | 关闭 |

## 离线汇总

已有 JSONL 日志时，可直接离线统计整体开销数据：

```bash
python monitor --summary token_usage.jsonl
```

输出维度与运行结束时的汇总报告一致，包括总请求数、总耗时、Prompt Tokens、缓存命中、有效输入、Completion Tokens、Reasoning Tokens、总 Tokens 和开销当量 Tokens。

## 支持的服务商

| ID | 名称 | Endpoint |
| --- | --- | --- |
| `huawei` | 华为云 MaaS | `api.modelarts-maas.com` |
| `kimi` | 月之暗面 Moonshot | `api.moonshot.cn` |
| `zhipu` | 智谱 GLM | `open.bigmodel.cn` |
| `deepseek` | DeepSeek | `api.deepseek.com` |
| `minimax` | MiniMax | `api.minimaxi.com` |

## 日志格式

每行一条 JSON 记录，包含时间戳、模型、token 用量明细：

```json
{
  "timestamp": "2026-05-27T08:00:00Z",
  "model": "glm-5.1",
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "reasoning_tokens": 89,
  "cached_tokens": 0,
  "total_tokens": 1801
}
```

## 模块结构

```
monitor/
├── __init__.py
├── __main__.py         入口（调用 main.main）
├── main.py             命令行解析与启动
├── config.py           服务商路由表与配置
├── proxy.py            HTTP 反向代理服务端
├── sse_handler.py      SSE 流式响应处理
├── upstream.py         上游请求转发
├── token_stats.py      token 统计与 JSONL 日志
└── e2etest/            端到端测试
    └── run_token_monitor.py
```

## 端到端测试

```bash
# 默认服务商（kimi）
python monitor/e2etest/run_token_monitor.py

# 指定服务商
python monitor/e2etest/run_token_monitor.py --provider zhipu --model glm-5.1
python monitor/e2etest/run_token_monitor.py --provider deepseek --model deepseek-v4-pro
```

测试流程：启动 Monitor 网关 → 启动 Coder 连接网关 → 执行 LRU Cache 修复任务 → 验证 token 日志写入正确。
