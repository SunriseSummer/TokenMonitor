# Monitor：轻量 Token 网关

`monitor` 是一个本地 Token 网关。它部署在客户端工具和上游大模型服务之间，向下游提供 OpenAI 兼容的 `chat/completions` 接口，同时把请求转发给真实模型服务商，并记录每次请求的 token 开销，实时呈现汇总数据。

## 基本原理

调用链如下：

```text
客户端工具
  -> http://127.0.0.1:9100/v1/chat/completions
  -> monitor
  -> 上游模型服务商
  -> monitor 统计 usage
  -> 客户端工具
```

monitor 本身不生成模型结果，只做三件事：

1. 接收下游 OpenAI 兼容请求。
2. 按 `--provider`、`--model`、`--api-key` 转发到上游模型服务商。
3. 从上游响应中的 `usage` 字段提取 token 数据，写入 `token_usage.jsonl`，并在终端显示汇总。

流式请求会自动补充：

```json
{
  "stream_options": {
    "include_usage": true
  }
}
```

这样支持该选项的上游会在流式响应末尾返回完整 usage。对于 MiniMax 等模型，monitor 还会把响应正文中的 `<think>...</think>` 转成 OpenAI/DeepSeek 风格的 `reasoning_content` 字段，避免客户端把思考内容当正文显示。

## 启动 monitor

在当前目录执行：

```shell
python monitor --provider deepseek --model deepseek-v4-pro --api-key <API_KEY>
```

MiniMax 示例：

```shell
python monitor --provider minimax --model MiniMax-M3 --api-key <API_KEY>
```

默认监听地址：

```text
http://127.0.0.1:9100/v1/chat/completions
```

停止服务时，在 monitor 终端按 `Ctrl+C`。停止后会输出本次运行的 token 汇总。

## 支持的服务商

| Provider | Endpoint |
| --- | --- |
| `huawei` | `https://api.modelarts-maas.com/v2/chat/completions` |
| `kimi` | `https://api.moonshot.cn/v1/chat/completions` |
| `deepseek` | `https://api.deepseek.com/chat/completions` |
| `minimax` | `https://api.minimaxi.com/v1/chat/completions` |
| `zhipu` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` |

也可以使用自定义上游地址：

```shell
python monitor --endpoint https://example.com/v1/chat/completions --model <MODEL> --api-key <API_KEY>
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--provider` | 内置上游服务商 ID | `kimi` |
| `--model` | 覆盖下游请求中的模型名，转发给上游 | 空 |
| `--api-key` | 上游服务商 API Key | 必填 |
| `--endpoint` | 自定义上游 endpoint，优先级高于 provider | 空 |
| `--host` | 本地监听地址 | `127.0.0.1` |
| `--port` | 本地监听端口 | `9100` |
| `--log-file` | token 日志文件 | `token_usage.jsonl` |
| `--summary` | 读取 JSONL 日志并输出汇总 | 空 |


## 接入客户端工具

在 opencode 或其他支持 OpenAI 兼容接口的工具中配置：

```text
Base URL: http://127.0.0.1:9100/v1/chat/completions
API Key: 任意非空字符串
Model: 任意字符串
```

下游填写的 API Key 不会用于访问上游。monitor 使用启动命令中的 `--api-key` 调用真实模型服务。

如果 monitor 启动时传了 `--model`，下游请求里的 `model` 会被覆盖。这样客户端可以固定写一个占位模型名，由 monitor 统一决定真实上游模型。

## 查看 token 开销

monitor 会把每次请求写入 JSONL 日志，默认文件是：

```text
token_usage.jsonl
```

每行是一条请求记录，包含：

```json
{
  "request_id": "...",
  "model": "MiniMax-M3",
  "prompt_tokens": 183,
  "completion_tokens": 32,
  "total_tokens": 215,
  "cached_tokens": 169,
  "effective_prompt_tokens": 14,
  "reasoning_tokens": 0,
  "duration": 3.09,
  "stream": true
}
```

运行 monitor 的终端窗口中，会实时刷新显示 token 开销汇总数据，也可以离线查看汇总数据：

```shell
python monitor --summary token_usage.jsonl
```

## 使用 test.py 验证

当前目录提供了一个简单聊天工具 `test.py`，它默认连接`http://127.0.0.1:9100/v1/chat/completions`，用来验证 monitor 是否可用。

先启动 monitor：

```shell
python monitor --provider minimax --model MiniMax-M3 --api-key <API_KEY>
```

再执行一次性测试：

```shell
python test.py --once "Think briefly, then answer OK." --assert-no-think-tags
```

如果成功，会看到模型回复、`reasoning_content`、usage 和 `PASS`。

也可以进入交互聊天：

```shell
python test.py
```

## 常见问题

### 为什么客户端 API Key 可以随便填？

monitor 是本地网关。下游客户端请求 monitor 时只需要通过客户端自己的校验；真正访问上游模型服务商时，monitor 使用启动参数中的 `--api-key`。

### 为什么流式请求过程中 token 一直不变？

monitor 只有在一次请求结束、拿到上游 usage 后才记录 token。流式生成中间不会估算 token。

### MiniMax 为什么要特殊处理思考内容？

DeepSeek 等模型通常把思考内容放在 `reasoning_content` 字段里。MiniMax 可能把思考内容写进正文并用 `<think>...</think>` 包裹。monitor 会把 MiniMax 的 `<think>` 块转换为 `reasoning_content`，让 opencode 等客户端能正确分离思考内容和正文。

### 使用 MiniMax-M2.6 失败怎么办？

以服务商模型列表和账号权限为准。当前验证过的模型包括 `MiniMax-M3` 和 `MiniMax-M2.7`。如果某个模型返回 `unknown model`，说明该模型在当前 endpoint 或账号下不可用。
