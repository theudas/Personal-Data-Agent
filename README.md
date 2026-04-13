# Personal Data Assistant Agent (Ollama + Qwen3 Tool Calling)

一个可运行的个人数据助手 Agent，支持：
- 完整 Agent Loop：输入解析 -> 工具选择 -> 参数生成 -> 执行 -> 结果整合
- 多步工具调用（单轮内可多次调用工具）
- 错误处理与重试（模型调用与工具执行都支持）
- 文件型向量库（不使用数据库）
- 严格目录访问控制（只能访问用户指定目录）
- 仅处理 `.txt` / `.md` / `.markdown`

## 1. 项目结构

```text
.
├── main.py
├── streamlit_app.py
├── README.md
├── requirements.txt
├── requirement.txt
└── personal_data_agent
    ├── __init__.py
    ├── config.py
    ├── agent
    │   ├── __init__.py
    │   ├── loop.py
    │   └── prompts.py
    ├── api
    │   ├── __init__.py
    │   └── cli.py
    ├── runtime
    │   ├── __init__.py
    │   ├── errors.py
    │   ├── retry.py
    │   └── schema.py
    ├── tools
    │   ├── __init__.py
    │   └── note_tools.py
    └── vector_store
        ├── __init__.py
        ├── chunking.py
        └── index.py
```

## 2. 环境准备

- Python 3.10+
- 本地可运行 Ollama（OpenAI 兼容接口）
- 本地已准备 embedding 模型目录：`./bge-base-zh-v1.5`

### 2.1 安装 Python 依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 安装 Ollama 并拉取对话模型

安装 Ollama：

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows 请从官网下载安装包
```

检查 Ollama 版本：

```bash
ollama --version
```

打开一个终端，启动 Ollama 服务：

```bash
ollama serve
```

再打开另一个终端，拉取项目默认使用的模型：

```bash
ollama pull modelscope.cn/dDreamer/qwen3-8b-toolcalling-1e-4-r16-500steps-gguf
```

下载完成后，可以查看本地模型列表：

```bash
ollama list
```

如果模型下载成功，能看到类似下面的输出，关键是 `NAME` 和项目配置一致：

```text
NAME                                                                 ID              SIZE      MODIFIED
modelscope.cn/dDreamer/qwen3-8b-toolcalling-1e-4-r16-500steps-gguf:latest    6f6bdd2fd25b    5.0 GB    6 minutes ago
```

可以用下面的命令直接测试启动模型：

```bash
ollama run modelscope.cn/dDreamer/qwen3-8b-toolcalling-1e-4-r16-500steps-gguf:latest
```

如果之后想删除这个模型：

```bash
ollama rm modelscope.cn/dDreamer/qwen3-8b-toolcalling-1e-4-r16-500steps-gguf:latest
```

### 2.3 下载 `bge-base-zh-v1.5` embedding 模型

这个项目里的向量检索使用 `SentenceTransformer` 按本地目录加载 embedding 模型，因此需要先把模型下载到本地，例如下载到项目根目录下的 `./bge-base-zh-v1.5`。

先安装 Hugging Face CLI：

```bash
pip install -U "huggingface_hub[cli]"
```

然后下载模型到项目默认目录：

```bash
hf download BAAI/bge-base-zh-v1.5 --local-dir ./bge-base-zh-v1.5
```

下载完成后，可以简单检查目录是否存在关键文件：

```bash
ls ./bge-base-zh-v1.5
```

正常情况下，你会看到类似 `config.json`、`modules.json`、`tokenizer.json`、模型权重等文件。

## 3. 启动方式

### Web 可视化界面

```bash
python streamlit_app.py
```

也可以使用 Streamlit 标准启动方式：

```bash
streamlit run streamlit_app.py
```

界面能力：
- 侧边栏直接编辑笔记目录，修改后自动生效
- 运行过程中实时显示工具调用（工具名、参数、返回值）
- 工具调用区为浅色可折叠
- 用户与助手最终回复为深色消息气泡

### 单轮问答

```bash
python main.py \
  --notes-dir /path/to/your/notes \
  --embedding-model ./bge-base-zh-v1.5 \
  --query "帮我总结今天的会议纪要" \
  --show-trace
```

### 交互模式

```bash
python main.py --notes-dir /path/to/your/notes --embedding-model ./bge-base-zh-v1.5
```

输入 `exit` 退出。

## 4. 核心能力与实现映射

### 4.1 完整 Agent Loop
- 入口：`personal_data_agent/agent/loop.py`
- 核心流程：
  1. 组装 `messages`（系统提示 + 历史 + 当前输入）
  2. 调用模型（`tools=...`, `tool_choice="auto"`）
  3. 解析 `tool_calls`
  4. 校验参数并执行工具
  5. 把工具结果回填到消息列表
  6. 循环直到模型返回最终文本答案

### 4.2 多步工具调用
- `max_steps` 控制最大推理轮数
- `max_tool_calls` 控制单次请求内总工具调用上限
- 单轮模型可发多个 `tool_calls`，代码逐个执行并回填

### 4.3 错误处理与重试
- 重试模块：`personal_data_agent/runtime/retry.py`
- 参数校验：`personal_data_agent/runtime/schema.py`
- 错误分类：`personal_data_agent/runtime/errors.py`
- 处理策略：
  - 模型请求失败：自动重试（指数退避）
  - IO 临时失败：自动重试
  - 参数错误/越权访问：立即返回工具错误，交给模型修正计划

## 5. 工具列表

| 工具名称 | 核心功能 | 参数(JSON) | 逻辑说明 |
|---|---|---|---|
| `list_notes` | 列出所有笔记 | `{ "directory": "string" }` | 扫描目录，返回文件名和最后修改时间 |
| `semantic_search` | 语义检索内容 | `{ "query": "string" }` | 使用 `bge-base-zh-v1.5` 对全部笔记切片做向量匹配，返回相关片段和文件名 |
| `read_note` | 读取完整笔记 | `{ "filename": "string" }` | 读取指定文件全文 |
| `write_note` | 新建/覆盖笔记 | `{ "filename": "string", "content": "string" }` | 创建新文件或完整覆盖 |
| `append_note` | 追加写入内容 | `{ "filename": "string", "content": "string" }` | 追加到笔记末尾 |
| `delete_note` | 删除笔记文件 | `{ "filename": "string" }` | 删除指定笔记文件（仅限 `.txt/.md/.markdown`） |
| `extract_highlights` | 提取关键部分 | `{ "filename": "string", "focus": "string" }` | 内部调用 LLM，对指定 focus 输出结构化重点 |

## 6. 安全约束（重点）

- 启动时必须传 `--notes-dir`。
- Agent 所有文件访问都会做路径校验：
  - 只能访问 `--notes-dir` 及其子目录
  - 访问越界会抛 `SecurityError`
- 只允许处理后缀：`.txt` / `.md` / `.markdown`

## 7. 文件型向量库设计

向量索引保存在：

```text
<notes_dir>/.personal_data_agent/index/
  ├── chunks.jsonl
  ├── embeddings.npy
  └── state.json
```

- `chunks.jsonl`：切片文本与来源文件
- `embeddings.npy`：切片向量矩阵
- `state.json`：文件指纹（mtime/size），用于判断是否重建

## 8. 常见问题

1. `embedding 模型加载失败`
- 检查 `--embedding-model` 是否指向本地存在的 `bge-base-zh-v1.5` 目录。
- 如果目录不存在，可以重新执行：

```bash
hf download BAAI/bge-base-zh-v1.5 --local-dir ./bge-base-zh-v1.5
```

2. `工具返回权限错误`
- 检查访问的路径是否在 `--notes-dir` 目录内。

3. `检索为空`
- 确保目录下存在 `.txt/.md/.markdown` 文件。
