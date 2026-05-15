# Campus Smart Q&A Assistant

**校园智能问答系统** — 基于 FastAPI + LangChain + ChromaDB + DeepSeek + MCP 的多模态 RAG 问答系统。

## 架构

```text
用户输入 → FastAPI → [ MCP 工具调用 ] → [ RAG 向量检索 ] → [ LLM 自动生成 ] → 回答
```

三层回退路由：

1. **MCP 实时数据** — 课表、图书馆、食堂、校车等校园工具优先
2. **RAG 知识库** — ChromaDB 向量检索校园资料
3. **LLM 自动生成** — 兜底策略，带【自动生成】标记

## 技术栈

| 层 | 技术 | 用途 |
|---|------|------|
| 框架 | **FastAPI** (异步) | HTTP 路由、SSE 流式输出 |
| RAG | **LangChain** + **ChromaDB** | 文档切分、向量化、相似检索 |
| 嵌入 | **DashScope Embeddings** / DemoEmbeddings | 真实/演示嵌入降级 |
| LLM | **DeepSeek Chat** (OpenAI 兼容接口) | 问答生成 |
| 工具 | **MCP SDK** (官方 Python SDK) | 校园工具注册与调用 |
| 前端 | 原生 HTML/CSS/JS | 可视化交互 |
| 部署 | **Docker** + docker-compose | 容器化 |

## 特性

- **三层自动路由**：MCP 工具 → RAG 检索 → LLM 自动生成，逐级回退
- **6 个校园 MCP 工具**：课表查询、图书馆状态、教室可用性、食堂菜单、校车时刻、宿舍报修
- **13 个知识库文档**：宿舍、食堂、奖学金、考试、校园卡、体育、就业等
- **多轮对话记忆**：按学生 ID 保留最近 4 轮对话
- **SSE 流式输出**：`/ask/stream` 端点逐 token 返回
- **对话管理**：`/reset` 端点清除历史
- **34 个单元测试**：API 层、MCP 工具函数、RAG 全链路
- **Docker 一键部署**
- **演示嵌入降级**：无 API Key 时本地 DemoEmbeddings 自动降级

## 快速开始

### 前置条件

- Python 3.10+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）
- 可选：DashScope API Key（百炼嵌入，[申请地址](https://help.aliyun.com/zh/model-studio/)）

### 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd campus-qa

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 运行

```bash
uvicorn main:app --reload
```

访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Docker

```bash
docker compose up --build
```

## API 文档

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | 自动路由问答 |
| `/ask_with_tools` | POST | 问答 + 工具详情 |
| `/ask/stream` | POST | SSE 流式问答 |
| `/reset` | POST | 清除对话历史 |

### 请求示例

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "图书馆几点开门？", "student_id": "S001"}'
```

### 响应格式

```json
{
  "answer": "图书馆总馆开放时间为周一至周日 7:30-22:30",
  "mode": "rag",
  "auto_generated": false,
  "tools_used": [],
  "tool_results": {}
}
```

### 流式响应

```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍一下学校食堂"}'
```

返回 SSE 格式：

```
data: {"event": "meta", "mode": "mcp", "auto_generated": false, ...}

data: {"event": "token", "token": "学校"}

data: {"event": "token", "token": "有"}

data: {"event": "token", "token": "三个食堂..."}

data: {"event": "done"}
```

## 运行测试

```bash
pytest -v
```

## 项目结构

```text
.
├── app/
│   └── main.py              # FastAPI 应用（异步路由）
├── frontend/
│   ├── index.html           # 前端页面
│   ├── app.js               # 前端交互
│   └── style.css            # 样式
├── data/                    # 13 个校园知识库文档
├── tests/                   # 34 个测试用例
│   ├── test_api.py
│   ├── test_mcp.py
│   └── test_retrieval.py
├── mcp_server.py            # MCP 工具服务器
├── mcp_client.py            # MCP 异步客户端
├── vector_store.py          # ChromaDB 向量检索
├── document_processing.py   # 文档切分（chunk_size=700）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 设计决策

### 为什么用 MCP 而非 REST 调用工具？

MCP（Model Context Protocol）是 Anthropic 提出的标准化工具协议，与 LangChain 的工具调用相比：
- **协议标准化**：工具定义、调用、结果返回统一
- **进程隔离**：工具服务器独立进程，不影响主服务稳定性
- **多语言**：MCP Server 可用 Python/TypeScript/Java 等实现

### 为什么选 ChromaDB？

- **轻量零配置**：嵌入式向量库，无需单独部署
- **文件持久化**：DB 存在磁盘文件，重启不丢失
- **适合 demo 场景**：数据量 < 1M 条时性能足够

### RAG chunk_size 为什么取 700？

- 校园资料平均段落长度约 300-500 字符
- chunk_size=700 保证每个文本块包含 1-2 个完整知识点
- chunk_overlap=150 确保跨块信息不丢失

## FAQ

**Q: 如何添加上下文？** → 在 `data/` 下添加 `.txt` 或 `.pdf` 文件，重启服务自动重建索引。

**Q: 能否接入其他 LLM？** → 修改 `app/main.py` 中 `_get_llm()` 的 `base_url` 和 `model` 参数即可。

**Q: 如何新增 MCP 工具？** → 在 `mcp_server.py` 添加工具函数 + `@mcp.tool()` 装饰器，然后在 `app/main.py` 的 `_call_tools_for_question` 中添加路由逻辑。

## License

MIT
