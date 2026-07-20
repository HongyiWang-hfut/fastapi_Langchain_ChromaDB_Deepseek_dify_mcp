# HFUT Q&A Assistant

> **合肥工业大学智能问答系统** · 基于 FastAPI + LangChain + ChromaDB + DeepSeek + MCP 的多模态 RAG 问答系统，面向 HFUT 校园场景。

<p align="center">
  <a href="https://github.com/HongyiWang-hfut/hfut-qa-assistant"><img src="https://img.shields.io/badge/GitHub-hfut--qa--assistant-blue?logo=github&style=flat-square" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&style=flat-square" alt="FastAPI">
  <img src="https://img.shields.io/badge/VectorStore-ChromaDB-FF6B6B?style=flat-square" alt="ChromaDB">
  <img src="https://img.shields.io/badge/LLM-DeepSeek-4D6BFE?style=flat-square" alt="DeepSeek">
  <img src="https://img.shields.io/badge/License-MIT-2EA043?style=flat-square" alt="License">
</p>

---

## 📑 目录

- [项目简介](#项目简介)
- [🔑 申请 API Key（必看）](#-申请-api-key必看)
- [架构与原理](#架构与原理)
- [技术栈](#技术栈)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [构建知识库索引](#构建知识库索引)
- [Docker 部署](#docker-部署)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [设计决策](#设计决策)
- [常见问题 FAQ](#常见问题-faq)
- [License](#license)

---

## 项目简介

HFUT Q&A Assistant 是一个**校园智能问答系统**：用户输入问题后，系统会按「三层回退路由」自动选择最优答案来源——优先调用实时校园工具（MCP），其次检索校园知识库（RAG），最后由大模型兜底生成。

```text
用户输入 → FastAPI → [ MCP 工具调用 ] → [ RAG 向量检索 ] → [ LLM 自动生成 ] → 回答
```

**三层回退路由：**

1. **MCP 实时数据** — 课表、图书馆、教室、食堂、校车、天气、报修等校园工具优先
2. **RAG 知识库** — ChromaDB 向量检索校园资料（混合检索：向量语义 + BM25 关键词，RRF 融合）
3. **LLM 自动生成** — 兜底策略，答案带【自动生成】标记

---

## 🔑 申请 API Key（必看）

本项目需要 **1 个外部 Key** 才能跑通问答；另 1 个可选（用于更精准的语义嵌入）。`API_KEY` 为本地接口鉴权，**无需申请**，项目已内置默认值。

| 变量名 | 用途 | 必填 | 申请地址 |
|--------|------|:----:|----------|
| `DEEPSEEK_API_KEY` | DeepSeek 大模型（问答生成核心） | ✅ **必填** | 🔗 [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |
| `DASHSCOPE_API_KEY` | 阿里云百炼嵌入（知识库向量化，text-embedding-v2） | ⬜ 可选 | 🔗 [dashscope.console.aliyun.com/apiKey](https://dashscope.console.aliyun.com/apiKey) |
| `API_KEY` | 本地 HTTP 接口鉴权（`X-API-Key` 请求头） | ⬜ 自定 | 无需申请，默认 `campus-qa-dev-key` |

> 💡 **没有 DashScope Key 也能跑**：未配置时会自动降级为本地 `DemoEmbeddings`，检索功能正常，仅语义精度略降。所以**第一步只需申请 DeepSeek Key** 即可体验完整问答。

**申请步骤（以 DeepSeek 为例）：**

1. 打开 🔗 [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
2. 注册 / 登录 → 点击「创建 API Key」→ 复制生成的 `sk-...` 字符串
3. 填入项目 `.env` 的 `DEEPSEEK_API_KEY=` 一行即可

---

## 架构与原理

系统采用**异步 FastAPI** 作为入口，结合 **MCP 协议**调用校园工具、**LangChain + ChromaDB** 完成 RAG 检索，最终由 **DeepSeek Chat** 生成自然语言回答。

- **混合检索（Hybrid Retrieval）**：向量语义检索 + BM25 关键词检索，经 RRF（Reciprocal Rank Fusion）融合排序，兼顾「意思对」和「词对得上」。
- **运行时入库**：知识库支持启动时从 `data/` 构建、运行时用户上传、以及脚本爬取官网资讯，三种来源统一聚合展示。

---

## 技术栈

| 层 | 技术 | 用途 |
|---|------|------|
| 框架 | **FastAPI**（异步） | HTTP 路由、SSE 流式输出、API Key 鉴权 |
| RAG | **LangChain** + **ChromaDB** | 文档切分、向量化、混合相似检索 |
| 嵌入 | **DashScope Embeddings**（text-embedding-v2）/ DemoEmbeddings | 真实 / 演示嵌入降级 |
| LLM | **DeepSeek Chat**（OpenAI 兼容接口） | 问答生成 |
| 工具 | **MCP SDK**（官方 Python SDK） | 校园工具注册与进程隔离调用 |
| 前端 | 原生 HTML/CSS/JS（SPA + 玻璃拟态） | 可视化交互、流式渲染 |
| 部署 | **Docker** + docker-compose | 容器化一键部署 |

---

## 核心特性

- **三层自动路由**：MCP 工具 → RAG 检索 → LLM 自动生成，逐级回退
- **7 个校园 MCP 工具**：课表查询、图书馆借阅、教室可用性、食堂菜单、校车时刻、实时天气、宿舍报修
- **多源知识库**：
  - 13 个内置校园文档（`data/`：宿舍、食堂、奖学金、考试、校园卡、体育、就业、社团、图书馆、校医院、校园网、交通、自习室等）
  - 合肥工业大学官网资讯爬取（`scripts/scrape_hfut_news.py`，自动入库 `hfut_official` 来源）
  - 用户运行时上传文档（`.txt`/`.md`/`.pdf`，通过 `/knowledge/upload` 入库）
- **知识库管理 API**：检索、列举、上传、删除（三种来源统一聚合展示）
- **多轮对话记忆**：按学生 ID 保留最近 4 轮对话（`/history/{student_id}` 可查）
- **SSE 流式输出**：`/ask/stream` 端点逐 token 返回，前端打字机效果
- **对话管理**：`/reset` 端点清除历史
- **50+ 测试用例**：覆盖 API 层、MCP 工具函数、意图分类、RAG 全链路
- **前端视觉系统**：SPA 多模块架构、玻璃拟态、固定背景图、校园文化画廊、暗色模式、Markdown 渲染、加载/错误态
- **演示嵌入降级**：无 DashScope Key 时本地 DemoEmbeddings 自动降级（检索仍可用，仅语义精度下降）
- **Docker 一键部署**

---

## 快速开始

### 前置条件

- Python 3.10+
- **DeepSeek API Key**（[点此申请](https://platform.deepseek.com/api_keys)）
- 可选：DashScope API Key（[点此申请](https://dashscope.console.aliyun.com/apiKey)）；不填则走演示嵌入降级

### 安装与配置

```bash
# 1. 克隆项目
git clone https://github.com/HongyiWang-hfut/hfut-qa-assistant.git
cd hfut-qa-assistant

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（把 DeepSeek Key 填进去）
cp .env.example .env
#   然后编辑 .env，将 DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here 换成你的真实 Key
```

### 运行

```bash
# 方式一（推荐）：根目录兼容入口
uvicorn main:app --reload

# 方式二：直接指向应用模块（等价）
uvicorn app.main:app --reload
```

启动后访问 👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

> 注：仓库根目录的 `main.py` 仅一行 `from app.main import app`，是给 `uvicorn main:app` 用的兼容入口，两种方式等价。

---

## 构建知识库索引

服务启动时若 `chroma_db/` 不存在或维度不一致，会自动从 `data/` 重建索引。也可手动预构建 / 爬取官网资讯：

```bash
# 可选：爬取合肥工业大学官网资讯并入库（source=hfut_official）
python scripts/scrape_hfut_news.py
```

---

## Docker 部署

```bash
docker compose up --build
```

> 记得先在 `.env` 中填好 `DEEPSEEK_API_KEY` 再构建，容器启动后会自动读取。

---

## API 文档

所有**写 / 敏感接口**均通过 `X-API-Key` 请求头鉴权（默认值见 `.env` 的 `API_KEY`，开发默认 `campus-qa-dev-key`；前端已内置该默认 Key）。

| 端点 | 方法 | 鉴权 | 说明 |
|------|------|:----:|------|
| `/` | GET | 否 | 前端 SPA 页面（强防缓存） |
| `/health` | GET | 否 | 健康检查 |
| `/ask` | POST | 是 | 自动路由问答（MCP→RAG→自动生成） |
| `/ask_with_tools` | POST | 是 | 问答 + 工具原始结果 |
| `/ask/stream` | POST | 是 | SSE 流式问答 |
| `/reset` | POST | 是 | 清除指定学生 ID 的对话历史 |
| `/history/{student_id}` | GET | 是 | 获取该学生的交互历史 |
| `/knowledge/search` | GET | 是 | RAG 混合检索演示（返回命中文档+相似度） |
| `/knowledge/files` | GET | 是 | 列出知识库所有已入库文件/资讯条目 |
| `/knowledge/files` | DELETE | 是 | 按标识删除条目（含其全部向量 chunk） |
| `/knowledge/upload` | POST | 是 | 上传 `.txt/.md/.pdf` 文档并切分入库 |

### 请求示例

```bash
# 问答接口（带 X-API-Key，开发默认 campus-qa-dev-key）
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: campus-qa-dev-key" \
  -d '{"question": "图书馆几点开门？", "student_id": "S001"}'
```

```bash
# 知识库检索
curl "http://localhost:8000/knowledge/search?q=%E5%9B%BE%E4%B9%A6%E9%A6%86&k=3" \
  -H "X-API-Key: campus-qa-dev-key"
```

```bash
# 上传知识文档
curl -X POST http://localhost:8000/knowledge/upload \
  -H "X-API-Key: campus-qa-dev-key" \
  -F "file=@./my-doc.txt"
```

> 提示：中文参数请用 `encodeURIComponent` / URL 编码（如上面 `%E5%9B%BE...`），避免终端编码导致检索串乱码。

### 响应格式（`/ask`）

```json
{
  "answer": "图书馆总馆开放时间为周一至周日 7:30-22:30",
  "mode": "rag",
  "auto_generated": false,
  "tools_used": [],
  "tool_results": {}
}
```

### 流式响应（`/ask/stream`）

```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: campus-qa-dev-key" \
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

---

## 项目结构

```text
.
├── main.py                 # 兼容入口（from app.main import app）
├── app/
│   └── main.py             # FastAPI 应用（异步路由、知识库管理、SSE、鉴权）
├── config/
│   ├── auth.py             # X-API-Key 鉴权依赖
│   ├── intent_classifier.py  # 基于 Embedding 的意图分类（7 类）
│   └── prompts.py          # 系统/用户提示词模板
├── frontend/
│   ├── index.html          # SPA 页面（玻璃拟态 + 背景图 + 知识库上传模块）
│   ├── app.js              # 前端交互（流式、历史、知识库管理）
│   └── style.css           # 样式
├── data/                   # 13 个内置校园知识库文档（含 uploads/ 用户上传目录）
├── scripts/
│   └── scrape_hfut_news.py  # 合肥工业大学官网资讯爬取入库
├── tests/                  # 50+ 测试用例
│   ├── test_api.py
│   ├── test_database.py
│   ├── test_intent.py
│   ├── test_mcp.py
│   └── test_retrieval.py
├── mcp_server.py           # MCP 工具服务器（7 个校园工具）
├── mcp_client.py           # MCP 异步客户端
├── vector_store.py         # ChromaDB 向量检索 + 混合检索器
├── document_processing.py  # 文档切分（chunk_size=700, overlap=150）
├── database.py             # SQLite 对话历史持久化
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

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

### 知识库为什么支持运行时上传 / 爬取？

内置 `data/` 文档覆盖常规问答，但校园资讯（新闻、通知）是动态的。因此提供：

- `scripts/scrape_hfut_news.py` 爬取官网资讯，带 `source=hfut_official` / `url` / `title` / `date` 元数据入库
- `/knowledge/upload` 接受用户上传，标记 `source=user_upload`
- 检索层每次请求重新打开 Chroma 并重建 BM25（绕过启动缓存），保证能看到运行时新增内容

---

## 常见问题 FAQ

**Q: 如何添加上下文？**
→ 在 `data/` 下添加 `.txt` 或 `.pdf` 文件，重启服务自动重建索引；或用 `/knowledge/upload` 运行时上传；或跑 `scripts/scrape_hfut_news.py` 抓取官网资讯。

**Q: 能否接入其他 LLM？**
→ 修改 `app/main.py` 中 `_get_llm()` 的 `base_url` 和 `model` 参数即可（OpenAI 兼容接口）。

**Q: 如何新增 MCP 工具？**
→ 在 `mcp_server.py` 添加工具函数 + `@mcp.tool()` 装饰器，然后在 `app/main.py` 的 `_call_tools_for_intent` 中添加意图→工具路由。

**Q: 上传的文件重启后会丢失吗？**
→ 不会。上传原文保存在 `data/uploads/`，且向量已写入 `chroma_db/`；索引重建时会从 `data/` 整体重新切分入库，上传内容自动保留。

**Q: 中文检索出现乱码 / 不匹配？**
→ 多为终端编码问题（如 Git Bash 将中文按 GBK 传给 curl）。请用浏览器（UTF-8）或 Python `requests`/`urllib` 带 URL 编码调用，服务端本身使用 UTF-8。

**Q: 没有 DashScope Key 能用吗？**
→ 能。未配置时自动降级为本地 `DemoEmbeddings`，检索与问答均正常，仅语义精度略降。

---

## License

[MIT](./LICENSE)
