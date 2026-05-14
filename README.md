# Campus Smart Q&A Assistant

Minimal FastAPI RAG (Retrieval-Augmented Generation) service that retrieves context from Chroma vector database and answers questions using DeepSeek LLM via OpenAI-compatible API.

## 核心架构

- **向量化检索**：基于 Chroma + Alibaba DashScope 嵌入
- **大模型回答**：调用 DeepSeek（通过 OpenAI 兼容接口）
- **内容约束**：系统提示词确保模型只基于检索结果回答，减少幻觉
- **自动初始化**：第一次启动时自动扫描必需配置，从 `data/` 构建向量库

## 环境配置

创建 `.env` 文件，放在项目根目录（与 `main.py` 同级），包含以下内容：

### 必需（用于大模型）
```dotenv
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 可选（真实嵌入）
```dotenv
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```
- 若未配置 `DASHSCOPE_API_KEY`，系统会自动使用**演示嵌入**（本地随机向量，仅用于测试）
- 配置后使用阿里百炼 `text-embedding-v2`（1536 维）

### 可选（DeepSeek）
```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

## 启动

```bash
uvicorn main:app --reload
```

### 启动日志示例

首次启动（向量库不存在）：
```
2026-05-14 14:44:20 [INFO] 向量库初始化 | DASHSCOPE_API_KEY=已配置 | mode=真实嵌入 | embedding_dim=1536
2026-05-14 14:44:20 [INFO] 未发现现有向量库，准备从 data/ 初始化
2026-05-14 14:44:21 [INFO] 开始构建向量库 | chunks=20 | persist_dir=...chroma_db
2026-05-14 14:44:22 [INFO] ✅ 向量库已重建 | docs=20 | embedding_dim=1536 | mode=真实嵌入
2026-05-14 14:44:23 [INFO] Application startup complete
```

再次启动（向量库已存在）：
```
2026-05-14 14:45:00 [INFO] 向量库初始化 | DASHSCOPE_API_KEY=已配置 | mode=真实嵌入 | embedding_dim=1536
2026-05-14 14:45:00 [INFO] ✅ 向量库可用 | docs=20 | embedding_dim=1536 | collection='campus_rag'
2026-05-14 14:45:01 [INFO] Application startup complete
```

无嵌入 Key 时（演示模式）：
```
2026-05-14 14:46:00 [INFO] 向量库初始化 | DASHSCOPE_API_KEY=未配置 | mode=演示嵌入 | embedding_dim=1536
2026-05-14 14:46:00 [INFO] ✅ 向量库可用 | docs=20 | embedding_dim=1536 | collection='campus_rag'
```

## 测试

### 方式 1：curl

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "图书馆借书能借多少册？"}'
```

响应示例：
```json
{
  "answer": "根据校园资料，图书馆借书规则为..."
}
```

### 方式 2：Python 客户端

```bash
python scripts/ask_client.py "图书馆借书能借多少册？"
```

### 方式 3：在线 Swagger API 文档

启动后访问：`http://127.0.0.1:8000/docs`

## 目录结构

```
.
├── main.py                    # FastAPI 主入口，/ask 接口定义
├── vector_store.py            # Chroma 向量库管理、嵌入配置、检索逻辑
├── document_processing.py     # 文档读取与分块
├── .env                       # 环境变量（需自建）
├── chroma_db/                 # 向量库持久化目录（运行时自动创建）
├── data/                      # 校园资料源文件
│   ├── campus_clubs.txt
│   ├── campus_health.pdf
│   ├── campus_library.txt
│   └── campus_wifi.txt
└── scripts/
    └── ask_client.py          # 命令行测试客户端
```

## 向量库工作流

1. **初始化时**：`_get_vectorstore()` 调用 `load_or_rebuild_campus_chroma()`
   - 检查 `DASHSCOPE_API_KEY`，决定使用真实嵌入或演示嵌入
   - 探测当前嵌入维度
   - 加载已有向量库或从 `data/` 构建

2. **加载已有向量库时**：
   - 校验集合文档数是否为 0
   - 校验向量维度是否与当前嵌入配置一致
   - 若维度不一致或集合为空，自动删除旧库、重建

3. **检索时**：`retrieve_top_chunks(question)` 返回最相关的 `k=3` 条文档片段

## 关键功能

- ✅ **自动初始化**：无需手动运行脚本，启动时自动构建向量库
- ✅ **维度校验**：自动检测客户端嵌入维度，不一致时重建库
- ✅ **灵活降级**：无 DASHSCOPE_API_KEY 时自动切换到演示嵌入
- ✅ **容错日志**：详细的初始化日志，便于排查配置问题
- ✅ **减少幻觉**：系统提示词严格约束模型只基于检索结果回答

