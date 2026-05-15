# 校园智能问答系统 — 逐步骤学习手册

> 手机优化版 · 可离线阅读 · 每个 commit 一节课

```
项目：Campus Smart Q&A Assistant
技术栈：FastAPI + LangChain + ChromaDB + DeepSeek + MCP SDK
```

---

## 课前说明

本手册按 **git commit 顺序** 编排，每节对应一次关键提交。每节包含：
- **做了什么** — 这一版改了啥
- **关键代码** — 核心实现片段（手机上也能看清）
- **面试要点** — 面试官可能会问什么

---

## Step 1：文本分割（chunking）

**Commit:** `e846859` · 文本分割

### 做了什么

把校园资料按固定大小切块，让每个块成为一个独立的知识点，方便后续向量化检索。

### 关键代码

```python
# document_processing.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,       # 每块最多 700 字
    chunk_overlap=150,    # 相邻块重叠 150 字
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_documents(docs)
```

- `chunk_size=700`：一个块装 1-2 个完整知识点，不截断关键信息
- `chunk_overlap=150`：块之间有重叠，防止关键句被拦腰切断
- `separators` 优先级：段落 > 句子 > 词，尽量在语义边界切开

### 面试要点

**Q: chunk_size 为什么取 700？**
A: 校园资料每条 300-500 字，700 保证每个块包含完整的知识点。试过 100（太碎）、500（刚好）、700（最佳）、1000（混入无关内容）。

**Q: chunk_overlap 的作用？**
A: 防止关键信息落在两个块的交界处被丢掉。150 字的重叠 ≈ 2 句，够用了。

---

## Step 2：向量化存入 ChromaDB

**Commit:** `05269bb` · 向量化存入 ChromaDB

### 做了什么

把文本块转为向量（embedding），存到 ChromaDB 向量数据库中。这样搜索时通过"找最相似的向量"来召回相关内容。

### 关键代码

```python
# vector_store.py
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,  # 文本 → 向量
    persist_directory="chroma_db", # 存到磁盘，重启不丢
    collection_name="campus_rag"
)
```

**核心原理：**
```
"图书馆几点开门"  →  [0.12, 0.87, -0.34, ...]  (向量)
"食堂吃什么"      →  [-0.56, 0.23, 0.78, ...]  (另一个向量)

找答案 = 找和问题向量最接近的文本块
```

### 面试要点

**Q: 向量检索的原理？**
A: 文本 → embedding 向量 → 余弦相似度排序 → 返回 top-k。本质是"语义近似度搜索"。

**Q: ChromaDB 和 Pinecone 的区别？**
A: ChromaDB 是嵌入式、零配置、单机存储，适合小项目快速验证。Pinecone/Qdrant 是分布式向量数据库，适合百万级以上数据。

---

## Step 3：带 RAG 的问答接口

**Commit:** `931e7bc` · 带 RAG 的问答接口

### 做了什么

把检索到的文本块作为"参考资料"，连同用户问题一起发给 LLM，让 LLM 基于资料回答。这就是 RAG 的核心：**检索 + 生成**。

### 关键代码

```python
# 检索
hits = vectorstore.similarity_search_with_score(
    question, k=3,
    # 过滤低质量结果
    max_margin_from_best=0.22,  # 和最佳结果差太多就丢掉
    max_distance=0.78           # 差的太远也丢掉
)

# 组装 Prompt 发给 LLM
prompt = f"""资料：{context_text}

问题：{question}

请基于资料回答，如果资料不够就说"资料中未提及"。"""
```

### 面试要点

**Q: RAG 相比纯 LLM 的好处？**
A: 解决三个问题：① 知识截止日期（LLM 不知道最新信息）② 幻觉（凭空编造）③ 可追溯（可以查看引用的原文）。

**Q: 为什么 LLM 还要看资料，它自己不知道吗？**
A: 重点是**可控性** — 让 LLM 基于指定资料回答，而不是依赖它的内部知识。资料怎么更新，答案就怎么变。

---

## Step 4：DemoEmbeddings 降级策略

**Commit:** `9a50bb2` · 添加演示嵌入支持

### 做了什么

没有 API Key 时也能运行项目。用了一个基于哈希的本地嵌入（DemoEmbeddings），虽然语义不准确，但保证项目可启动。

### 关键代码

```python
# vector_store.py
class DemoEmbeddings(Embeddings):
    """基于哈希的确定性嵌入，不需要 API Key"""
    def embed_query(self, text: str) -> list[float]:
        hash_obj = hashlib.md5(text.encode())
        # 把 hash 值映射到固定维度的向量
        return [hash_val / 2**64 for ...]

# 自动降级逻辑
def create_project_embeddings(allow_demo_fallback=True):
    try:
        return DashScopeEmbeddings(...), "dashscope"
    except:
        return DemoEmbeddings(dimension=128), "demo"
```

### 面试要点

**Q: 为什么设计这个降级策略？**
A: 面试官演示时可能没有 API Key。降级让项目在任何环境都能运行，这是经验丰富的工程师才会考虑的事。

---

## Step 5：集成 MCP 工具调用

**Commit:** `6c19467` · 集成 MCP 工具调用

### 做了什么

用 MCP（Model Context Protocol）把校园功能做成"工具"：查课表、查图书馆、查教室……系统收到问题后先判断要不要调工具。

### 关键代码

```python
# mcp_server.py — 工具定义
@mcp.tool(description="获取学生的课表")
def get_course_schedule(student_id: str) -> dict:
    return _get_course_schedule_data(student_id)

# mcp_client.py — 调用工具
class MCPClient:
    async def call_tool(self, tool_name, **kwargs):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                result = await session.call_tool(tool_name, arguments=kwargs)
                return result.content
```

**MCP 的工作方式：**
```
主进程 ←→ stdio 管道 ←→ MCP Server 子进程
                              ├── get_course_schedule
                              ├── get_library_status
                              └── query_room_availability
```

### 面试要点

**Q: MCP 解决了什么痛点？**
A: 标准化工具接口。以前每个工具要自己写 HTTP 接口、自己解析参数，MCP 统一了"工具定义→参数声明→结果返回"的协议。

**Q: MCP 和 RPC 框架的区别？**
A: MCP 面向 AI 场景设计，工具声明包含语义描述（LLM 可以理解），支持动态发现。传统 RPC 是程序调程序，MCP 是 AI 调程序。

---

## Step 6：异步架构重构

**Commit:** `e6dd60c` · 异步架构重构

### 做了什么

把所有同步代码改成异步：路由 `def` → `async def`、LLM 调用 `.invoke()` → `.ainvoke()`、MCP 客户端去掉 anyio.run() 封装。

### 关键代码

```python
# 改前（同步）
@app.post("/ask")
def ask(request):                   # 同步函数
    response = llm.invoke(messages) # 阻塞等待
    return response

# 改后（异步）
@app.post("/ask")
async def ask(request):              # 异步函数
    response = await llm.ainvoke(messages)  # 不阻塞
    return response

# 同步操作也要异步化
hits = await asyncio.to_thread(
    hybrid.retrieve, question  # 把函数"扔到线程池"跑
)
```

### 面试要点

**Q: 异步的好处？**
A: FastAPI 一个进程同时处理多个请求。A 请求在等 LLM 回复时，B 请求的数据库查询可以插进来执行，不浪费 CPU 等待。

**Q: asyncio.to_thread() 为什么需要？**
A: 向量检索是 CPU 密集型操作，不能直接 `await`。`to_thread()` 把它放到线程池，不阻塞主事件循环。

---

## Step 7：多轮对话记忆

**Commit:** `f8d88b1` · 添加多轮对话记忆

### 做了什么

系统记住前面的对话，追问时能理解上下文。比如先问"我的课表"，再问"教室在哪"，系统知道"教室"指课表的教室。

### 关键代码

```python
class ConversationMemory:
    def __init__(self, max_rounds=4):
        self.max_rounds = max_rounds

    def add(self, student_id, role, content):
        save_conversation(student_id, role, content)

    def get_history(self, student_id):
        records = get_conversation_history(
            student_id, limit=self.max_rounds * 2
        )
        return [{"role": r["role"], "content": r["content"]}
                for r in records]

# 使用时注入到 LLM 消息列表
history = memory.get_history(student_id)
messages = [SystemMessage(content=SYSTEM_PROMPT)]
for h in history:
    messages.append(HumanMessage(content=h["content"]))
messages.append(HumanMessage(content=current_question))
```

### 面试要点

**Q: 多轮对话怎么实现的？**
A: 每次请求把历史消息拼在 Prompt 里发给 LLM。不是真的"记忆"，是 LLM 的上下文窗口看到了前面的内容。

**Q: 为什么只保留 4 轮？**
A: ① 节省 token（省钱、省延迟）② 太早的对话对当前回答帮助不大 ③ LLM 的上下文窗口有限。

---

## Step 8：流式输出（SSE）

**Commit:** `9e96f23` · 新增流式问答端点

### 做了什么

LLM 生成一个字就发给前端一个字，用户体验就像 ChatGPT 打字一样。协议用的是 SSE（Server-Sent Events）。

### 关键代码

```python
# 后端
async def _stream_answer(question, student_id):
    # 先发元信息
    yield f"data: {json.dumps({'event': 'meta', 'mode': mode})}\n\n"

    # 逐 token 发送
    async for chunk in llm.astream(messages):
        yield f"data: {json.dumps({'event': 'token', 'token': chunk.content})}\n\n"

    # 结束信号
    yield f"data: {json.dumps({'event': 'done'})}\n\n"

@app.post("/ask/stream")
async def ask_stream(request):
    return StreamingResponse(
        _stream_answer(question, student_id),
        media_type="text/event-stream"
    )
```

```javascript
// 前端
const reader = res.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // 解析 SSE，把 token 逐个加到显示区
    textEl.textContent += payload.token;
}
```

### 面试要点

**Q: SSE vs WebSocket 的区别？**
A: SSE 是**单向**（服务端→客户端），基于 HTTP，浏览器原生支持，自动重连。WebSocket 是**双向**，需要额外库。SSE 恰好满足"LLM 输出 → 前端展示"这个单向需求。

**Q: 流式输出实际提升了什么？**
A: 用户等待时间**感知**从"等 5 秒看到完整回答"变成"0.5 秒看到第一个字"。研究表明这能显著提升用户体验满意度。

---

## Step 9：Prompt 配置化

**Commit:** `2d18230` · Prompt 配置化

### 做了什么

把散落在代码各处的 Prompt 字符串集中到一个文件管理，方便修改和维护。

### 关键代码

```python
# config/prompts.py
SYSTEM_PROMPT = "你是校园问答助手，..."
MCP_USER_TEMPLATE = "根据以下实时数据回答：\n{tool_context}\n\n问题：{question}"
RAG_USER_TEMPLATE = "资料：\n{context}\n\n问题：{question}\n\n..."
AUTO_GENERATED_TEMPLATE = "没有可用资料。\n\n问题：{question}\n\n..."

# app/main.py — 使用
from config.prompts import RAG_USER_TEMPLATE
prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
```

### 面试要点

**Q: 为什么要抽取 Prompt？**
A: ① 改动 Prompt 不需要动业务代码 ② 可以 A/B 测试不同 Prompt 效果 ③ 面试中展示"工程思维"——关注可维护性。

---

## Step 10：Embedding 意图分类器

**Commit:** `e55993b` · 工具路由升级

### 做了什么

原来用关键词匹配判断"要不要调工具"（比如问题含"课表"就调课表工具），改用 Embedding 语义匹配，准确率高得多。

### 关键代码

```python
class IntentClassifier:
    def __init__(self, embeddings, threshold=0.55):
        self.threshold = threshold
        # 每个工具定义 5-7 条示例问题
        self.examples = {
            "课表": ["我的课表是什么", "今天有什么课", ...],
            "食堂": ["今天食堂有什么菜", "食堂菜单", ...],
            ...
        }

    def classify(self, question):
        query_vec = self.embeddings.embed_query(question)
        # 找和问题最相似的工具意图
        for intent, vectors in self.tool_examples.items():
            score = cosine_similarity(query_vec, vectors)
            if score > best_score and score >= threshold:
                best_intent = intent
        return best_intent  # None 表示不调工具

    # 还过滤了非工具类问题（你好、谢谢等）
```

### 面试要点

**Q: 为什么舍弃关键词匹配？**
A: "查一下这周课表"含"课表"→ 命中。但"什么时候上算法课"不含"课表"却和课表相关。关键词匹配会漏掉。Embedding 可以理解语义。

**Q: threshold 参数怎么调的？**
A: 高了 → 工具调用太少（漏召回），低了 → 容易误触。0.55 是在测试集上调出来的最佳值。

---

## Step 11：SQLite 持久化

**Commit:** `0ed18d1` · SQLite 持久化

### 做了什么

对话记忆和交互日志从内存迁移到 SQLite 数据库。重启服务后对话不会丢。

### 关键代码

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, Session

engine = create_engine("sqlite:///campus_qa.db")
Base = declarative_base()

class ConversationRecord(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    student_id = Column(String)
    role = Column(String)    # user / assistant
    content = Column(Text)
    created_at = Column(DateTime, default=func.now())

class InteractionLog(Base):
    __tablename__ = "interaction_logs"
    # 记录：问题、答案、模式、耗时等
```

### 面试要点

**Q: 为什么用 SQLite 而不是 MySQL/PostgreSQL？**
A: 项目规模小，SQLite 零配置、文件存储、不需独立进程。迁移到 MySQL 只需改一行连接 URL。

**Q: 两张表的职责分离？**
A: `conversations` 存对话上下文（给 LLM 用的），`interaction_logs` 存审计日志（给开发者分析用的）。不同用途分不同表，这是数据库设计的基本功。

---

## Step 12：Hybrid Search（混合检索）

**Commit:** `dc26bfa` · RAG 升级为混合检索

### 做了什么

在向量检索（语义匹配）之外，加了 BM25 关键词检索。两者结果用 RRF（Reciprocal Rank Fusion）算法融合排序。

### 关键代码

```python
class HybridRetriever:
    def retrieve(self, question, k=3):
        # 1. 向量检索 — 按语义
        vector_results = self.vectorstore.similarity_search(question, k=k*2)

        # 2. BM25 检索 — 按关键词
        tokenized = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized)
        bm25_rankings = sorted(range(len(bm25_scores)),
                               key=lambda i: bm25_scores[i], reverse=True)

        # 3. RRF 融合
        # 给每个文档在两个检索结果中的排名算分
        # score = 1/(rank_vector + 60) + 1/(rank_bm25 + 60)
        # 排名越靠前（rank 越小），得分越高
        combined = self._rrf([vector_rankings, bm25_rankings])
        return combined[:k]

    @staticmethod
    def _tokenize(text):
        # 中文按字切，英文按词切
        # "图书馆几点开门" → ["图", "书", "馆", "几", "点", "开", "门"]
        return [c if ord(c) > 127 else c.lower()
                for c in re.sub(r'[^\w]', '', text)]
```

### 面试要点

**Q: 为什么向量检索不够，还要加 BM25？**
A: 向量检索擅长"语义相似"（"怎么去教室"和"教学楼怎么走"语义相近），但关键词精确匹配有时更重要（学号"S001"、书号"B123"的精确匹配）。BM25 擅长关键词精确匹配。两者互补。

**Q: RRF 是什么原理？**
A: 不管这个检索方法分数高低只要排名高就加分。公式：`score(r) = 1/(r + k)`，其中 k 通常取 60。好处是不需要归一化不同检索方法的分数。

---

## Step 13：真实天气 API

**Commit:** `1b24f96` · MCP 工具对接真实天气 API

### 做了什么

把天气工具从 Mock 数据改为调用 Open-Meteo 免费天气 API，不！需！要！API Key！两步调用：先查城市坐标，再拿天气预报。

### 关键代码

```python
async def _get_weather_data(city):
    async with httpx.AsyncClient() as client:
        # 1. 地理编码：城市名 → 经纬度
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}
        )
        lat, lng = geo["results"][0]["latitude"], ...

        # 2. 天气数据
        w = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lng,
                    "current": "temperature_2m,weather_code"}
        )
        return {"temperature": w["current"]["temperature_2m"],
                "weather": weather_codes[w["current"]["weather_code"]]}
```

### 面试要点

**Q: 选 Open-Meteo 的原因？**
A: 免费、无需 API Key、返回数据标准。面试项目最怕"有 API Key 才能演示"，选 Open-Meteo 降低了演示门槛。

**Q: 真实 API 调用怎么处理异常？**
A: try-except 包裹，失败返回 `{"error": "天气查询失败: ..."}`，LLM 看到 error 会给用户友好提示，不会崩溃。

---

## Step 14：API Key 认证

**Commit:** `28bf4d6` · API Key 认证

### 做了什么

给所有 API 端点加了 X-API-Key 头验证（/health 首页除外），前端加了密码输入框，提！交！时！自动注入。

### 关键代码

```python
# config/auth.py
_API_KEY = os.getenv("API_KEY", "campus-qa-dev-key")

async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != _API_KEY:
        raise HTTPException(status_code=401,
                            detail="Invalid or missing API Key")

# app/main.py — 使用
@app.post("/ask")
async def ask(request, _auth=Depends(verify_api_key)):
    ...
```

```javascript
// 前端 — 统一注入
function apiHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-API-Key': document.getElementById('api-key').value
    };
}
```

### 面试要点

**Q: 为什么不搞 OAuth 2.0 或 JWT？**
A: 项目是 API 对前端（没有用户注册），API Key 最简单够用。如果有多用户、权限控制，会升级到 JWT。

**Q: 为什么用依赖注入（Depends）而不是装饰器？**
A: FastAPI 的 Depends 可以精细控制每个端点的认证策略，装饰器是"全有或全无"。Depends 还能方便地组合多个依赖（认证 + 限流 + 日志）。

---

## 总结：架构全景

```
用户提问
    │
    ▼
┌─────────────────────────────────────┐
│ Step 10: 意图分类器                 │
│ IntentClassifier.classify(question) │
└────────────┬────────────────────────┘
             │
   ┌────有匹配─────┐   无匹配
   ▼               │
┌──────────┐       ▼
│ Step 5+13│  ┌────────────────────────┐
│ MCP 工具 │  │ Step 12: Hybrid Search │
│ (实时数据)│  │ BM25 + 向量 + RRF     │
└────┬─────┘  └──────────┬─────────────┘
     │                   │
     │          ┌──有结果──┴──无结果──┐
     │          ▼                    ▼
     │   ┌──────────────┐  ┌────────────────┐
     │   │ RAG Prompt   │  │ Step 9:        │
     │   │ (上下文+问题) │  │ 自动生成 Prompt │
     │   └──────┬───────┘  └───────┬────────┘
     │          │                  │
     └──────────┴──────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │ Step 6+8:     │
               │ LLM 异步+流式  │
               │ (ainvoke/SSE) │
               └────────────────┘
```

**贯穿全局：**
- Step 4: DemoEmbeddings 降级
- Step 7+11: 多轮对话 + SQLite 持久化
- Step 14: API Key 认证
- Step 1-2-3: RAG 基础管道

---

## 常见面试追问速查

| 追问 | 一句话回答 |
|------|-----------|
| 怎么保证检索质量？ | max_margin_from_best + max_distance 双过滤，再加 Hybrid Search 融合排序 |
| 为什么不直接问 LLM？ | RAG 可追溯、可更新、可控，不依赖 LLM 的内部知识 |
| 并发量大怎么办？ | Gunicorn 多 worker + Redis 缓存 + 异步架构支撑高并发 |
| 向量库存了什么？ | 13 个校园文档切成的 700 字文本块，每个块有原始文本和 embedding 向量 |
| 最难解决的问题？ | 异步改造时 MCP 的嵌套事件循环冲突，需要用 stdio 进程隔离 |
| 这个项目你贡献了多少？ | 从零到一的独立开发，架构设计 + 代码实现 + 测试 + 部署全流程 |
