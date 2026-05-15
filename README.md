# Campus Smart Q&A Assistant

一个简约时尚的校园智能问答项目：

- **RAG**：从校园资料中检索相关内容
- **MCP**：调用教务 / 图书馆 / 教室等工具
- **自动生成**：当工具和检索都不够时，LLM 给出标注答案
- **前端页面**：直接在浏览器里测试每种能力

## 现在的运行方式

启动后直接打开浏览器即可：

```bash
uvicorn main:app --reload
```

然后访问：

```text
http://127.0.0.1:8000/
```

## 前端可以测试什么

前端页面支持一键测试三种路径：

1. **MCP 工具调用**
   - 例如：`我的课表是什么？`
   - 会自动调用 `get_course_schedule`

2. **RAG 检索回答**
   - 例如：`图书馆总馆开放时间是什么？`
   - 会优先检索 `data/` 里的资料

3. **自动生成回答**
   - 例如：`请用一句话介绍你的能力`
   - 如果工具和检索都不够，LLM 会给出并标注 `【自动生成】`

## 项目结构

```text
.
├── app/
│   ├── __init__.py
│   └── main.py              # 真实的 FastAPI 应用
├── frontend/
│   ├── index.html           # 简约时尚前端
│   ├── app.js               # 前端交互逻辑
│   └── style.css            # 页面样式
├── data/                    # 校园资料
├── mcp_server.py            # 官方 MCP SDK Server
├── mcp_client.py            # 官方 MCP SDK Client 封装
├── vector_store.py          # 向量库与检索
├── document_processing.py   # 文档切分
├── main.py                  # 兼容入口，转发到 app/main.py
└── requirements.txt
```

## 环境配置

在项目根目录创建 `.env`：

```dotenv
DEEPSEEK_API_KEY=sk-xxxxxx
# 可选：真实嵌入
DASHSCOPE_API_KEY=sk-xxxxxx
```

## 主要接口

### `POST /ask`
自动路由：**MCP → RAG → 自动生成**

请求示例：

```json
{ "question": "我的课表是什么？", "student_id": "S001" }
```

### `POST /ask_with_tools`
与 `/ask` 同一套流程，但更适合查看工具结果。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 启动项目

```bash
uvicorn main:app --reload
```

## 建议的验证方式

打开前端页面后，依次点击：

- 课表测试
- 图书馆测试
- 教室测试
- 自动生成测试

页面会展示：

- 当前模式：`mcp` / `rag` / `auto`
- 是否自动生成
- 使用了哪些工具
- 工具返回的摘要

## MCP 相关说明

当前项目使用的是 **官方 MCP SDK**：

- `mcp_server.py`：`FastMCP`
- `mcp_client.py`：`ClientSession` + `stdio_client`

这意味着你不需要再维护一套手写 JSON-RPC 客户端。

## 已删除的内容

- 旧的临时测试脚本已经移除
- 现在功能验证主要通过前端页面完成

## 适合简历的亮点

- 校园知识问答 RAG
- MCP 工具自动调用
- 自动生成兜底与标注
- 浏览器可视化验证
- 简约统一的 UI 风格
