"""校园智能问答助手：FastAPI HTTP 入口，支持 MCP、RAG 和自动生成回退（异步）。"""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from config.auth import verify_api_key
from config.intent_classifier import IntentClassifier
from database import (
    clear_conversation_history,
    get_conversation_history,
    log_interaction,
    save_conversation,
)
from config.prompts import (
    AUTO_GENERATED_TEMPLATE,
    MCP_USER_TEMPLATE,
    RAG_USER_TEMPLATE,
    SYSTEM_PROMPT,
)
from vector_store import (
    HybridRetriever,
    create_project_embeddings,
    load_or_rebuild_campus_chroma,
    retrieve_top_chunks,
)


class ConversationMemory:
    """对话记忆（SQLite 持久化），按学生 ID 保留最近 N 轮对话。"""

    def __init__(self, max_rounds: int = 4):
        self.max_rounds = max_rounds

    def add(self, student_id: str, role: str, content: str, mode: str = "rag") -> None:
        save_conversation(student_id, role, content, mode)

    def get_history(self, student_id: str) -> list[dict[str, str]]:
        records = get_conversation_history(student_id, limit=self.max_rounds * 2)
        return [{"role": r["role"], "content": r["content"]} for r in records]

    def clear(self, student_id: str) -> None:
        clear_conversation_history(student_id)


_conversation_memory = ConversationMemory()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Campus Smart Q&A Assistant",
    description="Campus Q&A with RAG + MCP + auto-generated fallback",
    version="0.4.0",
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """返回前端页面。"""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="frontend index.html not found")
    return FileResponse(index_file)


@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_id: str = Field(default="S001", description="学生 ID（用于工具调用）")


class AskResponse(BaseModel):
    answer: str
    mode: str = "rag"
    auto_generated: bool = False
    tools_used: list[str] = Field(default_factory=list)
    tool_results: dict[str, str] = Field(default_factory=dict)


@lru_cache
def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=SecretStr(api_key),
        temperature=0.2,
    )


@lru_cache
def _get_vectorstore():
    """加载或自动初始化向量库。"""
    return load_or_rebuild_campus_chroma(
        data_directory=PROJECT_ROOT / "data",
        persist_directory=PROJECT_ROOT / "chroma_db",
        collection_name="campus_rag",
    )


@lru_cache
def _get_hybrid_retriever():
    """构建混合检索器（向量+BM25）。"""
    vs = _get_vectorstore()
    # 从向量库中提取所有文本作为 BM25 语料
    try:
        all_docs = vs._collection.get()
        corpus = all_docs.get("documents", []) or []
    except Exception:
        corpus = []
    return HybridRetriever(vs, corpus=corpus)


@lru_cache
def _get_mcp_client():
    """初始化 MCP 客户端。"""
    from mcp_client import MCPClient

    return MCPClient(PROJECT_ROOT / "mcp_server.py")


@lru_cache
def _get_intent_classifier():
    """初始化基于 Embedding 的意图分类器。"""
    embeddings = create_project_embeddings(allow_demo_fallback=True)[0]
    return IntentClassifier(embeddings)


async def _call_tools_for_intent(intent: str | None, question: str, student_id: str = "S001") -> dict[str, str]:
    """根据分类后的意图调用对应工具。"""
    if intent is None:
        return {}

    mcp_client = _get_mcp_client()
    results: dict[str, str] = {}
    intent_to_tool: dict[str, list[tuple[str, dict]]] = {
        "课表": [("get_course_schedule", {"student_id": student_id})],
        "借阅": [("get_library_status", {"student_id": student_id})],
        "教室": [("query_room_availability", {"room_id": "教学楼 101"})],
        "食堂": [("get_cafeteria_menu", {})],
        "校车": [("get_bus_schedule", {})],
        "天气": [("get_weather", {"city": question.replace("天气", "").replace("怎么样", "").replace("今天", "").replace("明天", "").strip() or "北京"})],
        "报修": [("submit_maintenance_request", {"student_id": student_id, "location": question, "description": question})],
    }

    tool_calls = intent_to_tool.get(intent, [])
    for tool_name, kwargs in tool_calls:
        label_map = {
            "get_course_schedule": "课表数据",
            "get_library_status": "借阅数据",
            "query_room_availability": "教室数据",
            "get_cafeteria_menu": "食堂菜单",
            "get_bus_schedule": "校车时刻",
            "submit_maintenance_request": "报修结果",
        }
        label = label_map.get(tool_name, tool_name)
        try:
            data = await mcp_client.call_tool(tool_name, **kwargs)
            results[label] = str(data)
        except Exception as exc:
            results[label] = f"获取失败: {exc}"

    return results


def _mark_auto_generated(answer: str, auto_generated: bool) -> str:
    """给自动生成的答案加标记。"""
    if not auto_generated:
        return answer
    stripped = answer.strip()
    if stripped.startswith("【自动生成】"):
        return stripped
    return f"【自动生成】{stripped}"


async def _build_context_prompt(question: str, student_id: str) -> tuple[str, list[str], dict[str, str], bool, str]:
    """统一构建上下文：MCP -> RAG -> 自动生成。"""
    tool_results: dict[str, str] = {}
    tools_used: list[str] = []
    auto_generated = False
    mode = "rag"

    # Step 1: MCP 工具调用（基于 Embedding 意图分类）
    classifier = _get_intent_classifier()
    intent = classifier.classify(question)
    if intent:
        tool_results = await _call_tools_for_intent(intent, question, student_id)
        if tool_results:
            tools_used = list(tool_results.keys())
            tool_context = "\n".join([f"【{k}】\n{v}" for k, v in tool_results.items()])
            llm_prompt = MCP_USER_TEMPLATE.format(tool_context=tool_context, question=question)
            return llm_prompt, tools_used, tool_results, auto_generated, "mcp"

    # Step 2: RAG 混合检索（向量 + BM25）
    hybrid = _get_hybrid_retriever()
    hits = await asyncio.to_thread(
        hybrid.retrieve,
        question,
        k=3,
        max_margin_from_best=0.22,
        max_distance=0.78,
    )
    if not hits:
        auto_generated = True
        mode = "auto"
        return AUTO_GENERATED_TEMPLATE.format(question=question), tools_used, tool_results, auto_generated, mode

    # Step 3: RAG 命中
    context_text = "\n\n".join(f"{idx + 1}. {hit['content']}" for idx, hit in enumerate(hits))
    return RAG_USER_TEMPLATE.format(context=context_text, question=question), tools_used, tool_results, auto_generated, mode


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, _auth=Depends(verify_api_key)) -> AskResponse:
    """主问答接口：自动 MCP -> RAG -> 自动生成（含多轮对话记忆）。"""
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, tools_used, tool_results, auto_generated, mode = await _build_context_prompt(
            question, student_id
        )
        history = _conversation_memory.get_history(student_id)
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(SystemMessage(content=f"你之前的回答：{h['content']}"))

        messages.append(HumanMessage(content=llm_prompt))

        response = await _get_llm().ainvoke(messages)
        answer = _mark_auto_generated(response.content.strip(), auto_generated)

        _conversation_memory.add(student_id, "user", question)
        _conversation_memory.add(student_id, "assistant", answer)
        log_interaction(student_id, question, answer, mode, auto_generated, tools_used)

        return AskResponse(
            answer=answer,
            mode=mode,
            auto_generated=auto_generated,
            tools_used=tools_used,
            tool_results=tool_results,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc


class AskWithToolsRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_id: str = Field(default="S001", description="学生 ID")


class AskWithToolsResponse(BaseModel):
    answer: str
    mode: str = "rag"
    auto_generated: bool = False
    tools_used: list[str] = Field(default_factory=list)
    tool_results: dict[str, str] = Field(default_factory=dict)


@app.post("/ask_with_tools", response_model=AskWithToolsResponse)
async def ask_with_tools(request: AskWithToolsRequest, _auth=Depends(verify_api_key)) -> AskWithToolsResponse:
    """工具感知问答接口，保留给需要显式查看工具结果的页面使用。"""
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, tools_used, tool_results, auto_generated, mode = await _build_context_prompt(
            question, student_id
        )
        history = _conversation_memory.get_history(student_id)
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(SystemMessage(content=f"你之前的回答：{h['content']}"))

        messages.append(HumanMessage(content=llm_prompt))

        response = await _get_llm().ainvoke(messages)
        answer = _mark_auto_generated(response.content.strip(), auto_generated)

        _conversation_memory.add(student_id, "user", question)
        _conversation_memory.add(student_id, "assistant", answer, mode)
        log_interaction(student_id, question, answer, mode, auto_generated, tools_used)

        return AskWithToolsResponse(
            answer=answer,
            mode=mode,
            auto_generated=auto_generated,
            tools_used=tools_used,
            tool_results=tool_results,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {exc}") from exc


class ResetRequest(BaseModel):
    student_id: str = Field(default="S001", description="学生 ID")


@app.post("/reset")
async def reset_conversation(request: ResetRequest, _auth=Depends(verify_api_key)) -> dict[str, str]:
    """清除指定学生 ID 的对话历史。"""
    _conversation_memory.clear(request.student_id)
    return {"status": "ok", "message": f"已清除 {request.student_id} 的对话历史"}


async def _stream_answer(question: str, student_id: str):
    """生成 SSE 事件流：先发元信息，再逐个发送 LLM token。"""
    llm_prompt, tools_used, tool_results, auto_generated, mode = await _build_context_prompt(
        question, student_id
    )
    yield f"data: {json.dumps({'event': 'meta', 'mode': mode, 'auto_generated': auto_generated, 'tools_used': tools_used})}\n\n"

    history = _conversation_memory.get_history(student_id)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for h in history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(SystemMessage(content=f"你之前的回答：{h['content']}"))
    messages.append(HumanMessage(content=llm_prompt))

    full_answer = ""
    async for chunk in _get_llm().astream(messages):
        if chunk.content:
            token = chunk.content
            full_answer += token
            yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

    final_answer = _mark_auto_generated(full_answer.strip(), auto_generated)
    _conversation_memory.add(student_id, "user", question)
    _conversation_memory.add(student_id, "assistant", final_answer, mode)
    log_interaction(student_id, question, final_answer, mode, auto_generated, tools_used)

    yield f"data: {json.dumps({'event': 'done'})}\n\n"


@app.post("/ask/stream")
async def ask_stream(request: AskRequest, _auth=Depends(verify_api_key)) -> StreamingResponse:
    """流式问答接口：SSE 格式返回，适合前端流式展示。"""
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    return StreamingResponse(
        _stream_answer(question, student_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
