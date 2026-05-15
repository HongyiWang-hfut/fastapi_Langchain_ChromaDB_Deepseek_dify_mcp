"""校园智能问答助手：FastAPI HTTP 入口，支持 MCP、RAG 和自动生成回退（异步）。"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from vector_store import load_or_rebuild_campus_chroma, retrieve_top_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Campus Smart Q&A Assistant",
    description="Campus Q&A with RAG + MCP + auto-generated fallback",
    version="0.3.0",
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


ASK_SYSTEM_PROMPT = (
    "你是校园问答助手，只能依据提供的资料回答。"
    "如果有工具数据或检索资料，请优先依据它们回答。"
    "如果明确提示没有可用资料，请允许基于一般常识回答，并在答案开头标注：自动生成。"
    "如果仍然无法判断，请直接回答：我不知道。"
)
ASK_USER_TEMPLATE = (
    "资料：\n{context}\n\n"
    "问题：{question}\n\n"
    "请基于资料给出简洁、准确的中文回答。"
)
AUTO_GENERATED_USER_TEMPLATE = (
    "当前没有可用的工具数据，也没有检索到足够相关的资料。\n\n"
    "问题：{question}\n\n"
    "请基于一般常识尽量回答；如果你不确定，请明确说不知道。"
    "你的回答必须以【自动生成】开头。"
)


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
def _get_mcp_client():
    """初始化 MCP 客户端。"""
    from mcp_client import MCPClient

    return MCPClient(PROJECT_ROOT / "mcp_server.py")


def _should_use_tools(question: str) -> bool:
    """判断问题是否应该调用工具。"""
    keywords = ["课表", "课程", "上课", "借书", "借阅", "图书馆", "教室", "房间", "自习室"]
    return any(kw in question for kw in keywords)


async def _call_tools_for_question(question: str, student_id: str = "S001") -> dict[str, str]:
    """根据问题调用相应的工具，收集结果。"""
    try:
        mcp_client = _get_mcp_client()
        results: dict[str, str] = {}

        if any(kw in question for kw in ["课表", "课程", "上课"]):
            try:
                course_data = await mcp_client.call_tool("get_course_schedule", student_id=student_id)
                results["课表数据"] = str(course_data)
            except Exception as exc:
                results["课表数据"] = f"获取失败: {exc}"

        if any(kw in question for kw in ["借书", "借阅", "图书馆"]):
            try:
                library_data = await mcp_client.call_tool("get_library_status", student_id=student_id)
                results["借阅数据"] = str(library_data)
            except Exception as exc:
                results["借阅数据"] = f"获取失败: {exc}"

        if any(kw in question for kw in ["教室", "房间", "自习室"]):
            try:
                room_data = await mcp_client.call_tool("query_room_availability", room_id="教学楼 101")
                results["教室数据"] = str(room_data)
            except Exception as exc:
                results["教室数据"] = f"获取失败: {exc}"

        return results
    except Exception as exc:
        return {"error": f"工具调用失败: {exc}"}


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

    if _should_use_tools(question):
        tool_results = await _call_tools_for_question(question, student_id)
        if tool_results and "error" not in tool_results:
            tools_used = list(tool_results.keys())
            tool_context = "\n".join([f"【{k}】\n{v}" for k, v in tool_results.items()])
            llm_prompt = (
                "你是校园智能助手。根据以下实时数据回答用户的问题。\n\n"
                f"实时数据：\n{tool_context}\n\n"
                f"用户问题：{question}\n\n"
                "请用简洁、准确的中文回答，基于提供的实时数据整理信息。"
            )
            return llm_prompt, tools_used, tool_results, auto_generated, "mcp"

    vectorstore = _get_vectorstore()
    hits = await asyncio.to_thread(
        retrieve_top_chunks,
        vectorstore,
        question,
        k=3,
        max_margin_from_best=0.22,
        max_distance=0.78,
    )
    if not hits:
        auto_generated = True
        mode = "auto"
        return AUTO_GENERATED_USER_TEMPLATE.format(question=question), tools_used, tool_results, auto_generated, mode

    context_text = "\n\n".join(f"{idx + 1}. {hit['content']}" for idx, hit in enumerate(hits))
    return ASK_USER_TEMPLATE.format(context=context_text, question=question), tools_used, tool_results, auto_generated, mode


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """主问答接口：自动 MCP -> RAG -> 自动生成。"""
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, tools_used, tool_results, auto_generated, mode = await _build_context_prompt(question, student_id)
        messages = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content=llm_prompt),
        ]
        response = await _get_llm().ainvoke(messages)
        answer = _mark_auto_generated(response.content.strip(), auto_generated)
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
async def ask_with_tools(request: AskWithToolsRequest) -> AskWithToolsResponse:
    """工具感知问答接口，保留给需要显式查看工具结果的页面使用。"""
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, tools_used, tool_results, auto_generated, mode = await _build_context_prompt(question, student_id)
        messages = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content=llm_prompt),
        ]
        response = await _get_llm().ainvoke(messages)
        answer = _mark_auto_generated(response.content.strip(), auto_generated)
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
