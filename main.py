"""校园智能问答助手：FastAPI HTTP 入口，启动时加载项目根 ``.env``。"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SecretStr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from vector_store import load_or_rebuild_campus_chroma, retrieve_top_chunks

_env_path = Path(__file__).resolve().parent / ".env"  # 与 main.py 同目录，不随终端 cd 变化
load_dotenv(_env_path)  # 将 KEY 写入 os.environ，供后续路由与其它模块使用

app = FastAPI(  # ASGI 应用实例，供 uvicorn 挂载
    title="Campus Smart Q&A Assistant",
    description="Step 1: Hello World API",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """根路径探活。"""
    return {"message": "Hello World", "service": "campus-qa-assistant"}


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查（网关/编排常用）。"""
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
    project_root = Path(__file__).resolve().parent
    return load_or_rebuild_campus_chroma(
        data_directory=project_root / "data",
        persist_directory=project_root / "chroma_db",
        collection_name="campus_rag",
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    student_id = request.student_id.strip() or "S001"
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, _, _, auto_generated = _build_context_prompt(question, student_id)

        messages = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content=llm_prompt),
        ]
        answer = _get_llm().invoke(messages).content.strip()
        answer = _mark_auto_generated(answer, auto_generated)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    return AskResponse(answer=answer)


# ============================================================================
# MCP 工具调用集成
# ============================================================================


@lru_cache
def _get_mcp_client():
    """初始化 MCP 客户端。"""
    from mcp_client import MCPClient
    project_root = Path(__file__).resolve().parent
    return MCPClient(project_root / "mcp_server.py")


def _should_use_tools(question: str) -> bool:
    """判断问题是否应该调用工具（而不是向量库）。"""
    keywords = ["课表", "课程", "上课", "借书", "借阅", "图书馆", "教室", "房间", "自习室"]
    return any(kw in question for kw in keywords)


def _call_tools_for_question(question: str, student_id: str = "S001") -> dict[str, str]:
    """根据问题调用相应的工具，收集结果。"""
    try:
        mcp_client = _get_mcp_client()
        results = {}
        
        if any(kw in question for kw in ["课表", "课程", "上课"]):
            try:
                course_data = mcp_client.call_tool("get_course_schedule", student_id=student_id)
                results["课表数据"] = str(course_data)
            except Exception as e:
                results["课表数据"] = f"获取失败: {e}"
        
        if any(kw in question for kw in ["借书", "借阅", "图书馆"]):
            try:
                library_data = mcp_client.call_tool("get_library_status", student_id=student_id)
                results["借阅数据"] = str(library_data)
            except Exception as e:
                results["借阅数据"] = f"获取失败: {e}"
        
        if any(kw in question for kw in ["教室", "房间", "自习室"]):
            try:
                room_data = mcp_client.call_tool("query_room_availability", room_id="教学楼 101")
                results["教室数据"] = str(room_data)
            except Exception as e:
                results["教室数据"] = f"获取失败: {e}"
        
        return results
    except Exception as e:
        return {"error": f"工具调用失败: {e}"}


def _build_context_prompt(question: str, student_id: str) -> tuple[str, list[str], dict[str, str], bool]:
    """统一构建上下文：优先尝试 MCP 工具，不可用时自动降级到 RAG，最后自动生成。"""
    tool_results: dict[str, str] = {}
    tools_used: list[str] = []
    auto_generated = False

    if _should_use_tools(question):
        tool_results = _call_tools_for_question(question, student_id)
        if tool_results and "error" not in tool_results:
            tools_used = list(tool_results.keys())
            tool_context = "\n".join([f"【{k}】\n{v}" for k, v in tool_results.items()])
            llm_prompt = (
                "你是校园智能助手。根据以下实时数据回答用户的问题。\n\n"
                f"实时数据：\n{tool_context}\n\n"
                f"用户问题：{question}\n\n"
                "请用简洁、准确的中文回答，基于提供的实时数据整理信息。"
            )
            return llm_prompt, tools_used, tool_results, auto_generated

    vectorstore = _get_vectorstore()
    hits = retrieve_top_chunks(
        vectorstore,
        question,
        k=3,
        max_margin_from_best=0.22,
        max_distance=0.78,
    )
    if not hits:
        auto_generated = True
        return AUTO_GENERATED_USER_TEMPLATE.format(question=question), tools_used, tool_results, auto_generated

    context_text = "\n\n".join(
        f"{idx + 1}. {hit['content']}" for idx, hit in enumerate(hits)
    )
    return ASK_USER_TEMPLATE.format(context=context_text, question=question), tools_used, tool_results, auto_generated


def _mark_auto_generated(answer: str, auto_generated: bool) -> str:
    """给自动生成的答案加标记。"""
    if not auto_generated:
        return answer
    stripped = answer.strip()
    if stripped.startswith("【自动生成】"):
        return stripped
    return f"【自动生成】{stripped}"


class AskWithToolsRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_id: str = Field(default="S001", description="学生 ID")


class AskWithToolsResponse(BaseModel):
    answer: str
    tools_used: list[str] = []
    tool_results: dict[str, str] = {}


@app.post("/ask_with_tools", response_model=AskWithToolsResponse)
def ask_with_tools(request: AskWithToolsRequest) -> AskWithToolsResponse:
    """
    智能问答接口：支持工具调用。
    如果问题涉及课表、借书等，会自动调用相应工具获取实时数据，再由大模型整理回答。
    """
    question = request.question.strip()
    student_id = request.student_id.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        llm_prompt, tools_used, tool_results, auto_generated = _build_context_prompt(question, student_id)

        # 调用大模型
        messages = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content=llm_prompt),
        ]
        answer = _get_llm().invoke(messages).content.strip()
        answer = _mark_auto_generated(answer, auto_generated)

        return AskWithToolsResponse(
            answer=answer,
            tools_used=tools_used,
            tool_results=tool_results,
        )
    
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {exc}") from exc
