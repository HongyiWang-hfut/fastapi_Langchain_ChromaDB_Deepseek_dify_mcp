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
    "如果资料为空或找不到答案，请直接回答：我不知道。"
)
ASK_USER_TEMPLATE = (
    "资料：\n{context}\n\n"
    "问题：{question}\n\n"
    "请基于资料给出简洁、准确的中文回答。"
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


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
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    try:
        vectorstore = _get_vectorstore()
        hits = retrieve_top_chunks(
            vectorstore,
            question,
            k=3,
            max_margin_from_best=0.22,
        )
        context_text = "\n\n".join(
            f"{idx + 1}. {hit['content']}" for idx, hit in enumerate(hits)
        )
        if not context_text:
            context_text = "（无相关资料）"

        messages = [
            SystemMessage(content=ASK_SYSTEM_PROMPT),
            HumanMessage(content=ASK_USER_TEMPLATE.format(context=context_text, question=question)),
        ]
        answer = _get_llm().invoke(messages).content.strip()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    return AskResponse(answer=answer)
