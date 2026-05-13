"""校园智能问答助手：FastAPI HTTP 入口，启动时加载项目根 ``.env``。"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from vector_store import load_campus_chroma, retrieve_top_chunks

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
        api_key=api_key,
        temperature=0.2,
    )


@lru_cache
def _get_vectorstore():
    """加载或自动初始化向量库"""
    chroma_dir = Path("chroma_db")
    # 情况1：向量库已存在，直接加载
    if chroma_dir.exists():
        try:
            vs = load_campus_chroma()
            doc_count = vs._collection.count()
            if doc_count == 0:
                raise RuntimeError("向量库为空（0 个文档），需要重新初始化")
            return vs
        except Exception as e:
            raise RuntimeError(f"向量库加载失败: {e}")

    # 情况2：向量库不存在，自动初始化
    print("⚠️  向量库不存在，正在初始化...")
    try:
        from document_processing import load_and_split_campus_documents
        from langchain_core.embeddings import Embeddings
        from langchain_chroma import Chroma
        import numpy as np

        # 检查是否有真实的 DashScope API Key
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")

        if dashscope_key:
            # 使用真实的百炼嵌入
            print("  使用阿里百炼嵌入模型...")
            from vector_store import build_campus_chroma
            data_path = Path("data")
            if not data_path.exists():
                raise RuntimeError(f"data 目录不存在: {data_path.resolve()}")

            _, chunks = load_and_split_campus_documents(data_path, verbose=False)
            vs = build_campus_chroma(chunks, reset=False)
            print(f"✅ 向量库初始化成功 ({vs._collection.count()} 个文档)")
            return vs
        else:
            # 没有 API Key，使用虚拟嵌入进行演示
            print("  ⚠️  未检测到 DASHSCOPE_API_KEY，使用虚拟嵌入（仅用于演示）")

            class DummyEmbeddings(Embeddings):
                """虚拟嵌入：与 DashScope text-embedding-v2 保持一致（1536维）"""

                def embed_documents(self, texts):
                    np.random.seed(42)  # 固定种子保证示例可复现
                    return [np.random.randn(1536).tolist() for _ in texts]

                def embed_query(self, text):
                    np.random.seed(42)
                    return np.random.randn(1536).tolist()

            data_path = Path("data")
            if not data_path.exists():
                raise RuntimeError(f"data 目录不存在: {data_path.resolve()}")

            _, chunks = load_and_split_campus_documents(data_path, verbose=False)
            vs = Chroma.from_documents(
                documents=chunks,
                embedding=DummyEmbeddings(),
                persist_directory="chroma_db",
                collection_name="campus_rag",
            )
            print(f"✅ 向量库已用虚拟嵌入初始化 ({vs._collection.count()} 个文档)")
            print("   提示：为获得真实功能，请在 .env 中配置 DASHSCOPE_API_KEY")
            return vs

    except Exception as e:
        raise RuntimeError(
            f"向量库初始化失败: {e}\n"
            f"请确保 data/ 目录存在并包含校园资料文件。"
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
