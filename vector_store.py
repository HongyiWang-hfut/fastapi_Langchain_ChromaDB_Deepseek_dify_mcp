"""校园 RAG：百炼向量嵌入 + Chroma 持久化 + 相似度检索。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import chromadb.config
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

_PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录路径


def _chroma_client_settings() -> chromadb.config.Settings:  # 创建 Chroma 客户端配置
    """供 Chroma 构造函数传入的 Settings（关闭遥测）。"""
    return chromadb.config.Settings(anonymized_telemetry=False)


def load_project_env() -> None:  # 从 .env 文件加载环境变量
    """把 ``_PROJECT_ROOT/.env`` 加载进环境变量。"""
    load_dotenv(_PROJECT_ROOT / ".env")


def create_dashscope_embeddings(  # 创建阿里百炼向量嵌入模型
    *,
    model: str = "text-embedding-v2",
    dashscope_api_key: str | None = None,
) -> DashScopeEmbeddings:
    """创建百炼 ``DashScopeEmbeddings``；未传 key 时读 ``DASHSCOPE_API_KEY``。"""
    if dashscope_api_key is None:
        load_project_env()
    kwargs: dict[str, Any] = {"model": model}  # LangChain 构造参数字典
    if dashscope_api_key:
        kwargs["dashscope_api_key"] = dashscope_api_key
    return DashScopeEmbeddings(**kwargs)


DEFAULT_CHROMA_DIR = "chroma_db"  # 默认向量库持久化目录名
DEFAULT_COLLECTION = "campus_rag"  # 默认 Chroma 集合名


def build_campus_chroma(  # 构建并持久化向量库
    chunks: list[Document],
    *,
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embeddings: DashScopeEmbeddings | None = None,
    reset: bool = False,
) -> Chroma:
    """将 ``chunks`` 嵌入后写入 Chroma；``reset`` 为真则先删库目录。"""
    persist_path = Path(persist_directory).resolve()  # 向量库完整路径
    if reset and persist_path.exists():
        shutil.rmtree(persist_path)

    embedder = embeddings or create_dashscope_embeddings()  # 嵌入模型实例
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(persist_path),
        collection_name=collection_name,
        client_settings=_chroma_client_settings(),
    )


def load_campus_chroma(  # 加载已保存的向量库
    *,
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embeddings: DashScopeEmbeddings | None = None,
) -> Chroma:
    """打开已落盘的 Chroma；嵌入配置须与建库时相同。"""
    embedder = embeddings or create_dashscope_embeddings()
    return Chroma(
        persist_directory=str(Path(persist_directory).resolve()),
        collection_name=collection_name,
        embedding_function=embedder,
        client_settings=_chroma_client_settings(),
    )


def retrieve_top_chunks(  # 相似度检索，返回最相关的文本片段
    vectorstore: Chroma,
    question: str,
    k: int = 3,
    *,
    fetch_k: int | None = None,
    max_margin_from_best: float | None = None,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """向量检索：返回 ``[{content, metadata, distance}, ...]``；``distance`` 越小越相似。"""
    if fetch_k is not None:
        n_fetch = fetch_k  # 首轮召回条数
    elif max_margin_from_best is not None or max_distance is not None:
        n_fetch = max(k * 5, 20)  # 过滤会丢条，提前多取候选
    else:
        n_fetch = k  # 精确取 k 条

    pairs = vectorstore.similarity_search_with_score(question, k=n_fetch)  # (Document, 距离) 对
    results: list[dict[str, Any]] = []  # 检索结果列表
    for doc, score in pairs:
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance": float(score),
            }
        )

    if results and max_margin_from_best is not None:
        best_d = results[0]["distance"]  # 最相关项的距离
        results = [
            r for r in results if r["distance"] <= best_d + max_margin_from_best
        ]

    if max_distance is not None:
        results = [r for r in results if r["distance"] <= max_distance]

    return results[:k]


def _main() -> None:  # 脚本主入口：数据→嵌入→持久化→检索演示
    """脚本入口：切片 → 建库 → 打印检索示例。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    load_project_env()
    os.chdir(_PROJECT_ROOT)  # 保证相对路径 DEFAULT_CHROMA_DIR 落在项目下

    from document_processing import load_and_split_campus_documents

    print("嵌入模型: 阿里百炼 text-embedding-v2 (DashScopeEmbeddings)")
    _, chunks = load_and_split_campus_documents(  # chunks: 文本切片列表
        _PROJECT_ROOT / "data",
        verbose=True,
    )

    vs = build_campus_chroma(chunks, reset=True)  # 向量库实例（可查可写）
    print(f"Chroma 已写入: {Path(DEFAULT_CHROMA_DIR).resolve()}  collection={DEFAULT_COLLECTION!r}")

    question = "图书馆借书能借多少册？续借怎么办理？"  # 查询问题
    print(f"\n查询: {question}\n")
    margin = 0.22  # 距离余量阈值
    hits = retrieve_top_chunks(vs, question, k=3, max_margin_from_best=margin)  # 检索结果
    print(
        f"（max_margin_from_best={margin}；不需要过滤时传 max_margin_from_best=None。）\n"
    )
    if not hits:
        print("无满足条件的片段，可调大 margin 或改用纯 Top‑k。\n")
    for i, hit in enumerate(hits, 1):
        print(f"--- 结果 {i} | distance={hit['distance']:.6f} ---")
        print(hit["content"])
        print()


if __name__ == "__main__":
    _main()
