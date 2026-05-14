"""校园 RAG：百炼向量嵌入 + Chroma 持久化 + 相似度检索。"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import chromadb.config
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

_PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录路径

# 配置日志格式（便于运行时观察）
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
_logger = logging.getLogger(__name__)


def _chroma_client_settings() -> chromadb.config.Settings:  # 创建 Chroma 客户端配置
    """供 Chroma 构造函数传入的 Settings（关闭遥测）。"""
    return chromadb.config.Settings(anonymized_telemetry=False)


def load_project_env() -> None:  # 从 .env 文件加载环境变量
    """把 ``_PROJECT_ROOT/.env`` 加载进环境变量。"""
    load_dotenv(_PROJECT_ROOT / ".env")


EMBEDDING_PROBE_TEXT = "__embedding_dimension_probe__"  # 维度探测文本
DEMO_EMBEDDING_DIMENSION = 1536  # 演示嵌入固定维度


class DemoEmbeddings(Embeddings):
    """本地演示嵌入：无 API Key 时使用，维度固定且可复现。"""

    def __init__(self, dimension: int = DEMO_EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def _vector_for_text(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for_text(text)


def _probe_embedding_dimension(embeddings: Embeddings) -> int:
    """通过一次查询向量推断当前嵌入维度。"""
    return len(embeddings.embed_query(EMBEDDING_PROBE_TEXT))


def create_project_embeddings(  # 统一创建项目内的嵌入模型
    *,
    allow_demo_fallback: bool = True,
) -> tuple[Embeddings, str, int]:
    """返回 ``(embeddings, mode, dimension)``；mode 为 ``真实嵌入`` 或 ``演示嵌入``。"""
    load_project_env()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        embeddings: Embeddings = create_dashscope_embeddings(dashscope_api_key=api_key)
        mode = "真实嵌入"
    elif allow_demo_fallback:
        embeddings = DemoEmbeddings()
        mode = "演示嵌入"
    else:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")

    dimension = _probe_embedding_dimension(embeddings)
    return embeddings, mode, dimension


def get_chroma_collection_dimension(vectorstore: Chroma) -> int | None:
    """从 Chroma 集合里读取已保存向量的维度。"""
    try:
        peek = vectorstore._collection.peek()
        embeddings = peek.get("embeddings")
        if embeddings is None:
            return None
        if hasattr(embeddings, "shape") and len(embeddings.shape) >= 2:
            return int(embeddings.shape[1])
        if len(embeddings) > 0:
            first = embeddings[0]
            if hasattr(first, "__len__"):
                return len(first)
    except Exception:
        return None
    return None


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
    embeddings: Embeddings | None = None,
    reset: bool = False,
) -> Chroma:
    """将 ``chunks`` 嵌入后写入 Chroma；``reset`` 为真则先删库目录。"""
    persist_path = Path(persist_directory).resolve()  # 向量库完整路径
    if reset and persist_path.exists():
        shutil.rmtree(persist_path)

    embedder = embeddings or create_project_embeddings()[0]  # 嵌入模型实例
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
    embeddings: Embeddings | None = None,
) -> Chroma:
    """打开已落盘的 Chroma；嵌入配置须与建库时相同。"""
    embedder = embeddings or create_project_embeddings()[0]
    return Chroma(
        persist_directory=str(Path(persist_directory).resolve()),
        collection_name=collection_name,
        embedding_function=embedder,
        client_settings=_chroma_client_settings(),
    )


def load_or_rebuild_campus_chroma(  # 加载向量库；异常/空库/维度不一致时自动重建
    *,
    data_directory: str | Path = _PROJECT_ROOT / "data",
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    allow_demo_fallback: bool = True,
) -> Chroma:
    """优先加载现有 Chroma；若为空、维度不一致或加载失败，则用 ``data/`` 重建。"""
    data_path = Path(data_directory).resolve()
    persist_path = Path(persist_directory).resolve()

    embeddings, mode, expected_dim = create_project_embeddings(
        allow_demo_fallback=allow_demo_fallback
    )
    api_key_status = "已配置" if os.getenv("DASHSCOPE_API_KEY") else "未配置"
    _logger.info(
        f"向量库初始化 | DASHSCOPE_API_KEY={api_key_status} | mode={mode} | embedding_dim={expected_dim}"
    )

    if persist_path.exists():
        try:
            vectorstore = load_campus_chroma(
                persist_directory=persist_path,
                collection_name=collection_name,
                embeddings=embeddings,
            )
            doc_count = vectorstore._collection.count()
            actual_dim = get_chroma_collection_dimension(vectorstore)
            if doc_count > 0 and actual_dim is None:
                try:
                    vectorstore.similarity_search_with_score(EMBEDDING_PROBE_TEXT, k=1)
                except Exception as exc:
                    raise RuntimeError(f"向量库维度校验失败: {exc}") from exc
                print(
                    f"✅ 向量库可用 | docs={doc_count} | embedding_dim=auto-verified | collection={collection_name!r}"
                )
                return vectorstore

            if doc_count > 0 and actual_dim == expected_dim:
                _logger.info(
                    f"✅ 向量库可用 | docs={doc_count} | embedding_dim={actual_dim} | collection={collection_name!r}"
                )
                return vectorstore

            reason = "空集合" if doc_count == 0 else f"维度不一致（现有 {actual_dim}，当前 {expected_dim}）"
            _logger.warning(f"向量库需要重建 | reason={reason}")
        except Exception as exc:
            _logger.warning(f"向量库加载/校验失败 | err={type(exc).__name__}: {exc} | 准备重建")
    else:
        _logger.info("未发现现有向量库，准备从 data/ 初始化")

    if not data_path.exists():
        raise RuntimeError(f"data 目录不存在: {data_path}")

    from document_processing import load_and_split_campus_documents

    _, chunks = load_and_split_campus_documents(data_path, verbose=False)
    if not chunks:
        raise RuntimeError(f"data 目录中没有可用于建库的文档: {data_path}")

    _logger.info(f"开始构建向量库 | chunks={len(chunks)} | persist_dir={persist_path}")
    vectorstore = build_campus_chroma(
        chunks,
        persist_directory=persist_path,
        collection_name=collection_name,
        embeddings=embeddings,
        reset=True,
    )
    actual_dim = get_chroma_collection_dimension(vectorstore)
    _logger.info(
        f"✅ 向量库已重建 | docs={vectorstore._collection.count()} | embedding_dim={actual_dim or expected_dim} | mode={mode}"
    )
    return vectorstore


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
