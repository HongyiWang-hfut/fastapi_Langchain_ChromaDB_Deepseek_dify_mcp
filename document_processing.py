"""校园 RAG：从 ``data/`` 读入原始文件并切成文本块（LangChain ``Document``）。"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_campus_documents(data_dir: str | Path = "data") -> list[Document]:
    """加载目录下顶层 ``*.txt`` / ``*.pdf``，返回整篇级 ``Document`` 列表。"""
    base = Path(data_dir).resolve()  # 知识库目录绝对路径
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")

    documents: list[Document] = []  # 累积各 Loader.load() 的结果

    if any(base.glob("*.txt")):
        txt_loader = DirectoryLoader(  # 按扩展名批量文本加载
            str(base),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents.extend(txt_loader.load())

    if any(base.glob("*.pdf")):
        pdf_loader = DirectoryLoader(  # 每页通常对应一条 Document
            str(base),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
        documents.extend(pdf_loader.load())

    if not documents:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {base}. Add campus materials under this folder."
        )

    return documents


def split_campus_documents(
    documents: list[Document],
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> list[Document]:
    """对 ``documents`` 做递归字符切分，便于定长向量化。"""
    splitter = RecursiveCharacterTextSplitter(  # LangChain 内置分段策略
        chunk_size=chunk_size,  # 单块目标字符数（非 token）
        chunk_overlap=chunk_overlap,  # 块间重叠，减轻句子被拦腰截断
    )
    return splitter.split_documents(documents)


def load_and_split_campus_documents(
    data_dir: str | Path = "data",
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
    verbose: bool = True,
) -> tuple[list[Document], list[Document]]:
    """读盘 → 切分；返回 (原文档列表, 文本块列表)，可选打印数量。"""
    documents = load_campus_documents(data_dir)
    chunks = split_campus_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if verbose:
        print(f"已加载文档数量: {len(documents)}")
        print(f"分块数量: {len(chunks)}")
    return documents, chunks


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台 UTF-8，避免中文乱码
    except (AttributeError, OSError):
        pass

    project_root = Path(__file__).resolve().parent  # 本模块所在目录即项目根
    docs, splits = load_and_split_campus_documents(project_root / "data")
    if splits:
        print("\n--- 第一个分块预览（前 200 字符）---")
        preview_len = 200  # 只展示开头，避免刷屏
        print(splits[0].page_content[:preview_len].replace("\n", " "))
