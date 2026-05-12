"""Campus RAG: load documents from data/ and split into chunks."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_campus_documents(data_dir: str | Path = "data") -> list[Document]:
    """Load all ``*.txt`` and ``*.pdf`` in ``data_dir`` (top level only).

    Uses :class:`~langchain_community.document_loaders.DirectoryLoader` with
    :class:`~langchain_community.document_loaders.TextLoader` / ``PyPDFLoader``.
    """
    base = Path(data_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")

    documents: list[Document] = []

    if any(base.glob("*.txt")):
        txt_loader = DirectoryLoader(
            str(base),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents.extend(txt_loader.load())

    if any(base.glob("*.pdf")):
        pdf_loader = DirectoryLoader(
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
    chunk_size: int = 100,
    chunk_overlap: int = 10,
) -> list[Document]:
    """Split documents with fixed-size recursive character chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_and_split_campus_documents(
    data_dir: str | Path = "data",
    *,
    chunk_size: int = 100,
    chunk_overlap: int = 10,
    verbose: bool = True,
) -> tuple[list[Document], list[Document]]:
    """Load from disk, split, optionally print counts, return (docs, chunks)."""
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
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    # 调用示例：在项目根目录执行  python document_processing.py
    project_root = Path(__file__).resolve().parent
    docs, splits = load_and_split_campus_documents(project_root / "data")
    if splits:
        print("\n--- 第一个分块预览（前 200 字符）---")
        print(splits[0].page_content[:200].replace("\n", " "))
