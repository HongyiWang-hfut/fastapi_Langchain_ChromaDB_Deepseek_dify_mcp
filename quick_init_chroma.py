"""快速初始化向量库（自动选择真实或虚拟嵌入）"""
import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

# 配置路径
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# 加载环境变量
load_dotenv(PROJECT_ROOT / ".env")

# 清理旧的向量库
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
    print(f"已删除旧的向量库: {CHROMA_DIR}")

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import numpy as np


class DummyEmbeddings(Embeddings):
    """测试用的虚拟嵌入：与 DashScope text-embedding-v2 保持一致（1536维）"""
    def embed_documents(self, texts):
        np.random.seed(42)
        return [np.random.randn(1536).tolist() for _ in texts]

    def embed_query(self, text):
        np.random.seed(42)
        return np.random.randn(1536).tolist()


# 加载文档
from document_processing import load_and_split_campus_documents

print("正在加载校园文档...")
_, chunks = load_and_split_campus_documents(PROJECT_ROOT / "data", verbose=True)

# 检查是否有 DashScope API Key
dashscope_key = os.getenv("DASHSCOPE_API_KEY")

if dashscope_key:
    print("\n✅ 检测到 DASHSCOPE_API_KEY，使用真实的阿里百炼嵌入...")
    from vector_store import create_dashscope_embeddings
    embedder = create_dashscope_embeddings(dashscope_api_key=dashscope_key)
else:
    print("\n⚠️  未检测到 DASHSCOPE_API_KEY，使用虚拟嵌入（仅用于测试）...")
    embedder = DummyEmbeddings()

# 用嵌入建库
print("正在初始化向量库...")
chroma = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory=str(CHROMA_DIR),
    collection_name="campus_rag",
)

print(f"\n✅ 向量库初始化成功！")
print(f"   位置: {CHROMA_DIR}")
print(f"   文档数: {chroma._collection.count()}")
print(f"   嵌入模型: {'阿里百炼 text-embedding-v2' if dashscope_key else '虚拟嵌入 (1536维)'}")
print(f"\n下一步：启动服务进行测试")
print(f"  uvicorn main:app --reload")

