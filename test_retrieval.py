import os
os.chdir('E:\\CS_Study\\practice\\projects\\fastapi_Langchain_ChromaDB_Deepseek_dify_mcp')

from vector_store import load_campus_chroma, retrieve_top_chunks

vs = load_campus_chroma()
question = '社团招新集中在每年 9 月第三、四周，地点在哪？'

print("=== 向量库状态检查 ===")
# 直接用底层 Chroma 方法查看集合
try:
    coll = vs._collection
    count = coll.count()
    print(f'向量库包含 {count} 个文档')
except Exception as e:
    print(f'获取向量库文档数失败: {e}')

print("\n=== 不加 max_margin_from_best 的检索结果 ===")
hits_no_filter = retrieve_top_chunks(vs, question, k=3)
print(f'无过滤: 检索到 {len(hits_no_filter)} 个片段')

print("\n=== 加 max_margin_from_best=0.22 的检索结果 ===")
hits_with_margin = retrieve_top_chunks(vs, question, k=3, max_margin_from_best=0.22)
print(f'margin=0.22: 检索到 {len(hits_with_margin)} 个片段')

# 诊断：获取原始的未过滤结果
print("\n=== 原始相似度搜索（无过滤） ===")
raw_results = vs.similarity_search_with_score(question, k=5)
print(f'原始检索（k=5）: {len(raw_results)} 个结果')
for i, (doc, score) in enumerate(raw_results):
    print(f'[{i+1}] score={score:.6f}: {doc.page_content[:60]}...')

