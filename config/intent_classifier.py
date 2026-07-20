"""基于 Embedding 的意图分类器：替代关键词匹配进行工具路由。"""

from __future__ import annotations

import math
from typing import Any

from langchain_core.embeddings import Embeddings

# 每个工具对应的示例问题（语义原型）
_INTENT_EXAMPLES: dict[str, list[str]] = {
    "课表": [
        "我的课表是什么",
        "今天有什么课",
        "查看我的课程安排",
        "周一的课表",
        "我什么时候上课",
        "这学期的课程有哪些",
        "查一下课表",
    ],
    "借阅": [
        "我借了哪些书",
        "借阅记录查询",
        "我的借书情况",
        "图书馆借了几本书",
        "查一下我借的书什么时候到期",
    ],
    "教室": [
        "空教室查询",
        "教学楼 101 有空吗",
        "今天哪些教室可以用",
        "自习室哪里有空位",
        "查空闲教室",
    ],
    "食堂": [
        "今天食堂有什么菜",
        "食堂菜单",
        "第一食堂有什么吃的",
        "推荐一下食堂",
        "现在食堂开饭吗",
    ],
    "校车": [
        "校车时刻表",
        "校车路线",
        "怎么坐校车",
        "校车几点发车",
        "从南门到教学楼怎么走",
    ],
    "天气": [
        "今天天气怎么样",
        "天气预报",
        "北京多少度",
        "今天下雨吗",
        "气温多少",
        "明天天气如何",
    ],
    "报修": [
        "宿舍报修",
        "灯管坏了怎么报修",
        "宿舍漏水报修",
        "申请维修",
        "报修空调",
    ],
}

# 不触发工具的常见非工具问题
_NON_TOOL_EXAMPLES = [
    "你好",
    "谢谢",
    "再见",
    "介绍一下你自己",
    "你是谁",
    "你能做什么",
]


class IntentClassifier:
    """基于 Embedding 余弦相似度的轻量意图分类。"""

    def __init__(self, embeddings: Embeddings, threshold: float = 0.55):
        self._embeddings = embeddings
        self.threshold = threshold
        self._tool_examples: dict[str, list[list[float]]] = {}
        self._non_tool_vectors: list[list[float]] = []
        self._built = False

    def build(self) -> None:
        """将所有示例文本预向量化（批量调用以减少 API 往返）。

        DashScope text-embedding-v2 单次最多 25 条文本。
        原 40 次顺序 embed_query (~80s) 改为 2 次 embed_documents (~5s)。
        """
        # 收集所有示例文本，记录归属意图与位置
        all_texts: list[str] = []
        intent_slices: list[tuple[str, int, int]] = []  # (intent, start, count)
        for intent, examples in _INTENT_EXAMPLES.items():
            start = len(all_texts)
            all_texts.extend(examples)
            intent_slices.append((intent, start, len(examples)))
        non_tool_start = len(all_texts)
        all_texts.extend(_NON_TOOL_EXAMPLES)

        # 批量嵌入（分批，每批 25 条）
        all_vectors = self._batch_embed(all_texts)

        # 按意图分组存储
        for intent, start, count in intent_slices:
            self._tool_examples[intent] = all_vectors[start:start + count]
        self._non_tool_vectors = all_vectors[non_tool_start:]
        self._built = True

    def _batch_embed(self, texts: list[str], batch_size: int = 25) -> list[list[float]]:
        """分批调用 embed_documents，避免单次超限。"""
        result: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result.extend(self._embeddings.embed_documents(batch))
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _best_similarity(self, query_vec: list[float], vectors: list[list[float]]) -> float:
        if not vectors:
            return 0.0
        return max(self._cosine_similarity(query_vec, v) for v in vectors)

    def classify(self, question: str) -> str | None:
        """返回匹配的工具意图名称，若无匹配则返回 None。"""
        if not self._built:
            self.build()

        query_vec = self._embeddings.embed_query(question)

        # 先检查是否属于非工具类别
        max_non_tool_sim = self._best_similarity(query_vec, self._non_tool_vectors)
        if max_non_tool_sim >= self.threshold:
            return None

        # 检查每个工具类别的最高相似度
        best_intent: str | None = None
        best_score = 0.0
        for intent, vectors in self._tool_examples.items():
            score = self._best_similarity(query_vec, vectors)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent is not None and best_score >= self.threshold:
            return best_intent
        return None

    def get_intents(self) -> list[str]:
        """返回所有注册的工具意图。"""
        return list(_INTENT_EXAMPLES.keys())
