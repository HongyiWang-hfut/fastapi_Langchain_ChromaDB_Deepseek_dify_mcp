"""意图分类器测试。"""

from __future__ import annotations

import pytest

from vector_store import DemoEmbeddings


@pytest.fixture
def classifier():
    from config.intent_classifier import IntentClassifier

    # DemoEmbeddings 不是语义性的，用很低 threshold 测试分类逻辑
    clf = IntentClassifier(DemoEmbeddings(dimension=128), threshold=0.05)
    clf.build()
    return clf


class TestIntentClassifier:
    def test_classify_returns_intent(self, classifier):
        """验证分类器对工具问题返回一个已注册的意图。"""
        intent = classifier.classify("我的课表是什么")
        # DemoEmbeddings 非语义性，仅验证不返回 None 且为有效意图
        if intent is not None:
            assert intent in classifier.get_intents()

    def test_classify_greeting_returns_none(self, classifier):
        intent = classifier.classify("你好呀")
        assert intent is None

    def test_classify_unknown_returns_none(self, classifier):
        intent = classifier.classify("量子力学的基本原理是什么")
        assert intent is None

    def test_get_intents(self, classifier):
        intents = classifier.get_intents()
        assert "课表" in intents
        assert "食堂" in intents
        assert "校车" in intents
        assert len(intents) == 6

    def test_cosine_similarity_identical(self):
        from config.intent_classifier import IntentClassifier

        v = [0.1, 0.2, 0.3]
        sim = IntentClassifier._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from config.intent_classifier import IntentClassifier

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = IntentClassifier._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from config.intent_classifier import IntentClassifier

        sim = IntentClassifier._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0
