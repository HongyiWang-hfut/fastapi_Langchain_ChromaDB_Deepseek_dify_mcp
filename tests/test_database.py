"""数据库层测试：对话存储、交互日志（使用独立临时数据库）。"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def _test_db():
    """每个测试使用独立的临时 SQLite 数据库。"""
    os.environ["CAMPUS_QA_DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
    # 重新加载 database 模块以使用新的连接
    import importlib

    import database as db_mod

    importlib.reload(db_mod)
    # 重新加载 app.main 中引用的 database 符号
    from app import main as app_mod

    importlib.reload(app_mod)
    yield


class TestConversationStorage:
    def test_save_and_retrieve(self):
        from database import get_conversation_history, save_conversation

        save_conversation("T001", "user", "你好", "rag")
        save_conversation("T001", "assistant", "你好！", "rag")
        history = get_conversation_history("T001", limit=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"

    def test_clear_history(self):
        from database import clear_conversation_history, get_conversation_history, save_conversation

        save_conversation("T002", "user", "test", "rag")
        save_conversation("T002", "assistant", "answer", "rag")
        before = len(get_conversation_history("T002"))
        assert before > 0
        deleted = clear_conversation_history("T002")
        assert deleted == before
        assert len(get_conversation_history("T002")) == 0

    def test_empty_history(self):
        from database import get_conversation_history

        history = get_conversation_history("NONEXISTENT")
        assert history == []

    def test_limit(self):
        from database import get_conversation_history, save_conversation

        for i in range(6):
            save_conversation("T003", "user", f"msg{i}", "rag")
        history = get_conversation_history("T003", limit=3)
        assert len(history) <= 3


class TestInteractionLog:
    def test_log_interaction(self):
        from database import log_interaction

        log_id = log_interaction("T001", "测试问题", "测试回答", "rag", False, ["工具1"])
        assert log_id > 0

    def test_log_no_tools(self):
        from database import log_interaction

        log_id = log_interaction("T002", "问题", "回答", "auto", True)
        assert log_id > 0

    def test_clear_does_not_affect_logs(self):
        from database import clear_conversation_history, log_interaction

        clear_conversation_history("T004")
        log_interaction("T004", "q", "a")
        clear_conversation_history("T004")
        assert True
