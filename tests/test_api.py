"""API 层测试：覆盖 /health、/ask、/ask_with_tools、/reset、/ask/stream。"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mock API keys before importing app
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-key")

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_memory():
    """每个测试前重置对话记忆。"""
    from app.main import _conversation_memory

    _conversation_memory._store.clear()
    yield


class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAskEndpoint:
    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_returns_answer(self, mock_build, mock_llm):
        mock_build.return_value = ("test prompt", [], {}, False, "rag")
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = AsyncMock(return_value=MagicMock(content="图书馆7:30开门"))
        mock_llm.return_value = mock_llm_inst

        resp = client.post("/ask", json={"question": "图书馆几点开门？"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "图书馆7:30开门"
        assert data["mode"] == "rag"
        assert data["auto_generated"] is False

    def test_ask_empty_question_returns_422(self):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_mcp_mode(self, mock_build, mock_llm):
        mock_build.return_value = ("tool prompt", ["课表数据"], {"课表数据": "..."}, False, "mcp")
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = AsyncMock(return_value=MagicMock(content="你的课表是..."))
        mock_llm.return_value = mock_llm_inst

        resp = client.post("/ask", json={"question": "我的课表是什么？", "student_id": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "mcp"
        assert "课表数据" in data["tools_used"]

    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_auto_generated(self, mock_build, mock_llm):
        mock_build.return_value = ("auto prompt", [], {}, True, "auto")
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = AsyncMock(return_value=MagicMock(content="【自动生成】我是AI助手"))
        mock_llm.return_value = mock_llm_inst

        resp = client.post("/ask", json={"question": "介绍你自己"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["auto_generated"] is True

    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_conversation_memory(self, mock_build, mock_llm):
        mock_build.return_value = ("prompt", [], {}, False, "rag")
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = AsyncMock(return_value=MagicMock(content="回答"))
        mock_llm.return_value = mock_llm_inst

        client.post("/ask", json={"question": "第一轮", "student_id": "S001"})
        client.post("/ask", json={"question": "第二轮", "student_id": "S001"})

        from app.main import _conversation_memory
        history = _conversation_memory.get_history("S001")
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "第一轮"


class TestAskWithToolsEndpoint:
    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_with_tools_returns_tool_details(self, mock_build, mock_llm):
        mock_build.return_value = ("prompt", ["课表数据"], {"课表数据": '{"courses":[]}'}, False, "mcp")
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = AsyncMock(return_value=MagicMock(content="课表数据"))
        mock_llm.return_value = mock_llm_inst

        resp = client.post("/ask_with_tools", json={"question": "课表", "student_id": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert "课表数据" in data["tool_results"]


class TestResetEndpoint:
    def test_reset_clears_memory(self):
        from app.main import _conversation_memory
        _conversation_memory.add("S001", "user", "test")
        assert len(_conversation_memory.get_history("S001")) > 0

        resp = client.post("/reset", json={"student_id": "S001"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert len(_conversation_memory.get_history("S001")) == 0


class TestStreamEndpoint:
    @patch("app.main._get_llm")
    @patch("app.main._build_context_prompt")
    def test_ask_stream_returns_sse_events(self, mock_build, mock_llm):
        mock_build.return_value = ("prompt", [], {}, False, "rag")

        class AsyncIterator:
            def __init__(self):
                self.tokens = ["你好", "，", "世界"]
                self.idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.idx >= len(self.tokens):
                    raise StopAsyncIteration
                val = self.tokens[self.idx]
                self.idx += 1
                return MagicMock(content=val)

        mock_llm_inst = MagicMock()
        mock_llm_inst.astream = MagicMock(return_value=AsyncIterator())
        mock_llm.return_value = mock_llm_inst

        resp = client.post(
            "/ask/stream",
            json={"question": "你好", "student_id": "S001"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
