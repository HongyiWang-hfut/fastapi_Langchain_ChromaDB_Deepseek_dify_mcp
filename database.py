"""SQLite 持久化层：对话历史、交互日志。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = Path(__file__).resolve().parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
_DEFAULT_DB = f"sqlite:///{DB_DIR / 'campus_qa.db'}"
DATABASE_URL = os.environ.get("CAMPUS_QA_DATABASE_URL", _DEFAULT_DB)

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class ConversationRecord(Base):
    """对话记录表：按学生 ID 存储多轮对话。"""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(32), index=True, nullable=False)
    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    mode = Column(String(16), default="rag")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class InteractionLog(Base):
    """交互日志表：记录每次问答的元信息。"""

    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(32), index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    mode = Column(String(16), default="rag")
    auto_generated = Column(Integer, default=0)
    tools_used = Column(String(256), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """创建所有表（幂等）。"""
    Base.metadata.create_all(bind=engine)


def save_conversation(student_id: str, role: str, content: str, mode: str = "rag") -> int:
    """保存一条对话记录，返回记录 ID。"""
    with SessionLocal() as session:
        record = ConversationRecord(student_id=student_id, role=role, content=content, mode=mode)
        session.add(record)
        session.commit()
        return record.id


def get_conversation_history(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """获取指定学生最近的对话历史。"""
    with SessionLocal() as session:
        records = (
            session.query(ConversationRecord)
            .filter(ConversationRecord.student_id == student_id)
            .order_by(ConversationRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {"role": r.role, "content": r.content, "mode": r.mode, "created_at": r.created_at.isoformat()}
            for r in reversed(records)
        ]


def clear_conversation_history(student_id: str) -> int:
    """清除指定学生的对话历史，返回删除条数。"""
    with SessionLocal() as session:
        deleted = session.query(ConversationRecord).filter(
            ConversationRecord.student_id == student_id
        ).delete()
        session.commit()
        return deleted


def log_interaction(
    student_id: str,
    question: str,
    answer: str,
    mode: str = "rag",
    auto_generated: bool = False,
    tools_used: list[str] | None = None,
) -> int:
    """记录一次问答交互日志，返回日志 ID。"""
    with SessionLocal() as session:
        log = InteractionLog(
            student_id=student_id,
            question=question,
            answer=answer,
            mode=mode,
            auto_generated=int(auto_generated),
            tools_used=",".join(tools_used or []),
        )
        session.add(log)
        session.commit()
        return log.id


# 启动时自动建表
init_db()
