"""API Key 认证依赖：通过 X-API-Key 头部验证请求。"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, status

# 默认开发用 Key，生产环境务必通过 API_KEY 环境变量覆盖
_API_KEY = os.getenv("API_KEY", "campus-qa-dev-key")


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """FastAPI 依赖：校验 X-API-Key 头部。"""
    if not x_api_key or x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
