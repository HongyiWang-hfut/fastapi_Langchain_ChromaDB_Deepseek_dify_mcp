"""MCP Client：使用官方 MCP SDK 通过 stdio 调用工具（原生异步）。"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """原生异步 MCP 客户端，每次调用创建临时会话。"""

    def __init__(self, server_script: str | Path):
        """初始化客户端，仅保存服务端脚本路径与参数。"""
        self.server_script = Path(server_script)
        if not self.server_script.exists():
            raise RuntimeError(f"MCP Server 脚本不存在: {self.server_script}")

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env=env,
        )
        self._tools_cache = None

    @staticmethod
    def _extract_call_result(result: Any) -> Any:
        """将官方 CallToolResult 映射成业务层可直接使用的 Python 数据。"""
        content = getattr(result, "content", None) or []
        if len(content) == 1 and hasattr(content[0], "text"):
            text = content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text
        return [
            item.model_dump() if hasattr(item, "model_dump") else str(item)
            for item in content
        ]

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取服务器支持的所有工具（结果已缓存）。"""
        if self._tools_cache is None:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    self._tools_cache = [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {},
                        }
                        for tool in result.tools
                    ]
        return self._tools_cache

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """调用指定工具。"""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=kwargs)
                if getattr(result, "isError", False):
                    raise RuntimeError(f"MCP 工具执行失败: {tool_name}")
                return self._extract_call_result(result)

    def close(self):
        """兼容旧接口：SDK 方案按次创建会话，无需显式关闭。"""
        return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
