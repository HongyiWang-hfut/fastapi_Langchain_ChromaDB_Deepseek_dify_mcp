"""MCP Client：使用官方 MCP SDK 通过 stdio 调用工具。"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """同步封装：内部使用官方 MCP SDK。"""

    def __init__(self, server_script: str | Path):
        """初始化客户端，启动 MCP Server 进程。"""
        self.server_script = Path(server_script)
        if not self.server_script.exists():
            raise RuntimeError(f"MCP Server 脚本不存在: {self.server_script}")
        
        # 统一子进程编码为 UTF-8，避免 Windows 本地编码（如 GBK）导致解码失败。
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env=env,
        )
        self.tools_cache = None

    @staticmethod
    def _extract_call_result(result: Any) -> Any:
        """将官方 CallToolResult 映射成业务层可直接使用的 Python 数据。"""
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        content = getattr(result, "content", None) or []
        if len(content) == 1 and hasattr(content[0], "text"):
            text = content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text

        return [item.model_dump() if hasattr(item, "model_dump") else str(item) for item in content]

    def list_tools(self) -> list[dict[str, Any]]:
        """获取服务器支持的所有工具。"""
        if self.tools_cache is None:
            async def _list() -> list[dict[str, Any]]:
                async with stdio_client(self.server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        tools: list[dict[str, Any]] = []
                        for tool in result.tools:
                            tools.append(
                                {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "parameters": tool.inputSchema or {},
                                }
                            )
                        return tools

            self.tools_cache = anyio.run(_list)
        return self.tools_cache
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """调用指定工具。"""
        async def _call() -> Any:
            async with stdio_client(self.server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=kwargs)
                    if getattr(result, "isError", False):
                        raise RuntimeError(f"MCP 工具执行失败: {tool_name}")
                    return self._extract_call_result(result)

        return anyio.run(_call)

    def close(self):
        """兼容旧接口：官方 SDK 方案按次创建会话，无持久连接需关闭。"""
        return None

    def __del__(self):
        """析构函数，确保进程被关闭。"""
        try:
            self.close()
        except Exception:
            pass
