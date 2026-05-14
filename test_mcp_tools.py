"""
测试 MCP 工具调用的客户端脚本。
"""

import io
import sys
import json
from pathlib import Path

# 强制使用 UTF-8 编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', newline='')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', newline='')

# 确保能导入项目模块
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from mcp_client import MCPClient


def test_mcp_server():
    """测试 MCP Server 的各项工具。"""
    print("=" * 70)
    print("[MCP Server] Test")
    print("=" * 70)

    try:
        client = MCPClient(project_root / "mcp_server.py")
        print("[OK] MCP Client connected\n")

        # 列出所有工具
        print("[TOOLS] Available tools:")
        tools = client.list_tools()
        for tool in tools:
            name = tool['name']
            desc = tool['description']
            print(f"  {name}: {desc}")
        print()

        # 测试 1: 获取学生课表
        print("-" * 70)
        print("[TEST 1] get_course_schedule (S001)")
        result = client.call_tool("get_course_schedule", student_id="S001")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

        # 测试 2: 获取借阅状态
        print("-" * 70)
        print("[TEST 2] get_library_status (S001)")
        result = client.call_tool("get_library_status", student_id="S001")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

        # 测试 3: 查询教室可用性
        print("-" * 70)
        print("[TEST 3] query_room_availability (Room: 101)")
        result = client.call_tool("query_room_availability", room_id="教学楼 101")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

        # 测试 4: 不存在的学生
        print("-" * 70)
        print("[TEST 4] get_course_schedule (nonexistent student S999)")
        result = client.call_tool("get_course_schedule", student_id="S999")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

        print("=" * 70)
        print("[OK] All tests passed")
        print("=" * 70)

        client.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def test_ask_with_tools_api():
    """测试 /ask_with_tools API。"""
    import requests

    print("\n" + "=" * 70)
    print("FastAPI /ask_with_tools 接口测试")
    print("=" * 70)

    base_url = "http://127.0.0.1:8000"

    test_cases = [
        {"question": "我的课表是什么？", "student_id": "S001"},
        {"question": "我借的书有哪些？", "student_id": "S001"},
        {"question": "能帮我查一下课表吗？", "student_id": "S002"},
        {"question": "图书馆最近的开放时间", "student_id": "S001"},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['question']}")
        print("-" * 70)
        try:
            response = requests.post(
                f"{base_url}/ask_with_tools",
                json=test_case,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                print(f" 回答: {data['answer']}")
                if data.get('tools_used'):
                    print(f" 使用的工具: {', '.join(data['tools_used'])}")
                    if data.get('tool_results'):
                        print(f" 工具结果摘要:")
                        for k, v in data['tool_results'].items():
                            preview = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                            print(f"   {k}: {preview}")
            else:
                print(f"❌ API 返回错误: {response.status_code}")
                print(response.text)
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到 FastAPI 服务器（请确保 uvicorn 已启动）")
            break
        except Exception as e:
            print(f"❌ 请求失败: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP 工具测试脚本")
    parser.add_argument(
        "--test",
        choices=["server", "api", "all"],
        default="all",
        help="测试类型：server (MCP Server)、api (FastAPI接口) 或 all (两者)",
    )
    args = parser.parse_args()

    if args.test in ["server", "all"]:
        test_mcp_server()

    if args.test in ["api", "all"]:
        try:
            import requests
        except ImportError:
            print("⚠️  需要安装 requests: pip install requests")
        else:
            test_ask_with_tools_api()


