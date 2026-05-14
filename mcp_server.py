"""MCP Server：使用官方 MCP SDK 暴露校园工具。"""

from typing import Any

from mcp.server.fastmcp import FastMCP

# ============================================================================
# 模拟教务系统数据库
# ============================================================================

STUDENT_COURSES = {
    "S001": [
        {"course_id": "CS101", "name": "数据结构", "time": "周一 09:00-11:00", "room": "教学楼 101"},
        {"course_id": "CS102", "name": "算法设计", "time": "周三 14:00-16:00", "room": "教学楼 201"},
        {"course_id": "MATH201", "name": "离散数学", "time": "周五 10:00-12:00", "room": "理学楼 301"},
    ],
    "S002": [
        {"course_id": "ENG101", "name": "英文写作", "time": "周二 10:00-12:00", "room": "文学楼 102"},
        {"course_id": "PHYS101", "name": "大学物理", "time": "周四 14:00-16:00", "room": "理学楼 205"},
    ],
    "S003": [],  # 没有课程的学生
}

LIBRARY_RESERVATIONS = {
    "S001": [
        {"book_id": "B123", "title": "深入理解计算机系统", "due_date": "2026-05-20", "status": "已借"},
        {"book_id": "B456", "title": "Python 高性能编程", "due_date": "2026-05-25", "status": "预约中"},
    ],
    "S002": [
        {"book_id": "B789", "title": "英文文献阅读指南", "due_date": "2026-05-30", "status": "已借"},
    ],
    "S003": [],
}

# ============================================================================
# 工具函数：实现具体的业务逻辑
# ============================================================================


def _get_course_schedule_data(student_id: str) -> dict[str, Any]:
    """获取学生的课表。"""
    if student_id not in STUDENT_COURSES:
        return {"error": f"学生 {student_id} 不存在"}

    courses = STUDENT_COURSES[student_id]
    if not courses:
        return {"student_id": student_id, "courses": [], "message": "该学生暂无课程安排"}

    return {
        "student_id": student_id,
        "courses": courses,
        "total": len(courses),
    }


def _get_library_status_data(student_id: str) -> dict[str, Any]:
    """获取学生的图书馆借阅状态。"""
    if student_id not in LIBRARY_RESERVATIONS:
        return {"error": f"学生 {student_id} 不存在"}

    records = LIBRARY_RESERVATIONS[student_id]
    if not records:
        return {"student_id": student_id, "borrowed_books": [], "message": "该学生暂无借阅记录"}

    return {
        "student_id": student_id,
        "borrowed_books": records,
        "total": len(records),
    }


def _query_room_availability_data(room_id: str) -> dict[str, Any]:
    """查询教室可用性（模拟数据）。"""
    available_rooms = {
        "教学楼 101": {"capacity": 50, "available_slots": ["12:00-14:00", "16:00-18:00"]},
        "教学楼 201": {"capacity": 60, "available_slots": ["09:00-11:00", "17:00-19:00"]},
        "理学楼 301": {"capacity": 80, "available_slots": ["13:00-15:00"]},
    }
    if room_id in available_rooms:
        return {"room_id": room_id, **available_rooms[room_id]}
    return {"error": f"教室 {room_id} 不存在"}


mcp = FastMCP("Campus Tools")


@mcp.tool(description="获取学生的课表")
def get_course_schedule(student_id: str) -> dict[str, Any]:
    return _get_course_schedule_data(student_id)


@mcp.tool(description="获取学生的图书馆借阅状态")
def get_library_status(student_id: str) -> dict[str, Any]:
    return _get_library_status_data(student_id)


@mcp.tool(description="查询教室的可用时间段")
def query_room_availability(room_id: str) -> dict[str, Any]:
    return _query_room_availability_data(room_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")

