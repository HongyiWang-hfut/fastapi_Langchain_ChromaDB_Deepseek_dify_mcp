"""MCP Server：使用官方 MCP SDK 暴露校园工具。"""

import hashlib
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

CAFETERIA_MENU = {
    "第一食堂": {
        "一楼大众餐": ["红烧肉 6元", "番茄炒蛋 4元", "清炒时蔬 3元", "米饭 1元"],
        "二楼风味窗口": ["麻辣香锅 25元", "酸菜鱼 18元", "兰州拉面 12元", "煲仔饭 15元"],
    },
    "第二食堂": {
        "一楼": ["牛肉面 12元", "刀削面 10元", "盖浇饭 13元"],
        "二楼": ["自助餐 20元/位", "小火锅 28元/位"],
    },
}

BUS_SCHEDULE = {
    "route_1": {
        "name": "南门 → 教学楼 → 图书馆 → 北门",
        "first_bus": "07:00",
        "last_bus": "22:00",
        "interval_peak": "10分钟",
        "interval_off_peak": "20分钟",
    },
    "route_2": {
        "name": "生活区 → 实验楼 → 体育馆 → 西门",
        "first_bus": "07:00",
        "last_bus": "22:00",
        "interval_peak": "10分钟",
        "interval_off_peak": "20分钟",
    },
    "route_3": {
        "name": "研究生院 → 科技园 → 地铁站",
        "first_bus": "07:30",
        "last_bus": "21:30",
        "interval_peak": "15分钟",
        "interval_off_peak": "30分钟",
    },
}

MAINTENANCE_REQUESTS: dict[str, list[dict]] = {}

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


def _get_cafeteria_menu_data(canteen: str | None = None) -> dict[str, Any]:
    """获取食堂菜单。"""
    if canteen and canteen not in CAFETERIA_MENU:
        return {"error": f"食堂 {canteen} 不存在，可选：{list(CAFETERIA_MENU.keys())}"}
    if canteen:
        return {canteen: CAFETERIA_MENU[canteen]}
    return {"menus": CAFETERIA_MENU, "total_canteens": len(CAFETERIA_MENU)}


def _get_bus_schedule_data(route_id: str | None = None) -> dict[str, Any]:
    """获取校车时刻表。"""
    if route_id and route_id not in BUS_SCHEDULE:
        return {"error": f"路线 {route_id} 不存在，可选：{list(BUS_SCHEDULE.keys())}"}
    if route_id:
        return {"route": route_id, **BUS_SCHEDULE[route_id]}
    return {"schedules": BUS_SCHEDULE, "total_routes": len(BUS_SCHEDULE)}


def _submit_maintenance_request_data(student_id: str, location: str, description: str) -> dict[str, Any]:
    """提交宿舍报修请求。"""
    request_id = f"MNT-{hashlib.md5(f'{student_id}{location}{description}'.encode()).hexdigest()[:8].upper()}"
    record = {
        "request_id": request_id,
        "student_id": student_id,
        "location": location,
        "description": description,
        "status": "已提交",
        "created_at": "2026-05-15 12:00:00",
        "expected_response": "2个工作日内",
    }
    if student_id not in MAINTENANCE_REQUESTS:
        MAINTENANCE_REQUESTS[student_id] = []
    MAINTENANCE_REQUESTS[student_id].append(record)
    return record


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


@mcp.tool(description="查询食堂菜单，不指定食堂则返回所有")
def get_cafeteria_menu(canteen: str | None = None) -> dict[str, Any]:
    return _get_cafeteria_menu_data(canteen)


@mcp.tool(description="查询校车时刻表，不指定路线则返回所有")
def get_bus_schedule(route_id: str | None = None) -> dict[str, Any]:
    return _get_bus_schedule_data(route_id)


@mcp.tool(description="提交宿舍报修请求")
def submit_maintenance_request(student_id: str, location: str, description: str) -> dict[str, Any]:
    return _submit_maintenance_request_data(student_id, location, description)


if __name__ == "__main__":
    mcp.run(transport="stdio")

