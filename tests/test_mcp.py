"""MCP 工具函数测试：直接测试 mcp_server.py 中的纯函数。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure mcp_server module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server import (
    _get_course_schedule_data,
    _get_library_status_data,
    _query_room_availability_data,
    _get_cafeteria_menu_data,
    _get_bus_schedule_data,
    _submit_maintenance_request_data,
)


class TestCourseSchedule:
    def test_existing_student_returns_courses(self):
        result = _get_course_schedule_data("S001")
        assert result["student_id"] == "S001"
        assert "courses" in result
        assert len(result["courses"]) == 3
        assert result["courses"][0]["name"] == "数据结构"

    def test_nonexistent_student_returns_error(self):
        result = _get_course_schedule_data("S999")
        assert "error" in result

    def test_empty_courses_student(self):
        result = _get_course_schedule_data("S003")
        assert result["student_id"] == "S003"
        assert result["courses"] == []


class TestLibraryStatus:
    def test_existing_student_has_books(self):
        result = _get_library_status_data("S001")
        assert result["student_id"] == "S001"
        assert len(result["borrowed_books"]) == 2

    def test_nonexistent_student_returns_error(self):
        result = _get_library_status_data("S999")
        assert "error" in result

    def test_student_no_records(self):
        result = _get_library_status_data("S003")
        assert result["borrowed_books"] == []


class TestRoomAvailability:
    def test_available_room_returns_slots(self):
        result = _query_room_availability_data("教学楼 101")
        assert result["room_id"] == "教学楼 101"
        assert "available_slots" in result
        assert "capacity" in result

    def test_nonexistent_room_returns_error(self):
        result = _query_room_availability_data("不存在")
        assert "error" in result


class TestCafeteriaMenu:
    def test_all_canteens(self):
        result = _get_cafeteria_menu_data()
        assert "menus" in result
        assert result["total_canteens"] >= 2

    def test_specific_canteen(self):
        result = _get_cafeteria_menu_data("第一食堂")
        assert "第一食堂" in result

    def test_nonexistent_canteen(self):
        result = _get_cafeteria_menu_data("第五食堂")
        assert "error" in result


class TestBusSchedule:
    def test_all_routes(self):
        result = _get_bus_schedule_data()
        assert "schedules" in result
        assert result["total_routes"] >= 3

    def test_specific_route(self):
        result = _get_bus_schedule_data("route_1")
        assert result["route"] == "route_1"
        assert "first_bus" in result

    def test_nonexistent_route(self):
        result = _get_bus_schedule_data("route_99")
        assert "error" in result


class TestMaintenanceRequest:
    def test_submit_creates_record(self):
        result = _submit_maintenance_request_data("S001", "南苑3号楼205", "灯管损坏")
        assert result["request_id"].startswith("MNT-")
        assert result["status"] == "已提交"
        assert result["location"] == "南苑3号楼205"
