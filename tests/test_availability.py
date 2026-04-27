"""Tests for tools/availability.py — availability_check tool."""
import json
import pytest
from tools.availability import availability_check


class TestAvailabilityCheckHappyPath:
    def test_returns_string(self):
        result = availability_check.invoke(
            {"hotel_name": "Test Hotel", "check_in": "2025-06-01", "check_out": "2025-06-02"}
        )
        assert isinstance(result, str)

    def test_result_is_valid_json(self):
        result = availability_check.invoke(
            {"hotel_name": "Test Hotel", "check_in": "2025-06-01", "check_out": "2025-06-02"}
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_available_field(self):
        result = availability_check.invoke(
            {"hotel_name": "Test Hotel", "check_in": "2025-06-01", "check_out": "2025-06-02"}
        )
        parsed = json.loads(result)
        assert "available" in parsed

    def test_available_is_boolean(self):
        result = availability_check.invoke(
            {"hotel_name": "The Wharf Inn", "check_in": "2025-07-01", "check_out": "2025-07-02"}
        )
        parsed = json.loads(result)
        assert isinstance(parsed["available"], bool)

    def test_has_rooms_left(self):
        result = availability_check.invoke(
            {"hotel_name": "Test Hotel", "check_in": "2025-06-01", "check_out": "2025-06-02"}
        )
        parsed = json.loads(result)
        assert "rooms_left" in parsed
        assert isinstance(parsed["rooms_left"], int)

    def test_hotel_name_reflected_in_response(self):
        result = availability_check.invoke(
            {"hotel_name": "Argonaut Hotel", "check_in": "2025-06-01", "check_out": "2025-06-02"}
        )
        parsed = json.loads(result)
        assert parsed["hotel_name"] == "Argonaut Hotel"

    def test_dates_reflected_in_response(self):
        result = availability_check.invoke(
            {"hotel_name": "Test Hotel", "check_in": "2025-08-15", "check_out": "2025-08-17"}
        )
        parsed = json.loads(result)
        assert parsed["check_in"] == "2025-08-15"
        assert parsed["check_out"] == "2025-08-17"

    def test_different_hotels_return_response(self):
        for name in ["Hotel A", "Hotel B", "Hotel C"]:
            result = availability_check.invoke(
                {"hotel_name": name, "check_in": "2025-06-01", "check_out": "2025-06-02"}
            )
            parsed = json.loads(result)
            assert parsed["hotel_name"] == name
