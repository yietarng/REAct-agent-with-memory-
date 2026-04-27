"""Tests for graph/state.py — AgentState and HotelResult TypedDicts."""
import pytest
from graph.state import AgentState, HotelResult


def _hotel(**overrides) -> HotelResult:
    base = {
        "name": "Test Hotel",
        "price_per_night": 150.0,
        "distance_to_fishermans_wharf_miles": 0.5,
        "rating": 4.0,
        "availability": True,
        "booking_url": "https://example.com/test",
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


class TestHotelResult:
    def test_all_required_fields_present(self):
        h = _hotel()
        assert h["name"] == "Test Hotel"
        assert h["price_per_night"] == 150.0
        assert h["distance_to_fishermans_wharf_miles"] == 0.5
        assert h["rating"] == 4.0
        assert h["availability"] is True
        assert h["booking_url"] == "https://example.com/test"

    def test_price_accepts_integer(self):
        h = _hotel(price_per_night=200)
        assert h["price_per_night"] == 200

    def test_availability_false(self):
        h = _hotel(availability=False)
        assert h["availability"] is False


class TestAgentState:
    def test_messages_field_has_add_messages_metadata(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        meta = typing.get_args(hints["messages"])
        # Second element of Annotated args is the reducer
        from langgraph.graph.message import add_messages
        assert add_messages in meta

    def test_optional_fields_accept_none(self):
        from langchain_core.messages import HumanMessage
        state: AgentState = {
            "messages": [HumanMessage(content="hello")],
            "plan": None,
            "search_query": None,
            "raw_hotel_results": [],
            "top_3_hotels": [],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 0,
            "error": None,
        }
        assert state["plan"] is None
        assert state["error"] is None
