"""Tests for graph/graph_builder.py — routing functions and build_graph."""
import pytest
from unittest.mock import patch, MagicMock
from graph.graph_builder import should_retry_or_notify, ranker_router, VERIFIER_RETRY_CAP


def _state(**overrides):
    base = {
        "messages": [],
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
    base.update(overrides)
    return base


def _hotel():
    return {
        "name": "Test Hotel",
        "price_per_night": 150.0,
        "distance_to_fishermans_wharf_miles": 0.5,
        "rating": 4.0,
        "availability": True,
        "booking_url": "https://example.com/test",
    }


class TestShouldRetryOrNotify:
    def test_error_routes_to_end(self):
        assert should_retry_or_notify(_state(error="something failed")) == "end_with_error"

    def test_verification_passed_routes_to_notifier(self):
        assert should_retry_or_notify(_state(verification_passed=True)) == "notifier"

    def test_failed_under_cap_routes_to_react_agent(self):
        state = _state(verification_passed=False, react_iterations=VERIFIER_RETRY_CAP - 1)
        assert should_retry_or_notify(state) == "react_agent"

    def test_failed_at_cap_routes_to_end(self):
        state = _state(verification_passed=False, react_iterations=VERIFIER_RETRY_CAP)
        assert should_retry_or_notify(state) == "end_with_error"

    def test_failed_beyond_cap_routes_to_end(self):
        state = _state(verification_passed=False, react_iterations=VERIFIER_RETRY_CAP + 5)
        assert should_retry_or_notify(state) == "end_with_error"

    def test_error_takes_precedence_over_passed(self):
        # error field should short-circuit even if verification_passed somehow is True
        state = _state(error="oops", verification_passed=True)
        assert should_retry_or_notify(state) == "end_with_error"

    def test_zero_iterations_can_retry(self):
        state = _state(verification_passed=False, react_iterations=0)
        assert should_retry_or_notify(state) == "react_agent"


class TestRankerRouter:
    def test_error_routes_to_end(self):
        assert ranker_router(_state(error="no hotels")) == "end_with_error"

    def test_empty_top_3_routes_to_end(self):
        assert ranker_router(_state(top_3_hotels=[])) == "end_with_error"

    def test_non_empty_top_3_routes_to_verifier(self):
        assert ranker_router(_state(top_3_hotels=[_hotel()])) == "verifier"

    def test_error_and_hotels_present_still_routes_to_end(self):
        state = _state(error="partial", top_3_hotels=[_hotel()])
        assert ranker_router(state) == "end_with_error"


class TestBuildGraph:
    def test_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch("graph.graph_builder.get_checkpointer", return_value=MemorySaver()):
            from graph.graph_builder import build_graph
            graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch("graph.graph_builder.get_checkpointer", return_value=MemorySaver()):
            from graph.graph_builder import build_graph
            graph = build_graph()
        node_names = set(graph.nodes.keys())
        for expected in ("planner", "react_agent", "ranker", "verifier", "notifier"):
            assert expected in node_names
