"""Tests for graph/nodes/planner.py — planner_node."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from graph.nodes.planner import planner_node


def _make_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Find hotels near Fisherman's Wharf")],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": [],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 5,  # should be reset to 0
        "error": None,
    }
    state.update(overrides)
    return state


class TestPlannerNodeHappyPath:
    def test_returns_plan_string(self):
        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="1. Search\n2. Filter\n3. Rank")
            result = planner_node(_make_state())
        assert result["plan"] == "1. Search\n2. Filter\n3. Rank"

    def test_sets_search_query(self):
        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="plan")
            result = planner_node(_make_state())
        assert isinstance(result["search_query"], str)
        assert len(result["search_query"]) > 0

    def test_resets_react_iterations_to_zero(self):
        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="plan")
            result = planner_node(_make_state(react_iterations=5))
        assert result["react_iterations"] == 0

    def test_uses_last_message_as_user_input(self):
        messages = [
            HumanMessage(content="first message"),
            HumanMessage(content="second message"),
        ]
        captured = {}

        def capture_invoke(msgs):
            captured["msgs"] = msgs
            return MagicMock(content="plan")

        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture_invoke
            planner_node(_make_state(messages=messages))

        last_msg_content = captured["msgs"][-1].content
        assert "second message" in last_msg_content

    def test_preserves_other_state_fields(self):
        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="plan")
            state = _make_state(error="some prior error")
            result = planner_node(state)
        assert result["error"] == "some prior error"


class TestPlannerNodeErrors:
    def test_raises_on_empty_messages(self):
        with pytest.raises(ValueError, match="at least one message"):
            planner_node(_make_state(messages=[]))

    def test_llm_exception_propagates(self):
        with patch("graph.nodes.planner.llm") as mock_llm:
            mock_llm.invoke.side_effect = RuntimeError("API error")
            with pytest.raises(RuntimeError, match="API error"):
                planner_node(_make_state())
