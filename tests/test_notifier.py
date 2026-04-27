"""Tests for graph/nodes/notifier.py — notifier_node."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from graph.nodes.notifier import notifier_node


def _hotel(**overrides):
    base = {
        "name": "Test Hotel",
        "price_per_night": 150.0,
        "distance_to_fishermans_wharf_miles": 0.5,
        "rating": 4.0,
        "availability": True,
        "booking_url": "https://example.com/test",
    }
    base.update(overrides)
    return base


def _make_state(top_3=None, messages=None, **overrides):
    state = {
        "messages": messages or [HumanMessage(content="Find hotels")],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": [],
        "top_3_hotels": top_3 or [_hotel()],
        "verification_passed": True,
        "verification_notes": "All checks passed.",
        "notification_message": None,
        "react_iterations": 1,
        "error": None,
    }
    state.update(overrides)
    return state


class TestNotifierNodeHappyPath:
    def test_sets_notification_message(self):
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Here are 3 hotels!")
            result = notifier_node(_make_state())
        assert result["notification_message"] == "Here are 3 hotels!"

    def test_appends_ai_message_to_conversation(self):
        initial_msgs = [HumanMessage(content="Find hotels")]
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Notification text")
            result = notifier_node(_make_state(messages=initial_msgs))
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "Notification text"

    def test_does_not_modify_prior_messages(self):
        initial_msgs = [HumanMessage(content="Find hotels"), AIMessage(content="Searching...")]
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Done!")
            result = notifier_node(_make_state(messages=initial_msgs))
        assert result["messages"][0].content == "Find hotels"
        assert result["messages"][1].content == "Searching..."

    def test_preserves_other_state_fields(self):
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="msg")
            result = notifier_node(_make_state(react_iterations=3))
        assert result["react_iterations"] == 3
        assert result["verification_passed"] is True

    def test_passes_top_3_hotels_to_llm(self):
        hotels = [_hotel(name=f"Hotel {i}") for i in range(3)]
        captured = {}

        def capture(msgs):
            captured["msgs"] = msgs
            return MagicMock(content="notification")

        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture
            notifier_node(_make_state(top_3=hotels))

        import json
        human_content = captured["msgs"][1].content
        sent = json.loads(human_content)
        assert len(sent) == 3
        assert sent[0]["name"] == "Hotel 0"


class TestNotifierNodeEdgeCases:
    def test_empty_top_3_still_calls_llm(self):
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="No hotels found.")
            result = notifier_node(_make_state(top_3=[]))
        assert result["notification_message"] == "No hotels found."
        mock_llm.invoke.assert_called_once()

    def test_llm_exception_propagates(self):
        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.side_effect = RuntimeError("LLM error")
            with pytest.raises(RuntimeError, match="LLM error"):
                notifier_node(_make_state())
