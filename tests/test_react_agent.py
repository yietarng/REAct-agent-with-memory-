"""Tests for graph/nodes/react_agent.py — react_agent_node and _parse_hotel_results."""
import json
import logging
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph.nodes.react_agent import react_agent_node, _parse_hotel_results, MAX_ITERATIONS


def _tool_msg(content, name="hotel_search", tool_call_id="call_1"):
    return ToolMessage(content=content, name=name, tool_call_id=tool_call_id)


def _make_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Find hotels")],
        "plan": "1. Search\n2. Filter",
        "search_query": "hotels SF",
        "raw_hotel_results": [],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 0,
        "error": None,
    }
    state.update(overrides)
    return state


SAMPLE_HOTEL = {
    "name": "Test Hotel",
    "price_per_night": 150.0,
    "distance_to_fishermans_wharf_miles": 0.5,
    "rating": 4.0,
    "availability": True,
    "booking_url": "https://example.com/test",
}


class TestParseHotelResults:
    def test_extracts_hotels_from_hotel_search_message(self):
        msgs = [_tool_msg(json.dumps({"hotels": [SAMPLE_HOTEL]}))]
        results = _parse_hotel_results(msgs)
        assert len(results) == 1
        assert results[0]["name"] == "Test Hotel"

    def test_extracts_multiple_hotels_from_single_message(self):
        hotels = [SAMPLE_HOTEL, {**SAMPLE_HOTEL, "name": "Other Hotel"}]
        msgs = [_tool_msg(json.dumps({"hotels": hotels}))]
        results = _parse_hotel_results(msgs)
        assert len(results) == 2

    def test_aggregates_across_multiple_tool_messages(self):
        msg1 = _tool_msg(json.dumps({"hotels": [SAMPLE_HOTEL]}), tool_call_id="c1")
        msg2 = _tool_msg(
            json.dumps({"hotels": [{**SAMPLE_HOTEL, "name": "Hotel B"}]}),
            tool_call_id="c2",
        )
        results = _parse_hotel_results([msg1, msg2])
        assert len(results) == 2

    def test_ignores_non_hotel_search_tool_messages(self):
        msgs = [_tool_msg(json.dumps({"lat": 37.8}), name="geo_lookup")]
        results = _parse_hotel_results(msgs)
        assert results == []

    def test_ignores_non_tool_messages(self):
        msgs = [AIMessage(content="Thought: I should search"), HumanMessage(content="find hotels")]
        results = _parse_hotel_results(msgs)
        assert results == []

    def test_malformed_json_logs_warning_and_returns_empty(self, caplog):
        msgs = [_tool_msg("not valid json")]
        with caplog.at_level(logging.WARNING):
            results = _parse_hotel_results(msgs)
        assert results == []
        assert any("unparseable" in r.message for r in caplog.records)

    def test_empty_hotels_list_in_response(self):
        msgs = [_tool_msg(json.dumps({"hotels": []}))]
        results = _parse_hotel_results(msgs)
        assert results == []

    def test_empty_message_list(self):
        assert _parse_hotel_results([]) == []


class TestReactAgentNodeHappyPath:
    def _mock_executor_result(self, hotels=None):
        hotels = hotels or [SAMPLE_HOTEL]
        tool_msg = _tool_msg(json.dumps({"hotels": hotels}))
        return {"messages": [AIMessage(content="Done"), tool_msg]}

    def test_returns_hotel_results(self):
        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            mock_exec.invoke.return_value = self._mock_executor_result()
            result = react_agent_node(_make_state())
        assert len(result["raw_hotel_results"]) == 1
        assert result["raw_hotel_results"][0]["name"] == "Test Hotel"

    def test_increments_react_iterations(self):
        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            mock_exec.invoke.return_value = self._mock_executor_result()
            result = react_agent_node(_make_state(react_iterations=2))
        assert result["react_iterations"] == 3

    def test_updates_messages_from_executor(self):
        executor_msgs = [AIMessage(content="I searched"), _tool_msg(json.dumps({"hotels": []}))]
        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            mock_exec.invoke.return_value = {"messages": executor_msgs}
            result = react_agent_node(_make_state())
        assert result["messages"] == executor_msgs

    def test_injects_plan_and_query_into_executor_call(self):
        captured = {}

        def capture(inp):
            captured["messages"] = inp["messages"]
            return {"messages": []}

        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            mock_exec.invoke.side_effect = capture
            react_agent_node(_make_state(plan="my plan", search_query="my query"))

        injected = captured["messages"][-1].content
        assert "my plan" in injected
        assert "my query" in injected


class TestReactAgentNodeErrors:
    def test_returns_error_when_iterations_at_max(self):
        result = react_agent_node(_make_state(react_iterations=MAX_ITERATIONS))
        assert result["error"] == "Max ReAct iterations reached"

    def test_returns_error_when_iterations_exceed_max(self):
        result = react_agent_node(_make_state(react_iterations=MAX_ITERATIONS + 1))
        assert result["error"] is not None

    def test_does_not_call_executor_when_max_reached(self):
        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            react_agent_node(_make_state(react_iterations=MAX_ITERATIONS))
        mock_exec.invoke.assert_not_called()

    def test_executor_exception_propagates(self):
        with patch("graph.nodes.react_agent.react_executor") as mock_exec:
            mock_exec.invoke.side_effect = RuntimeError("network error")
            with pytest.raises(RuntimeError, match="network error"):
                react_agent_node(_make_state())
