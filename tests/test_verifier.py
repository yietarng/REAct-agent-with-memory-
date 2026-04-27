"""Tests for graph/nodes/verifier.py — verifier_node."""
import json
import pytest
from unittest.mock import patch, MagicMock
from graph.nodes.verifier import verifier_node


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


def _make_state(top_3=None, **overrides):
    state = {
        "messages": [],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": [],
        "top_3_hotels": top_3 or [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 1,
        "error": None,
    }
    state.update(overrides)
    return state


def _llm_response(passed: bool, issues: list):
    return MagicMock(content=json.dumps({"passed": passed, "issues": issues}))


class TestVerifierNodeHappyPath:
    def test_sets_verification_passed_true(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(True, [])
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_passed"] is True

    def test_notes_all_checks_passed_when_no_issues(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(True, [])
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_notes"] == "All checks passed."

    def test_sets_verification_failed(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(False, ["price too high"])
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_passed"] is False

    def test_single_issue_in_notes(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(False, ["price too high"])
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_notes"] == "price too high"

    def test_multiple_issues_joined_with_semicolon(self):
        issues = ["price too high", "distance > 2 miles", "duplicate names"]
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(False, issues)
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_notes"] == "price too high; distance > 2 miles; duplicate names"

    def test_preserves_other_state_fields(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(True, [])
            result = verifier_node(_make_state([_hotel()], react_iterations=2))
        assert result["react_iterations"] == 2


class TestVerifierNodeSanitisation:
    def test_long_hotel_name_truncated_before_llm(self):
        """Verifier must truncate strings to block prompt injection."""
        long_name = "A" * 600
        captured = {}

        def capture_invoke(msgs):
            captured["msgs"] = msgs
            return _llm_response(True, [])

        hotel = _hotel(name=long_name)
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture_invoke
            verifier_node(_make_state([hotel]))

        # The HumanMessage content passed to the LLM must contain a truncated name
        human_msg_content = captured["msgs"][1].content
        sent_hotels = json.loads(human_msg_content)
        assert len(sent_hotels[0]["name"]) == 500

    def test_booking_url_truncated_to_500_chars(self):
        long_url = "https://example.com/" + "x" * 600
        captured = {}

        def capture_invoke(msgs):
            captured["msgs"] = msgs
            return _llm_response(True, [])

        hotel = _hotel(booking_url=long_url)
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture_invoke
            verifier_node(_make_state([hotel]))

        human_msg_content = captured["msgs"][1].content
        sent_hotels = json.loads(human_msg_content)
        assert len(sent_hotels[0]["booking_url"]) == 500


class TestVerifierNodeErrors:
    def test_malformed_json_response_sets_failed(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="not json at all")
            result = verifier_node(_make_state([_hotel()]))
        assert result["verification_passed"] is False

    def test_malformed_json_response_sets_error_note(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="```json\n{bad}```")
            result = verifier_node(_make_state([_hotel()]))
        assert "malformed" in result["verification_notes"].lower()

    def test_empty_top_3_still_calls_llm(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response(False, ["no hotels"])
            verifier_node(_make_state([]))
        mock_llm.invoke.assert_called_once()

    def test_llm_exception_propagates(self):
        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
            with pytest.raises(RuntimeError, match="LLM unavailable"):
                verifier_node(_make_state([_hotel()]))
