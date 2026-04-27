"""Integration tests — state flowing through multiple nodes in sequence.

Patterns demonstrated:
  - Multi-node pipelines without mocking intermediate state
  - conftest fixtures (hotel_a, hotel_b, hotel_c, base_state) shared across tests
  - @pytest.mark.parametrize across routing scenarios
  - module-scoped fixture (human_message_cls) to avoid repeated imports
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from graph.nodes.ranker import ranker_node
from graph.nodes.verifier import verifier_node
from graph.nodes.notifier import notifier_node
from graph.graph_builder import ranker_router, should_retry_or_notify


# ---------------------------------------------------------------------------
# ranker → router: end-to-end filtering + routing decision
# ---------------------------------------------------------------------------

class TestRankerIntoRouter:
    def test_valid_hotels_route_to_verifier(self, hotel_a, hotel_b, hotel_c):
        state = {
            "messages": [],
            "plan": None,
            "search_query": None,
            "raw_hotel_results": [hotel_a, hotel_b, hotel_c],
            "top_3_hotels": [],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 1,
            "error": None,
        }
        ranked = ranker_node(state)
        assert ranked["error"] is None
        assert ranker_router(ranked) == "verifier"

    def test_no_valid_hotels_route_to_end(self):
        state = {
            "messages": [],
            "plan": None,
            "search_query": None,
            "raw_hotel_results": [],   # empty — nothing passes the filter
            "top_3_hotels": [],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 1,
            "error": None,
        }
        ranked = ranker_node(state)
        assert ranker_router(ranked) == "end_with_error"

    def test_ranked_hotels_ordered_by_price(self, hotel_a, hotel_b, hotel_c):
        state = {
            "messages": [],
            "plan": None,
            "search_query": None,
            "raw_hotel_results": [hotel_c, hotel_a, hotel_b],  # shuffled
            "top_3_hotels": [],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 1,
            "error": None,
        }
        ranked = ranker_node(state)
        prices = [h["price_per_night"] for h in ranked["top_3_hotels"]]
        assert prices == sorted(prices)


# ---------------------------------------------------------------------------
# ranker → verifier → router: quality gate decides next hop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("llm_passed,iterations,expected_route", [
    (True,  1, "notifier"),          # passed → proceed
    (False, 1, "react_agent"),       # failed, retries left → retry
    (False, 3, "end_with_error"),    # failed, cap hit → stop
])
def test_verifier_routing(hotel_a, hotel_b, hotel_c, llm_passed, iterations, expected_route):
    """Full ranker → verifier pipeline; router sees the verifier output."""
    ranker_state = {
        "messages": [],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": [hotel_a, hotel_b, hotel_c],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": iterations,
        "error": None,
    }
    ranked = ranker_node(ranker_state)

    with patch("graph.nodes.verifier.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"passed": llm_passed, "issues": [] if llm_passed else ["price too high"]})
        )
        verified = verifier_node(ranked)

    assert should_retry_or_notify(verified) == expected_route


# ---------------------------------------------------------------------------
# verifier → notifier: message appended to state
# ---------------------------------------------------------------------------

class TestVerifierIntoNotifier:
    def test_notification_added_after_passing_verification(self, hotel_a, hotel_b, hotel_c, human_message_cls):
        from langchain_core.messages import AIMessage

        state = {
            "messages": [human_message_cls(content="Find hotels")],
            "plan": None,
            "search_query": None,
            "raw_hotel_results": [],
            "top_3_hotels": [hotel_a, hotel_b, hotel_c],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 1,
            "error": None,
        }

        with patch("graph.nodes.verifier.llm") as mock_verifier_llm:
            mock_verifier_llm.invoke.return_value = MagicMock(
                content=json.dumps({"passed": True, "issues": []})
            )
            verified = verifier_node(state)

        assert verified["verification_passed"] is True

        with patch("graph.nodes.notifier.llm") as mock_notifier_llm:
            mock_notifier_llm.invoke.return_value = MagicMock(content="Here are your 3 hotels!")
            notified = notifier_node(verified)

        assert notified["notification_message"] == "Here are your 3 hotels!"
        assert isinstance(notified["messages"][-1], AIMessage)
        assert notified["messages"][-1].content == "Here are your 3 hotels!"

    def test_state_fields_preserved_across_pipeline(self, hotel_a, hotel_b, hotel_c, human_message_cls):
        """react_iterations and plan must survive the ranker → verifier → notifier chain."""
        state = {
            "messages": [human_message_cls(content="Find hotels")],
            "plan": "original plan",
            "search_query": "hotels SF",
            "raw_hotel_results": [hotel_a, hotel_b, hotel_c],
            "top_3_hotels": [],
            "verification_passed": False,
            "verification_notes": None,
            "notification_message": None,
            "react_iterations": 2,
            "error": None,
        }

        ranked = ranker_node(state)

        with patch("graph.nodes.verifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(
                content=json.dumps({"passed": True, "issues": []})
            )
            verified = verifier_node(ranked)

        with patch("graph.nodes.notifier.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="notification")
            notified = notifier_node(verified)

        assert notified["react_iterations"] == 2
        assert notified["plan"] == "original plan"
        assert notified["search_query"] == "hotels SF"
