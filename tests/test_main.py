"""Tests for main.py — run_hotel_agent entry point."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage


def _make_stream_steps(include_notifier=True):
    steps = [
        {"planner": {
            "plan": "1. Search\n2. Filter\n3. Rank",
            "search_query": "hotels SF",
            "messages": [],
            "react_iterations": 0,
        }},
        {"react_agent": {
            "react_iterations": 1,
            "raw_hotel_results": [{"name": "Test Hotel", "price_per_night": 150.0}],
            "messages": [],
        }},
        {"ranker": {
            "top_3_hotels": [{"name": "Test Hotel", "price_per_night": 150.0,
                              "distance_to_fishermans_wharf_miles": 0.4}],
            "messages": [],
        }},
        {"verifier": {
            "verification_passed": True,
            "verification_notes": "All checks passed.",
            "messages": [],
        }},
    ]
    if include_notifier:
        steps.append({"notifier": {
            "notification_message": "Here are your top 3 hotels!",
            "messages": [],
        }})
    return iter(steps)


class TestRunHotelAgentHappyPath:
    def test_completes_without_exception(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels", user_id="u1", session_id="s1")

    def test_prints_header(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        out = capsys.readouterr().out
        assert "Hotel Research Agent" in out

    def test_prints_completion_message(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "Agent run complete." in capsys.readouterr().out

    def test_prints_planner_output(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "1. Search" in capsys.readouterr().out

    def test_prints_ranker_hotels(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "Test Hotel" in capsys.readouterr().out

    def test_prints_verifier_status(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "PASSED" in capsys.readouterr().out

    def test_prints_notification(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "Here are your top 3 hotels!" in capsys.readouterr().out

    def test_passes_human_message_in_initial_state(self):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        captured = {}

        def capture_stream(state, config):
            captured["state"] = state
            return _make_stream_steps()

        mock_graph.stream.side_effect = capture_stream
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Cheapest hotels near the Wharf")

        msgs = captured["state"]["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Cheapest hotels near the Wharf"

    def test_uses_thread_config_with_user_and_session(self):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = _make_stream_steps()
        captured_config = {}

        def capture_stream(state, config):
            captured_config.update(config)
            return _make_stream_steps()

        mock_graph.stream.side_effect = capture_stream
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels", user_id="alice", session_id="trip_1")

        thread_id = captured_config["configurable"]["thread_id"]
        assert thread_id == "alice:trip_1"

    def test_default_user_and_session_ids(self):
        mock_graph = MagicMock()
        captured_config = {}

        def capture_stream(state, config):
            captured_config.update(config)
            return _make_stream_steps()

        mock_graph.stream.side_effect = capture_stream
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")

        thread_id = captured_config["configurable"]["thread_id"]
        assert "user_001" in thread_id
        assert "session_001" in thread_id


class TestRunHotelAgentEdgeCases:
    def test_handles_error_node_gracefully(self, capsys):
        mock_graph = MagicMock()
        # Stream that ends early with no notifier (error path)
        mock_graph.stream.return_value = _make_stream_steps(include_notifier=False)
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "Agent run complete." in capsys.readouterr().out

    def test_verifier_failed_prints_failed_status(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([
            {"verifier": {
                "verification_passed": False,
                "verification_notes": "price too high",
                "messages": [],
            }}
        ])
        with patch("main.build_graph", return_value=mock_graph):
            from main import run_hotel_agent
            run_hotel_agent("Find hotels")
        assert "FAILED" in capsys.readouterr().out
