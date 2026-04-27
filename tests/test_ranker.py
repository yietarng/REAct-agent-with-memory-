"""Tests for graph/nodes/ranker.py — ranker_node."""
import pytest
from graph.nodes.ranker import ranker_node, DISTANCE_LIMIT


def _hotel(**overrides):
    base = {
        "name": "Default Hotel",
        "price_per_night": 200.0,
        "distance_to_fishermans_wharf_miles": 0.5,
        "rating": 4.0,
        "availability": True,
        "booking_url": "https://example.com/default",
    }
    base.update(overrides)
    return base


def _make_state(hotels=None, **overrides):
    state = {
        "messages": [],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": hotels or [],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 1,
        "error": None,
    }
    state.update(overrides)
    return state


class TestRankerNodeHappyPath:
    def test_returns_top_3_by_price_ascending(self):
        hotels = [
            _hotel(name="Expensive", price_per_night=500.0),
            _hotel(name="Cheap", price_per_night=100.0),
            _hotel(name="Mid", price_per_night=250.0),
            _hotel(name="Cheaper", price_per_night=150.0),
        ]
        result = ranker_node(_make_state(hotels))
        top = result["top_3_hotels"]
        assert len(top) == 3
        prices = [h["price_per_night"] for h in top]
        assert prices == sorted(prices)
        assert prices[0] == 100.0

    def test_returns_all_when_fewer_than_3_qualify(self):
        hotels = [_hotel(name="A", price_per_night=100.0), _hotel(name="B", price_per_night=200.0)]
        result = ranker_node(_make_state(hotels))
        assert len(result["top_3_hotels"]) == 2

    def test_exactly_3_hotels_returns_all_3(self):
        hotels = [_hotel(name=f"H{i}", price_per_night=float(i * 100)) for i in range(1, 4)]
        result = ranker_node(_make_state(hotels))
        assert len(result["top_3_hotels"]) == 3

    def test_no_error_field_when_hotels_found(self):
        hotels = [_hotel()]
        result = ranker_node(_make_state(hotels))
        assert result.get("error") is None

    def test_preserves_other_state_fields(self):
        hotels = [_hotel()]
        result = ranker_node(_make_state(hotels, react_iterations=3))
        assert result["react_iterations"] == 3


class TestRankerNodeFiltering:
    def test_excludes_unavailable_hotels(self):
        hotels = [
            _hotel(name="Available", price_per_night=100.0, availability=True),
            _hotel(name="Unavailable", price_per_night=50.0, availability=False),
        ]
        result = ranker_node(_make_state(hotels))
        names = [h["name"] for h in result["top_3_hotels"]]
        assert "Available" in names
        assert "Unavailable" not in names

    def test_excludes_hotels_beyond_distance_limit(self):
        hotels = [
            _hotel(name="Near", price_per_night=100.0, distance_to_fishermans_wharf_miles=1.0),
            _hotel(name="Far", price_per_night=50.0, distance_to_fishermans_wharf_miles=3.0),
        ]
        result = ranker_node(_make_state(hotels))
        names = [h["name"] for h in result["top_3_hotels"]]
        assert "Near" in names
        assert "Far" not in names

    def test_includes_hotel_at_exact_distance_limit(self):
        hotels = [_hotel(name="Exact", distance_to_fishermans_wharf_miles=DISTANCE_LIMIT)]
        result = ranker_node(_make_state(hotels))
        assert result["top_3_hotels"][0]["name"] == "Exact"

    def test_excludes_hotel_just_over_distance_limit(self):
        hotels = [_hotel(name="Over", distance_to_fishermans_wharf_miles=DISTANCE_LIMIT + 0.01)]
        result = ranker_node(_make_state(hotels))
        assert result["top_3_hotels"] == []
        assert result["error"] is not None

    def test_missing_distance_field_treated_as_out_of_range(self):
        hotel = {
            "name": "No Distance",
            "price_per_night": 100.0,
            "rating": 4.0,
            "availability": True,
            "booking_url": "https://example.com",
        }
        result = ranker_node(_make_state([hotel]))
        assert result["top_3_hotels"] == []
        assert result["error"] is not None


class TestRankerNodeErrors:
    def test_empty_input_sets_error(self):
        result = ranker_node(_make_state([]))
        assert result["error"] is not None
        assert result["top_3_hotels"] == []

    def test_all_filtered_out_sets_error(self):
        hotels = [
            _hotel(availability=False),
            _hotel(distance_to_fishermans_wharf_miles=5.0),
        ]
        result = ranker_node(_make_state(hotels))
        assert result["error"] is not None
        assert result["top_3_hotels"] == []

    def test_error_message_mentions_fishermans_wharf(self):
        result = ranker_node(_make_state([]))
        assert "Fisherman's Wharf" in result["error"]
