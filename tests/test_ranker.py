"""Tests for graph/nodes/ranker.py — ranker_node.

Patterns demonstrated:
  - @pytest.mark.parametrize for data-driven filtering cases
  - pytest.approx for float boundary comparisons
  - conftest fixtures (hotel_a, hotel_b, hotel_c) injected directly
"""
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


# ---------------------------------------------------------------------------
# Happy path — uses conftest fixtures directly
# ---------------------------------------------------------------------------

class TestRankerNodeHappyPath:
    def test_returns_top_3_by_price_ascending(self):
        hotels = [
            _hotel(name="Expensive", price_per_night=500.0),
            _hotel(name="Cheap",     price_per_night=100.0),
            _hotel(name="Mid",       price_per_night=250.0),
            _hotel(name="Cheaper",   price_per_night=150.0),
        ]
        result = ranker_node(_make_state(hotels))
        prices = [h["price_per_night"] for h in result["top_3_hotels"]]
        assert prices == sorted(prices)
        assert prices[0] == pytest.approx(100.0)

    def test_uses_conftest_hotel_fixtures(self, hotel_a, hotel_b, hotel_c):
        """Verify conftest fixtures flow through to the ranker correctly."""
        result = ranker_node(_make_state([hotel_c, hotel_a, hotel_b]))  # unsorted
        top = result["top_3_hotels"]
        assert len(top) == 3
        assert top[0]["name"] == hotel_a["name"]  # cheapest first ($159)
        assert top[1]["name"] == hotel_b["name"]  # $199
        assert top[2]["name"] == hotel_c["name"]  # $245

    def test_returns_all_when_fewer_than_3_qualify(self):
        hotels = [_hotel(name="A", price_per_night=100.0), _hotel(name="B", price_per_night=200.0)]
        result = ranker_node(_make_state(hotels))
        assert len(result["top_3_hotels"]) == 2

    def test_exactly_3_hotels_returns_all_3(self):
        hotels = [_hotel(name=f"H{i}", price_per_night=float(i * 100)) for i in range(1, 4)]
        result = ranker_node(_make_state(hotels))
        assert len(result["top_3_hotels"]) == 3

    def test_no_error_field_when_hotels_found(self):
        result = ranker_node(_make_state([_hotel()]))
        assert result.get("error") is None

    def test_preserves_other_state_fields(self):
        result = ranker_node(_make_state([_hotel()], react_iterations=3))
        assert result["react_iterations"] == 3


# ---------------------------------------------------------------------------
# Filtering — parametrized over reasons a hotel should be excluded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides,reason", [
    ({"availability": False},                                         "unavailable"),
    ({"distance_to_fishermans_wharf_miles": DISTANCE_LIMIT + 0.01},  "just over distance limit"),
    ({"distance_to_fishermans_wharf_miles": 10.0},                   "far away"),
])
def test_hotel_excluded_from_results(overrides, reason):
    """Hotels that fail availability or distance constraints must not appear in top_3."""
    cheap_invalid = _hotel(name="Invalid", price_per_night=50.0, **overrides)
    valid = _hotel(name="Valid", price_per_night=300.0)
    result = ranker_node(_make_state([cheap_invalid, valid]))
    names = [h["name"] for h in result["top_3_hotels"]]
    assert "Invalid" not in names, f"Hotel should be excluded because: {reason}"
    assert "Valid" in names


@pytest.mark.parametrize("distance", [0.0, 0.5, DISTANCE_LIMIT])
def test_hotels_within_distance_are_included(distance):
    """Hotels at or within the distance limit must pass the filter."""
    hotel = _hotel(distance_to_fishermans_wharf_miles=distance)
    result = ranker_node(_make_state([hotel]))
    assert len(result["top_3_hotels"]) == 1
    assert result["top_3_hotels"][0]["distance_to_fishermans_wharf_miles"] == pytest.approx(distance)


# ---------------------------------------------------------------------------
# Distance boundary precision
# ---------------------------------------------------------------------------

class TestRankerDistanceBoundary:
    def test_includes_hotel_at_exact_distance_limit(self):
        hotel = _hotel(distance_to_fishermans_wharf_miles=DISTANCE_LIMIT)
        result = ranker_node(_make_state([hotel]))
        assert result["top_3_hotels"][0]["name"] == hotel["name"]

    def test_excludes_hotel_just_over_distance_limit(self):
        hotel = _hotel(distance_to_fishermans_wharf_miles=DISTANCE_LIMIT + 0.001)
        result = ranker_node(_make_state([hotel]))
        assert result["top_3_hotels"] == []
        assert result["error"] is not None

    def test_missing_distance_field_treated_as_out_of_range(self):
        hotel = {k: v for k, v in _hotel().items() if k != "distance_to_fishermans_wharf_miles"}
        result = ranker_node(_make_state([hotel]))
        assert result["top_3_hotels"] == []
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

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

    def test_missing_price_key_raises(self):
        """Ranker sorts on price_per_night; a missing key must raise KeyError."""
        hotel = {k: v for k, v in _hotel().items() if k != "price_per_night"}
        with pytest.raises(KeyError):
            ranker_node(_make_state([hotel]))
