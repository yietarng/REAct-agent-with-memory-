"""Tests for tools/hotel_search.py — hotel_search tool."""
import json
import pytest
from tools.hotel_search import hotel_search

REQUIRED_HOTEL_FIELDS = {
    "name", "price_per_night", "distance_to_fishermans_wharf_miles",
    "rating", "availability", "booking_url",
}


class TestHotelSearchHappyPath:
    def test_returns_string(self):
        result = hotel_search.invoke({"query": "hotels near Fisherman's Wharf"})
        assert isinstance(result, str)

    def test_result_is_valid_json(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_result_has_hotels_key(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        parsed = json.loads(result)
        assert "hotels" in parsed

    def test_hotels_is_a_list(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        assert isinstance(hotels, list)

    def test_hotels_list_is_non_empty(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        assert len(hotels) > 0

    def test_each_hotel_has_required_fields(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        for hotel in hotels:
            missing = REQUIRED_HOTEL_FIELDS - set(hotel.keys())
            assert not missing, f"Hotel missing fields: {missing}"

    def test_price_per_night_is_numeric(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        for hotel in hotels:
            assert isinstance(hotel["price_per_night"], (int, float))
            assert hotel["price_per_night"] > 0

    def test_availability_is_boolean(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        for hotel in hotels:
            assert isinstance(hotel["availability"], bool)

    def test_distance_is_non_negative(self):
        result = hotel_search.invoke({"query": "hotels SF"})
        hotels = json.loads(result)["hotels"]
        for hotel in hotels:
            assert hotel["distance_to_fishermans_wharf_miles"] >= 0

    def test_result_is_same_for_any_query(self):
        r1 = hotel_search.invoke({"query": "query one"})
        r2 = hotel_search.invoke({"query": "query two"})
        assert json.loads(r1)["hotels"] == json.loads(r2)["hotels"]
