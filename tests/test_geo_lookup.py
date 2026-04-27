"""Tests for tools/geo_lookup.py — geo_lookup tool."""
import json
import pytest
from tools.geo_lookup import geo_lookup


class TestGeoLookupHappyPath:
    def test_returns_string(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf San Francisco"})
        assert isinstance(result, str)

    def test_result_is_valid_json(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_returns_lat_and_lng(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        assert "lat" in parsed
        assert "lng" in parsed

    def test_lat_lng_are_numeric(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        assert isinstance(parsed["lat"], float)
        assert isinstance(parsed["lng"], float)

    def test_returns_district_and_city(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        assert "district" in parsed
        assert "city" in parsed

    def test_location_input_reflected_in_response(self):
        query = "Fisherman's Wharf San Francisco"
        result = geo_lookup.invoke({"location": query})
        parsed = json.loads(result)
        assert parsed["location"] == query

    def test_returns_california_state(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        assert parsed.get("state") == "CA"

    def test_san_francisco_coordinates_in_range(self):
        result = geo_lookup.invoke({"location": "Fisherman's Wharf"})
        parsed = json.loads(result)
        # Fisherman's Wharf is roughly 37.8°N, -122.4°W
        assert 37.0 < parsed["lat"] < 38.0
        assert -123.0 < parsed["lng"] < -122.0
