import json
from langchain_core.tools import tool


@tool
def geo_lookup(location: str) -> str:
    """
    Returns geographic coordinates and district info for a location.

    Args:
        location: Place name e.g. "Fisherman's Wharf San Francisco"
    """
    # Replace with a real geocoding API (e.g. Google Maps, Nominatim)
    return json.dumps({
        "location": location,
        "lat": 37.8080,
        "lng": -122.4177,
        "district": "Fisherman's Wharf",
        "city": "San Francisco",
        "state": "CA",
    })
