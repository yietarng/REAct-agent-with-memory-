import json
from langchain_core.tools import tool


@tool
def hotel_search(query: str) -> str:
    """
    Search for hotels in San Francisco near Fisherman's Wharf.
    Returns JSON list of hotels with name, price, distance, rating,
    availability, and booking URL.

    Args:
        query: Search query e.g. 'hotels near Fishermans Wharf SF'
    """
    # Replace with real API call, e.g.:
    # import httpx
    # params = {
    #     "engine": "google_hotels",
    #     "q": query,
    #     "check_in_date": "2025-06-01",
    #     "check_out_date": "2025-06-02",
    #     "adults": 1,
    #     "currency": "USD",
    #     "api_key": os.getenv("SERP_API_KEY"),
    # }
    # resp = httpx.get("https://serpapi.com/search", params=params)
    # raw = resp.json()

    mock_hotels = [
        {"name": "Argonaut Hotel",          "price_per_night": 289,
         "distance_to_fishermans_wharf_miles": 0.1, "rating": 4.5,
         "availability": True,
         "booking_url": "https://example.com/argonaut"},
        {"name": "Hotel Zephyr",            "price_per_night": 199,
         "distance_to_fishermans_wharf_miles": 0.2, "rating": 4.2,
         "availability": True,
         "booking_url": "https://example.com/zephyr"},
        {"name": "Pier 2620 Hotel",         "price_per_night": 245,
         "distance_to_fishermans_wharf_miles": 0.3, "rating": 4.3,
         "availability": True,
         "booking_url": "https://example.com/pier2620"},
        {"name": "The Wharf Inn",           "price_per_night": 159,
         "distance_to_fishermans_wharf_miles": 0.4, "rating": 3.8,
         "availability": True,
         "booking_url": "https://example.com/wharfinn"},
        {"name": "Hyatt Centric Fisherman", "price_per_night": 312,
         "distance_to_fishermans_wharf_miles": 0.5, "rating": 4.6,
         "availability": False,
         "booking_url": "https://example.com/hyatt"},
    ]

    return json.dumps({"hotels": mock_hotels})
