import pytest


@pytest.fixture(autouse=True)
def env_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture
def hotel_a():
    return {
        "name": "The Wharf Inn",
        "price_per_night": 159.0,
        "distance_to_fishermans_wharf_miles": 0.4,
        "rating": 3.8,
        "availability": True,
        "booking_url": "https://example.com/wharfinn",
    }


@pytest.fixture
def hotel_b():
    return {
        "name": "Hotel Zephyr",
        "price_per_night": 199.0,
        "distance_to_fishermans_wharf_miles": 0.2,
        "rating": 4.2,
        "availability": True,
        "booking_url": "https://example.com/zephyr",
    }


@pytest.fixture
def hotel_c():
    return {
        "name": "Pier 2620 Hotel",
        "price_per_night": 245.0,
        "distance_to_fishermans_wharf_miles": 0.3,
        "rating": 4.3,
        "availability": True,
        "booking_url": "https://example.com/pier2620",
    }


@pytest.fixture
def base_state(hotel_a):
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content="Find hotels near Fisherman's Wharf")],
        "plan": "1. Geolocate\n2. Search hotels\n3. Filter\n4. Rank",
        "search_query": "hotels near Fisherman's Wharf SF",
        "raw_hotel_results": [hotel_a],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 0,
        "error": None,
    }
