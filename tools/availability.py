import json
from langchain_core.tools import tool


@tool
def availability_check(hotel_name: str, check_in: str, check_out: str) -> str:
    """
    Checks real-time availability for a specific hotel.

    Args:
        hotel_name: Full hotel name
        check_in: ISO date string e.g. '2025-06-01'
        check_out: ISO date string e.g. '2025-06-02'
    """
    # Replace with a real availability API (e.g. Amadeus, Booking.com)
    return json.dumps({
        "hotel_name": hotel_name,
        "available": True,
        "rooms_left": 4,
        "check_in": check_in,
        "check_out": check_out,
    })
