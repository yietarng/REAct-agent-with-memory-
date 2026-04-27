from graph.state import AgentState, HotelResult

DISTANCE_LIMIT = 2.0


def ranker_node(state: AgentState) -> AgentState:
    candidates: list[HotelResult] = state.get("raw_hotel_results", [])

    filtered = [
        h for h in candidates
        if h.get("availability") is True
        and h.get("distance_to_fishermans_wharf_miles", 999) <= DISTANCE_LIMIT
    ]

    if not filtered:
        return {
            **state,
            "error": "No available hotels found within 2 miles of Fisherman's Wharf",
            "top_3_hotels": [],
        }

    sorted_hotels = sorted(filtered, key=lambda h: h["price_per_night"])
    return {**state, "top_3_hotels": sorted_hotels[:3]}
