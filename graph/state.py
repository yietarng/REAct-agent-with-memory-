from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class HotelResult(TypedDict):
    name: str
    price_per_night: float
    distance_to_fishermans_wharf_miles: float
    rating: float
    availability: bool
    booking_url: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: Optional[str]
    search_query: Optional[str]
    raw_hotel_results: list[HotelResult]
    top_3_hotels: list[HotelResult]
    verification_passed: bool
    verification_notes: Optional[str]
    notification_message: Optional[str]
    react_iterations: int
    error: Optional[str]
