from graph.graph_builder import build_graph
from memory.checkpointer import get_thread_config
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def run_hotel_agent(
    user_message: str,
    user_id: str = "user_001",
    session_id: str = "session_001",
):
    graph = build_graph()
    config = get_thread_config(user_id, session_id)

    print(f"\n{'='*60}")
    print("Hotel Research Agent")
    print(f"{'='*60}")
    print(f"User: {user_message}\n")

    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "plan": None,
        "search_query": None,
        "raw_hotel_results": [],
        "top_3_hotels": [],
        "verification_passed": False,
        "verification_notes": None,
        "notification_message": None,
        "react_iterations": 0,
        "error": None,
    }

    for step in graph.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        print(f"[NODE: {node_name.upper()}]")

        if node_name == "planner":
            print(f"  Plan:\n{node_output.get('plan', '')}\n")

        elif node_name == "react_agent":
            itr = node_output.get("react_iterations", 0)
            n = len(node_output.get("raw_hotel_results", []))
            print(f"  Iteration {itr} | Hotels found: {n}\n")

        elif node_name == "ranker":
            top3 = node_output.get("top_3_hotels", [])
            print("  Top 3 Hotels (by price):")
            for i, h in enumerate(top3, 1):
                print(f"    {i}. {h['name']} — ${h['price_per_night']}/night "
                      f"({h['distance_to_fishermans_wharf_miles']} mi)")
            print()

        elif node_name == "verifier":
            passed = node_output.get("verification_passed")
            notes = node_output.get("verification_notes")
            status = "PASSED" if passed else "FAILED"
            print(f"  {status}: {notes}\n")

        elif node_name == "notifier":
            msg = node_output.get("notification_message", "")
            print(f"  Notification to User:\n{'-'*40}\n{msg}\n{'-'*40}\n")

    print("Agent run complete.")


if __name__ == "__main__":
    run_hotel_agent(
        user_message=(
            "Find me the cheapest hotels in San Francisco "
            "close to Fisherman's Wharf. I need 3 options with prices."
        )
    )
