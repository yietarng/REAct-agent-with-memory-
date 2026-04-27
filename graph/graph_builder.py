from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.planner import planner_node
from graph.nodes.react_agent import react_agent_node
from graph.nodes.ranker import ranker_node
from graph.nodes.verifier import verifier_node
from graph.nodes.notifier import notifier_node
from memory.checkpointer import get_checkpointer

VERIFIER_RETRY_CAP = 3


def should_retry_or_notify(state: AgentState) -> str:
    if state.get("error"):
        return "end_with_error"
    if state.get("verification_passed"):
        return "notifier"
    if state.get("react_iterations", 0) >= VERIFIER_RETRY_CAP:
        return "end_with_error"
    return "react_agent"


def ranker_router(state: AgentState) -> str:
    if state.get("error") or not state.get("top_3_hotels"):
        return "end_with_error"
    return "verifier"


def build_graph():
    checkpointer = get_checkpointer()
    builder = StateGraph(AgentState)

    builder.add_node("planner",     planner_node)
    builder.add_node("react_agent", react_agent_node)
    builder.add_node("ranker",      ranker_node)
    builder.add_node("verifier",    verifier_node)
    builder.add_node("notifier",    notifier_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner",     "react_agent")
    builder.add_edge("react_agent", "ranker")
    builder.add_edge("notifier",    END)

    builder.add_conditional_edges(
        "ranker",
        ranker_router,
        {"verifier": "verifier", "end_with_error": END}
    )
    builder.add_conditional_edges(
        "verifier",
        should_retry_or_notify,
        {
            "notifier":       "notifier",
            "react_agent":    "react_agent",
            "end_with_error": END,
        }
    )

    return builder.compile(checkpointer=checkpointer)
