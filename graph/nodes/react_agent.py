import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from graph.state import AgentState
from tools.hotel_search import hotel_search
from tools.geo_lookup import geo_lookup
from tools.availability import availability_check
from prompts.react_prompt import REACT_SYSTEM

MAX_ITERATIONS = 6

tools = [hotel_search, geo_lookup, availability_check]
llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)
react_executor = create_react_agent(llm, tools, state_modifier=REACT_SYSTEM)


def react_agent_node(state: AgentState) -> AgentState:
    if state["react_iterations"] >= MAX_ITERATIONS:
        return {**state, "error": "Max ReAct iterations reached"}

    augmented_messages = state["messages"] + [
        AIMessage(content=f"[Plan]\n{state['plan']}\n\n"
                          f"[Search Query]: {state['search_query']}")
    ]

    result = react_executor.invoke({"messages": augmented_messages})
    hotel_results = _parse_hotel_results(result["messages"])

    return {
        **state,
        "messages": result["messages"],
        "raw_hotel_results": hotel_results,
        "react_iterations": state["react_iterations"] + 1,
    }


def _parse_hotel_results(messages) -> list:
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "hotel_search":
            try:
                data = json.loads(msg.content)
                results.extend(data.get("hotels", []))
            except Exception:
                pass
    return results
