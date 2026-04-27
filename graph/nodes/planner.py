from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from prompts.planner_prompt import PLANNER_SYSTEM

llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)


def planner_node(state: AgentState) -> AgentState:
    if not state.get("messages"):
        raise ValueError("planner_node requires at least one message in state")
    user_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"User request: {user_message}")
    ])

    return {
        **state,
        "plan": response.content,
        "search_query": "hotels San Francisco near Fisherman's Wharf price per night",
        "react_iterations": 0,
    }
