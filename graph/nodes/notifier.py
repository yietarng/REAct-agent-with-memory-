import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from graph.state import AgentState
from prompts.notifier_prompt import NOTIFIER_SYSTEM

llm = ChatAnthropic(model="claude-opus-4-5", temperature=0.3)


def notifier_node(state: AgentState) -> AgentState:
    top_3 = state.get("top_3_hotels", [])

    response = llm.invoke([
        SystemMessage(content=NOTIFIER_SYSTEM),
        HumanMessage(content=json.dumps(top_3, indent=2))
    ])

    notification = response.content

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=notification)],
        "notification_message": notification,
    }
