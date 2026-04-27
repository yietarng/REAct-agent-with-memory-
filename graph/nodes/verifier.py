import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState
from prompts.verifier_prompt import VERIFIER_SYSTEM

llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)


def verifier_node(state: AgentState) -> AgentState:
    top_3 = state.get("top_3_hotels", [])

    response = llm.invoke([
        SystemMessage(content=VERIFIER_SYSTEM),
        HumanMessage(content=json.dumps(top_3, indent=2))
    ])

    try:
        result = json.loads(response.content)
        passed = result.get("passed", False)
        issues = result.get("issues", [])
    except Exception:
        passed = False
        issues = ["Verifier returned malformed response"]

    notes = "; ".join(issues) if issues else "All checks passed."

    return {
        **state,
        "verification_passed": passed,
        "verification_notes": notes,
    }
