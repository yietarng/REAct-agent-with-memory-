# Hotel Search ReAct Agent

LangGraph agent that finds the 3 cheapest available hotels near Fisherman's Wharf, San Francisco, using a Plan → ReAct → Rank → Verify → Notify pipeline with persistent SQLite memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH STATE MACHINE                     │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐  │
│   │ PLANNER  │───▶│  REACT   │───▶│ RANKER   │───▶│VERIFIER │  │
│   │  Node    │    │  Agent   │    │  Node    │    │  Node   │  │
│   └──────────┘    │  Loop    │    └──────────┘    └────┬────┘  │
│                   └──────────┘                         │       │
│                        │                               │       │
│                   ┌────▼────┐                    ┌────▼────┐   │
│                   │  TOOLS  │                    │NOTIFIER │   │
│                   │ (Search │                    │  Node   │   │
│                   │  APIs)  │                    └─────────┘   │
│                   └─────────┘                                  │
│                                                                 │
│   ════════════════════ MEMORY LAYER ══════════════════════      │
│   [ SqliteSaver checkpoint ]  [ MemorySaver session store ]    │
└─────────────────────────────────────────────────────────────────┘
```

### Node responsibilities

| Node | File | Role |
|------|------|------|
| `planner` | `graph/nodes/planner.py` | Chain-of-Thought plan from raw user request |
| `react_agent` | `graph/nodes/react_agent.py` | ReAct loop: Thought → Action → Observe until results collected |
| `ranker` | `graph/nodes/ranker.py` | Filter (available, ≤ 2 mi) → sort by price → top 3 |
| `verifier` | `graph/nodes/verifier.py` | LLM quality gate; routes back to `react_agent` on failure (max 3 retries) |
| `notifier` | `graph/nodes/notifier.py` | Customer-facing Markdown notification |

### Routing logic

```
planner → react_agent → ranker ──(error)──────────────► END
                                  │
                                  └──► verifier ──(passed)──► notifier → END
                                            │
                                            └──(failed, iterations < 3)──► react_agent
                                            └──(failed, iterations ≥ 3)──► END
```

## File Map

```
hotel_agent/
├── CLAUDE.md
├── main.py                      # Entry point; streams graph execution
├── graph/
│   ├── __init__.py
│   ├── state.py                 # AgentState TypedDict
│   ├── graph_builder.py         # StateGraph wiring + conditional edges
│   └── nodes/
│       ├── planner.py
│       ├── react_agent.py       # create_react_agent wrapper; MAX_ITERATIONS=6
│       ├── ranker.py
│       ├── verifier.py
│       └── notifier.py
├── tools/
│   ├── __init__.py
│   ├── hotel_search.py          # @tool — hotel price search (SerpAPI/mock)
│   ├── geo_lookup.py            # @tool — coordinates + district info
│   └── availability.py          # @tool — real-time room availability
├── memory/
│   ├── __init__.py
│   └── checkpointer.py          # SqliteSaver; thread_id = "user_id:session_id"
├── prompts/
│   ├── planner_prompt.py
│   ├── react_prompt.py
│   ├── verifier_prompt.py
│   └── notifier_prompt.py
├── requirements.txt
└── .env
```

## State Schema (`graph/state.py`)

Key fields in `AgentState` (TypedDict):

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `list[BaseMessage]` | Full conversation history (LangGraph `add_messages` reducer) |
| `plan` | `str \| None` | Numbered search plan from planner |
| `search_query` | `str \| None` | Refined query passed to tools |
| `raw_hotel_results` | `list[HotelResult]` | All hotels returned by ReAct tools |
| `top_3_hotels` | `list[HotelResult]` | Filtered + ranked output from ranker |
| `verification_passed` | `bool` | Verifier gate result |
| `verification_notes` | `str \| None` | Issues list or "All checks passed." |
| `notification_message` | `str \| None` | Final customer-facing message |
| `react_iterations` | `int` | Loop counter — hard stop at 6 |
| `error` | `str \| None` | Short-circuits to END when set |

`HotelResult` fields: `name`, `price_per_night`, `distance_to_fishermans_wharf_miles`, `rating`, `availability`, `booking_url`.

## Memory Model (`memory/checkpointer.py`)

- **Persistence**: `SqliteSaver` writes full graph state after every node to `hotel_agent_memory.db`.
- **Thread isolation**: `thread_id = f"{user_id}:{session_id}"` — one conversation per thread, no cross-user leakage.
- **Resume**: passing the same `thread_id` in config restores prior state for multi-turn conversations.

```python
config = {"configurable": {"thread_id": "user_001:session_001"}}
graph.stream(initial_state, config=config)
```

## Tools (`tools/`)

| Tool | Signature | Returns |
|------|-----------|---------|
| `hotel_search(query: str)` | Free-text hotel query | JSON `{"hotels": [HotelResult, ...]}`  |
| `geo_lookup(location: str)` | Place name | JSON with `lat`, `lng`, `district`, `city` |
| `availability_check(hotel_name, check_in, check_out)` | Hotel + ISO dates | JSON with `available`, `rooms_left` |

All three are `@tool`-decorated LangChain callables. Mock responses are included for development; swap in real API calls (SerpAPI Hotels, Amadeus, etc.) behind the same interface.

## Key Constraints

- `MAX_ITERATIONS = 6` — maximum ReAct loop cycles per `react_agent` invocation (`react_agent.py`)
- Verifier retry cap: 3 — checked via `react_iterations` in `should_retry_or_notify` (`graph_builder.py`)
- Distance filter: `distance_to_fishermans_wharf_miles <= 2.0` (`ranker.py`)
- Price sanity: `$50 < price_per_night < $2000` (verified by LLM in `verifier.py`)
- Model: `claude-opus-4-5`, temperature 0 for planner/react/verifier; 0.3 for notifier

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY, HOTEL_API_KEY, SERP_API_KEY

# 3. Run the agent
python main.py

# 4. Custom query / session
python - <<'EOF'
from main import run_hotel_agent
run_hotel_agent(
    user_message="Cheapest hotels near Fisherman's Wharf for June 1-2",
    user_id="alice",
    session_id="trip_001",
)
EOF
```

`main.py` streams execution and prints each node's output as it runs.

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...
HOTEL_API_KEY=...          # SerpAPI / RapidAPI Hotels / Amadeus
SERP_API_KEY=...           # Fallback web search
```

## Extending the Agent

| Extension | Where to add |
|-----------|--------------|
| Real hotel API | Replace mock in `tools/hotel_search.py` |
| Booking step | New `booking_node` + edge after `notifier` |
| User preferences (price range, stars) | Store in memory; inject into `planner_node` |
| Multi-city support | Parameterize location in `AgentState` and planner prompt |
| Slack / email alerts | Replace `notifier_node` output with messaging API call |
| LangSmith tracing | Set `LANGCHAIN_TRACING_V2=true` in `.env` |

## Graph Builder Notes (`graph/graph_builder.py`)

- Compiled with `checkpointer=get_checkpointer()` so every node transition is persisted.
- `should_retry_or_notify` is the conditional edge after `verifier`; it reads `verification_passed`, `error`, and `react_iterations`.
- `ranker_router` short-circuits to `END` when `error` is set or `top_3_hotels` is empty.
- Use `graph.stream(state, config)` (not `graph.invoke`) in production to observe intermediate steps.
