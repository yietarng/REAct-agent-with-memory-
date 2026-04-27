from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

DB_PATH = "hotel_agent_memory.db"


def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def get_thread_config(user_id: str, session_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": f"{user_id}:{session_id}"
        }
    }
