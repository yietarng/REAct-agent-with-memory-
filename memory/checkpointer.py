from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

# Anchor the DB next to this file so the path is stable regardless of cwd.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hotel_agent_memory.db")

_checkpointer: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    global _checkpointer
    if _checkpointer is None:
        conn = sqlite3.connect(os.path.normpath(DB_PATH), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def get_thread_config(user_id: str, session_id: str) -> dict:
    # Reject IDs containing ":" to prevent thread_id collisions between users.
    if ":" in user_id:
        raise ValueError(f"user_id must not contain ':'; got {user_id!r}")
    if ":" in session_id:
        raise ValueError(f"session_id must not contain ':'; got {session_id!r}")
    return {
        "configurable": {
            "thread_id": f"{user_id}:{session_id}"
        }
    }
