"""Tests for memory/checkpointer.py — get_checkpointer and get_thread_config."""
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
import memory.checkpointer as cp_module
from memory.checkpointer import get_thread_config


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    original = cp_module._checkpointer
    cp_module._checkpointer = None
    yield
    # Close the connection if one was opened during the test
    if cp_module._checkpointer is not None:
        try:
            cp_module._checkpointer.conn.close()
        except Exception:
            pass
    cp_module._checkpointer = original


class TestGetCheckpointer:
    def test_returns_sqlite_saver_instance(self, tmp_path):
        from langgraph.checkpoint.sqlite import SqliteSaver
        db = str(tmp_path / "test.db")
        with patch.object(cp_module, "DB_PATH", db):
            saver = cp_module.get_checkpointer()
        assert isinstance(saver, SqliteSaver)

    def test_returns_same_instance_on_repeated_calls(self, tmp_path):
        db = str(tmp_path / "test.db")
        with patch.object(cp_module, "DB_PATH", db):
            first = cp_module.get_checkpointer()
            second = cp_module.get_checkpointer()
        assert first is second

    def test_singleton_not_recreated_when_already_set(self, tmp_path):
        mock_saver = MagicMock()
        cp_module._checkpointer = mock_saver
        result = cp_module.get_checkpointer()
        assert result is mock_saver

    def test_db_created_at_configured_path(self, tmp_path):
        db = str(tmp_path / "custom.db")
        with patch.object(cp_module, "DB_PATH", db):
            cp_module.get_checkpointer()
        import os
        assert os.path.exists(db)


class TestGetThreadConfig:
    def test_returns_correct_thread_id(self):
        config = get_thread_config("alice", "trip_001")
        assert config == {"configurable": {"thread_id": "alice:trip_001"}}

    def test_thread_id_format_is_user_colon_session(self):
        config = get_thread_config("bob", "session_42")
        thread_id = config["configurable"]["thread_id"]
        assert thread_id == "bob:session_42"

    def test_different_users_produce_different_thread_ids(self):
        c1 = get_thread_config("alice", "sess")
        c2 = get_thread_config("bob", "sess")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]

    def test_different_sessions_produce_different_thread_ids(self):
        c1 = get_thread_config("user", "session_1")
        c2 = get_thread_config("user", "session_2")
        assert c1["configurable"]["thread_id"] != c2["configurable"]["thread_id"]

    def test_raises_if_user_id_contains_colon(self):
        with pytest.raises(ValueError, match="user_id"):
            get_thread_config("bad:user", "session")

    def test_raises_if_session_id_contains_colon(self):
        with pytest.raises(ValueError, match="session_id"):
            get_thread_config("user", "bad:session")

    def test_raises_if_user_id_has_multiple_colons(self):
        with pytest.raises(ValueError):
            get_thread_config("a:b:c", "session")

    def test_user_and_session_without_colon_accepted(self):
        config = get_thread_config("user_001", "session_001")
        assert "thread_id" in config["configurable"]
