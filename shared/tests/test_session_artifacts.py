from __future__ import annotations

from slicer_agent_engine.session import SessionManager


def test_session_initializes_append_only_artifacts(tmp_path) -> None:
    session = SessionManager(out_dir=tmp_path)

    assert session.trace_path.is_file()
    assert session.trace_path.read_text(encoding="utf-8") == ""
    assert session.evidence_path.is_file()
    assert session.evidence_path.read_text(encoding="utf-8") == ""
    assert session.runlog_path.is_file()
    assert '"event_type": "session_created"' in session.runlog_path.read_text(encoding="utf-8")
