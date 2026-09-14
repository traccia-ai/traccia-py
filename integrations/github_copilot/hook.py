"""Entry point invoked by GitHub Copilot's hooks for the Traccia integration."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional


def _run(argv: List[str]) -> None:
    event_name = argv[1] if len(argv) > 1 else None
    if not event_name:
        return

    from traccia import config as sdk_config

    try:
        cfg = sdk_config.load_config()
        enabled = cfg.instrumentation.github_copilot
        capture_content = cfg.instrumentation.github_copilot_capture_content
    except Exception:
        enabled = True
        capture_content = False

    if not enabled:
        return

    raw = sys.stdin.read()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return

    session_id = payload.get("sessionId")
    if not session_id:
        return  # nothing to correlate this event to

    from traccia.integrations.github_copilot import mapping, state

    if event_name not in mapping.ALL_KNOWN_EVENTS:
        return  # forward-compatible: silently ignore events this version doesn't map yet

    payload = mapping.strip_content_fields(event_name, payload, capture_content)
    state.append_event(session_id, event_name, payload)

    if event_name in mapping.SESSION_END_EVENTS:
        if os.environ.get("TRACCIA_GITHUB_COPILOT_SYNC_FLUSH") == "1":
            _flush_sync(session_id)
        else:
            _spawn_flush(session_id)


def _flush_sync(session_id: str) -> None:
    """Export a completed session before this hook process exits."""
    try:
        from traccia.integrations.github_copilot.flush import flush_session

        flush_session(session_id)
    except BaseException:
        pass


def _spawn_flush(session_id: str) -> None:
    """Kick off export in a detached background process so the network call
    never blocks this hook (which Copilot is waiting on)."""
    import subprocess

    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "traccia.integrations.github_copilot.flush",
                "--session",
                session_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass  # an orphaned session is still recoverable via `traccia copilot flush --all`


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv) if argv is not None else list(sys.argv)
    try:
        _run(argv)
    except BaseException:
        pass
    try:
        sys.stdout.write("{}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
