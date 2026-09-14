"""Local per-session event log for GitHub Copilot hook events."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:  # POSIX advisory locking; absent on Windows
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore

_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]")

# Suffix appended to a session log while a flush holds it (see claim_session).
_CLAIM_SUFFIX = ".flushing"


def default_state_dir() -> Path:
    """Return the configured or per-user default Copilot journal directory."""
    try:
        from traccia import config as sdk_config

        configured = sdk_config.load_config().instrumentation.github_copilot_state_dir
    except Exception:
        configured = None
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".traccia" / "copilot" / "sessions"


def _safe_session_filename(session_id: str) -> str:
    cleaned = _SESSION_ID_RE.sub("_", str(session_id))[:200]
    return f"{cleaned or 'unknown'}.jsonl"


def session_log_path(session_id: str, state_dir: Optional[Path] = None) -> Path:
    """Return the sanitized JSONL journal path for a Copilot session."""
    base = state_dir or default_state_dir()
    return Path(base) / _safe_session_filename(session_id)


def _session_lock_path(session_id: str, state_dir: Optional[Path]) -> Path:
    return session_log_path(session_id, state_dir).with_suffix(".lock")


@contextmanager
def _session_lock(session_id: str, state_dir: Optional[Path]) -> Iterator[None]:
    """Serialize journal append and claim/rename operations for one session."""
    lock_path = _session_lock_path(session_id, state_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                pass
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def append_event(
    session_id: str,
    event_name: str,
    payload: Dict[str, Any],
    state_dir: Optional[Path] = None,
) -> None:
    """Append one event record to this session's log (creates the file/dir if needed)."""
    record = {"event": event_name, "payload": payload, "received_at": time.time()}
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    with _session_lock(session_id, state_dir):
        normal = session_log_path(session_id, state_dir)
        claimed = normal.with_name(normal.name + _CLAIM_SUFFIX)
        path = claimed if claimed.exists() else normal
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def read_events_from_path(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL session log at an explicit path (used for claimed logs)."""
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written last line
    return events


def read_events(session_id: str, state_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read and parse all valid records for a session."""
    return read_events_from_path(session_log_path(session_id, state_dir))


def has_end_event(session_id: str, state_dir: Optional[Path] = None) -> bool:
    """True if this session's log already contains a sessionEnd record."""
    
    path = session_log_path(session_id, state_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or '"sessionEnd"' not in line:
                    continue  # cheap pre-filter before the JSON parse
                try:
                    if json.loads(line).get("event") == "sessionEnd":
                        return True
                except json.JSONDecodeError:
                    continue
    except OSError:
        return False
    return False


def clear_session(session_id: str, state_dir: Optional[Path] = None) -> None:
    path = session_log_path(session_id, state_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def claim_session(session_id: str, state_dir: Optional[Path] = None) -> Optional[Path]:
    """Atomically take exclusive ownership of a session log for flushing."""
    
    with _session_lock(session_id, state_dir):
        src = session_log_path(session_id, state_dir)
        claimed = src.with_name(src.name + _CLAIM_SUFFIX)

        if claimed.exists() or not src.exists():
            return None
        try:
            os.replace(src, claimed)  # atomic on the same filesystem
        except OSError:
            return None
        try:
            if claimed.stat().st_size == 0:
                discard_claim(claimed)
                return None
        except OSError:
            return None
        return claimed


def claim_fingerprint(claimed_path: Path) -> Optional[Tuple[int, int]]:
    """Return size and mtime for detecting late appends to a claimed journal."""
    try:
        stat = Path(claimed_path).stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def claim_unchanged(claimed_path: Path, fingerprint: Optional[Tuple[int, int]]) -> bool:
    """Whether no hook appended to a claimed journal since it was read."""
    return fingerprint is not None and claim_fingerprint(claimed_path) == fingerprint


def list_stale_claims(
    older_than_seconds: float, state_dir: Optional[Path] = None
) -> List[Path]:
    """``.flushing`` markers left behind by a crashed flush, older than the cutoff."""
    base = Path(state_dir) if state_dir else default_state_dir()
    if not base.exists():
        return []
    now = time.time()
    stale: List[Path] = []
    for p in base.glob(f"*{_CLAIM_SUFFIX}"):
        try:
            if now - p.stat().st_mtime >= older_than_seconds:
                stale.append(p)
        except OSError:
            continue
    return sorted(stale)


def discard_claim(claimed_path: Path) -> None:
    """Delete a successfully-flushed claimed log."""
    try:
        Path(claimed_path).unlink(missing_ok=True)
    except OSError:
        pass


def restore_claim(claimed_path: Path) -> None:
    """Return a claimed log to its normal name so it can be retried later."""
    claimed_path = Path(claimed_path)
    if claimed_path.name.endswith(_CLAIM_SUFFIX):
        original = claimed_path.with_name(claimed_path.name[: -len(_CLAIM_SUFFIX)])
        try:
            os.replace(claimed_path, original)
        except OSError:
            pass


def salvage_late_appends(
    claimed_path: Path,
    byte_offset: Optional[int],
    state_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Move records appended to a claimed journal after ``byte_offset`` back to a
    normal (unclaimed) journal, then drop the claim.

    Returns the normal journal path if anything was salvaged, else ``None``.
    """
    claimed_path = Path(claimed_path)
    name = claimed_path.name
    if name.endswith(_CLAIM_SUFFIX):
        name = name[: -len(_CLAIM_SUFFIX)]
    session_id = Path(name).stem
    normal = claimed_path.with_name(name)

    with _session_lock(session_id, state_dir):
        try:
            data = claimed_path.read_bytes()
        except OSError:
            return None
        tail = data[byte_offset:] if byte_offset and byte_offset < len(data) else b""
        salvaged = bool(tail.strip())
        if salvaged:
            fd = os.open(str(normal), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, tail if tail.endswith(b"\n") else tail + b"\n")
            finally:
                os.close(fd)
        try:
            claimed_path.unlink(missing_ok=True)
        except OSError:
            pass
    return normal if salvaged else None


def archive_failed_claim(claimed_path: Path) -> Optional[Path]:
    """Move a claimed log whose export failed into a ``failed/`` sibling dir.

    Keeps the buffered events recoverable (via ``traccia copilot flush
    --retry-failed``) instead of deleting them when the network export didn't
    land. Returns the archived path.
    """
    claimed_path = Path(claimed_path)
    name = claimed_path.name
    if name.endswith(_CLAIM_SUFFIX):
        name = name[: -len(_CLAIM_SUFFIX)]
    failed_dir = claimed_path.parent / "failed"
    try:
        failed_dir.mkdir(parents=True, exist_ok=True)
        dst = failed_dir / f"{Path(name).stem}.{int(time.time())}.jsonl"
        os.replace(claimed_path, dst)
        return dst
    except OSError:
        return None


def list_failed(state_dir: Optional[Path] = None) -> List[Path]:
    base = Path(state_dir) if state_dir else default_state_dir()
    failed_dir = base / "failed"
    if not failed_dir.exists():
        return []
    return sorted(failed_dir.glob("*.jsonl"))


def list_sessions(state_dir: Optional[Path] = None) -> List[str]:
    base = Path(state_dir) if state_dir else default_state_dir()
    if not base.exists():
        return []
    return [p.stem for p in base.glob("*.jsonl")]


def session_age_seconds(session_id: str, state_dir: Optional[Path] = None) -> Optional[float]:
    """Return the age in seconds of a normal session journal."""
    path = session_log_path(session_id, state_dir)
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None
