"""Materialize and export a GitHub Copilot session's buffered hook events."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from traccia.integrations.github_copilot import mapping, state

_FLUSH_TIMEOUT_SECONDS = 15.0
_STALE_CLAIM_SECONDS = 3600.0


def _has_session_end(events: List[Dict[str, Any]]) -> bool:
    return any(e.get("event") in mapping.SESSION_END_EVENTS for e in events)


def _export_events(events: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], bool]:
    """Build spans for one session's events and flush them. Returns (summary, flush_ok)."""
    import traccia
    from traccia import auto as _auto_mod
    from traccia.integrations.github_copilot import spans as spans_mod

    started_here = not bool(getattr(_auto_mod, "_started", False))
    if started_here:
        traccia.init(
            auto_start_trace=False,
            openai_agents=False,
            crewai=False,
            github_copilot=False,
        )
        
        traccia.end_auto_trace()

    tracer = traccia.get_tracer("github_copilot")
    summary = spans_mod.build_trace(tracer, events)

    force_flush_result = traccia.force_flush(_FLUSH_TIMEOUT_SECONDS)
    flush_ok = True if force_flush_result is None else bool(force_flush_result)
    if started_here:
        traccia.stop_tracing()

    return summary, flush_ok


def flush_session(
    session_id: str, *, state_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Claim one session's buffered events, build+export its trace, then clear it.

    Returns the build summary dict, or None if there was nothing buffered or the
    session was already claimed by another flush.
    """
    claimed = state.claim_session(session_id, state_dir=state_dir)
    if claimed is None:
        return None  # nothing to flush, or a concurrent flush owns it

    events = state.read_events_from_path(claimed)
    initial_fingerprint = state.claim_fingerprint(claimed)
    if not events:
        state.discard_claim(claimed)
        return None

    try:
        summary, flush_ok = _export_events(events)
    except BaseException:
        state.restore_claim(claimed)
        raise

    if flush_ok:
        if state.claim_unchanged(claimed, initial_fingerprint):
            state.discard_claim(claimed)
        else:
            salvaged = state.salvage_late_appends(
                claimed,
                initial_fingerprint[0] if initial_fingerprint else None,
                state_dir=state_dir,
            )
            if salvaged is not None:
                print(
                    f"traccia copilot flush: session {session_id} exported; "
                    f"late hook events kept at {salvaged} for a follow-up flush",
                    file=sys.stderr,
                )
    else:
        archived = state.archive_failed_claim(claimed)
        print(
            f"traccia copilot flush: export for session {session_id} did not "
            f"confirm; buffered events kept at {archived or claimed} "
            f"(retry with `traccia copilot flush --retry-failed`)",
            file=sys.stderr,
        )
    return summary


def flush_all(
    *,
    max_age_seconds: Optional[float] = None,
    include_active: bool = False,
    state_dir: Optional[Path] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Flush buffered sessions."""
    
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for session_id in state.list_sessions(state_dir=state_dir):
        ended = state.has_end_event(session_id, state_dir=state_dir)
        old_enough = False
        if max_age_seconds is not None:
            age = state.session_age_seconds(session_id, state_dir=state_dir)
            old_enough = age is not None and age >= max_age_seconds

        if not (ended or include_active or old_enough):
            continue

        results[session_id] = flush_session(session_id, state_dir=state_dir)

    # Reclaim claims stranded by a hard-killed flush so their events aren't lost.
    for stale in state.list_stale_claims(_STALE_CLAIM_SECONDS, state_dir=state_dir):
        events = state.read_events_from_path(stale)
        if not events:
            state.discard_claim(stale)
            continue
        try:
            _summary, flush_ok = _export_events(events)
        except BaseException as exc:  # noqa: BLE001 - keep going through the backlog
            print(f"traccia copilot flush: stale claim {stale.name}: {exc}", file=sys.stderr)
            continue
        if flush_ok:
            state.discard_claim(stale)
        else:
            state.archive_failed_claim(stale)
        results[stale.name] = _summary
    return results


def retry_failed(*, state_dir: Optional[Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Re-attempt export for every log parked in the ``failed/`` directory."""
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for path in state.list_failed(state_dir=state_dir):
        events = state.read_events_from_path(path)
        if not events:
            state.discard_claim(path)
            continue
        try:
            summary, flush_ok = _export_events(events)
        except BaseException as exc:  # keep going through the rest of the backlog
            results[path.name] = None
            print(f"traccia copilot flush --retry-failed: {path.name}: {exc}", file=sys.stderr)
            continue
        if flush_ok:
            state.discard_claim(path)
        results[path.name] = summary
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="traccia-copilot-flush")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Flush a single session id")
    group.add_argument("--all", action="store_true", help="Flush every eligible buffered session")
    group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt export for sessions parked in failed/ after a prior export error",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=None,
        help="With --all, also flush sessions with no sessionEnd that have been idle at least this long",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="With --all, also flush sessions that appear still active (no sessionEnd yet)",
    )
    args = parser.parse_args(argv)

    try:
        if args.session:
            flush_session(args.session)
        elif args.retry_failed:
            retry_failed()
        else:
            flush_all(
                max_age_seconds=args.max_age_seconds,
                include_active=args.include_active,
            )
        return 0
    except Exception as exc:
        print(f"traccia copilot flush failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
