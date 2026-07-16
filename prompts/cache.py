"""Client-side prompt cache with TTL and stale-while-revalidate."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("traccia.prompts")

CacheKey = Tuple[str, str]  # (name, "label:production" | "version:12")


@dataclass
class CacheEntry:
    payload: Dict[str, Any]
    fetched_at: float
    etag: Optional[str] = None


class PromptCache:
    """Thread-safe TTL cache. Expired entries remain for SWR until replaced."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: Dict[CacheKey, CacheEntry] = {}
        self._lock = threading.Lock()
        self._refreshing: set = set()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._refreshing.clear()

    def make_key(self, name: str, *, label: Optional[str], version: Optional[int]) -> CacheKey:
        if version is not None:
            return (name, f"version:{version}")
        return (name, f"label:{label or 'production'}")

    def get(self, key: CacheKey) -> Optional[Tuple[CacheEntry, bool]]:
        """Return (entry, is_fresh) or None."""
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            age = time.time() - entry.fetched_at
            return entry, age < self.ttl_seconds

    def set(self, key: CacheKey, payload: Dict[str, Any], *, etag: Optional[str] = None) -> None:
        with self._lock:
            self._entries[key] = CacheEntry(payload=payload, fetched_at=time.time(), etag=etag)

    def begin_refresh(self, key: CacheKey) -> bool:
        """Return True if this caller should perform the background refresh."""
        with self._lock:
            if key in self._refreshing:
                return False
            self._refreshing.add(key)
            return True

    def end_refresh(self, key: CacheKey) -> None:
        with self._lock:
            self._refreshing.discard(key)

    def stale_while_revalidate(
        self,
        key: CacheKey,
        fetch: Callable[[], Tuple[Dict[str, Any], Optional[str]]],
    ) -> None:
        """Kick a daemon thread to refresh; on failure keep last good."""
        if not self.begin_refresh(key):
            return

        def _run() -> None:
            try:
                payload, etag = fetch()
                self.set(key, payload, etag=etag)
            except Exception as exc:
                logger.warning("prompt_stale_served: background refresh failed for %s: %s", key, exc)
            finally:
                self.end_refresh(key)

        threading.Thread(target=_run, name="traccia-prompt-swr", daemon=True).start()
