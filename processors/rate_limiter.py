"""Rate limiting processor for span export."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor as OTelSpanProcessor

from traccia.errors import RateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with hybrid blocking/dropping behavior.
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Short blocking period before dropping spans
    - Detailed logging of dropped spans
    - Thread-safe implementation
    """
    
    def __init__(
        self,
        max_spans_per_second: Optional[float] = None,
        max_block_ms: int = 100,
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_spans_per_second: Maximum spans per second (None = unlimited)
            max_block_ms: Maximum milliseconds to block before dropping
        """
        self.max_spans_per_second = max_spans_per_second
        self.max_block_ms = max_block_ms
        self.enabled = max_spans_per_second is not None and max_spans_per_second > 0
        
        # Token bucket state
        self._tokens: float = max_spans_per_second or 0
        self._max_tokens: float = max_spans_per_second or 0
        self._last_refill_time: float = time.time()
        self._lock = threading.Lock()
        
        # Stats
        self._total_spans = 0
        self._dropped_spans = 0
        self._blocked_spans = 0
        
        # Recent timestamps for sliding window (backup)
        self._recent_timestamps: deque = deque()
        self._window_seconds = 1.0
    
    def acquire(self, span: Optional[ReadableSpan] = None) -> bool:
        """
        Try to acquire permission to process a span.
        
        Returns True if span should be processed, False if it should be dropped.
        
        Behavior:
        1. If unlimited (disabled), always return True
        2. Try to acquire a token immediately
        3. If no token, block for up to max_block_ms
        4. If still no token after blocking, drop and return False
        
        Args:
            span: Optional span for logging purposes
            
        Returns:
            True if span should be processed, False if dropped
        """
        if not self.enabled:
            return True
        
        self._total_spans += 1
        
        with self._lock:
            # Refill tokens based on elapsed time
            self._refill_tokens()
            
            # Try to acquire immediately
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            
            # No tokens available, try blocking
            if self.max_block_ms > 0:
                block_start = time.time()
                blocked_ms = 0
                
                while blocked_ms < self.max_block_ms:
                    # Release lock briefly to allow other threads
                    self._lock.release()
                    time.sleep(0.001)  # Sleep 1ms
                    self._lock.acquire()
                    
                    # Refill and try again
                    self._refill_tokens()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        self._blocked_spans += 1
                        return True
                    
                    blocked_ms = (time.time() - block_start) * 1000
            
            # Still no tokens after blocking - drop the span
            self._dropped_spans += 1
            
            # Log dropped span
            span_name = span.name if span else "unknown"
            logger.warning(
                f"Rate limit exceeded - dropping span '{span_name}'. "
                f"Total dropped: {self._dropped_spans}/{self._total_spans} "
                f"({self._dropped_spans / self._total_spans * 100:.1f}%)"
            )
            
            return False
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time (token bucket algorithm)."""
        now = time.time()
        elapsed = now - self._last_refill_time
        
        if elapsed > 0:
            # Add tokens based on rate and elapsed time
            new_tokens = elapsed * self.max_spans_per_second
            self._tokens = min(self._max_tokens, self._tokens + new_tokens)
            self._last_refill_time = now
    
    def get_stats(self) -> dict:
        """Get rate limiting statistics."""
        with self._lock:
            drop_rate = (self._dropped_spans / self._total_spans * 100) if self._total_spans > 0 else 0
            return {
                "enabled": self.enabled,
                "max_spans_per_second": self.max_spans_per_second,
                "total_spans": self._total_spans,
                "dropped_spans": self._dropped_spans,
                "blocked_spans": self._blocked_spans,
                "drop_rate_percent": round(drop_rate, 2),
                "current_tokens": round(self._tokens, 2),
            }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._total_spans = 0
            self._dropped_spans = 0
            self._blocked_spans = 0


class RateLimitingSpanProcessor(OTelSpanProcessor):
    """
    Span processor that enforces rate limiting before passing to next processor.
    
    This should be added early in the processor chain to drop spans before
    they consume resources in downstream processors.
    """
    
    def __init__(
        self,
        next_processor: Any = None,
        max_spans_per_second: Optional[float] = None,
        max_block_ms: int = 100,
    ):
        """
        Initialize rate limiting processor.
        
        Args:
            next_processor: Next processor in the chain
            max_spans_per_second: Maximum spans per second (None = unlimited)
            max_block_ms: Maximum milliseconds to block before dropping
        """
        super().__init__()
        self.next_processor = next_processor
        self.rate_limiter = RateLimiter(
            max_spans_per_second=max_spans_per_second,
            max_block_ms=max_block_ms,
        )
    
    def on_start(self, span: Any, parent_context: Optional[Any] = None) -> None:
        """Called when span starts - pass through to next processor."""
        if self.next_processor and hasattr(self.next_processor, 'on_start'):
            self.next_processor.on_start(span, parent_context)

    def _on_ending(self, span: Any) -> None:
        """Called when span is ending - pass through to next processor."""
        if self.next_processor and hasattr(self.next_processor, '_on_ending'):
            self.next_processor._on_ending(span)
    
    def on_end(self, span: Any) -> None:
        """
        Called when span ends - check rate limit before passing to next processor.
        
        If rate limit is exceeded, span is dropped and not passed to next processor.
        """
        # Check rate limit
        if not self.rate_limiter.acquire(span):
            # Span dropped - don't pass to next processor
            return
        
        # Pass to next processor
        if self.next_processor and hasattr(self.next_processor, 'on_end'):
            self.next_processor.on_end(span)
    
    def shutdown(self) -> None:
        """Shutdown processor and log final stats."""
        stats = self.rate_limiter.get_stats()
        if stats["enabled"] and stats["dropped_spans"] > 0:
            logger.info(
                f"Rate limiter shutdown. Final stats: "
                f"{stats['dropped_spans']}/{stats['total_spans']} spans dropped "
                f"({stats['drop_rate_percent']}%)"
            )
        
        if self.next_processor and hasattr(self.next_processor, 'shutdown'):
            self.next_processor.shutdown()
    
    def force_flush(
        self,
        timeout_millis: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Force flush - pass through adapting to processor convention."""
        if timeout_millis is not None:
            ms = int(timeout_millis)
            sec = ms / 1000.0
        elif timeout is not None:
            sec = float(timeout)
            ms = int(sec * 1000)
        else:
            ms = 30000
            sec = None

        if not self.next_processor or not hasattr(self.next_processor, 'force_flush'):
            return True

        return self._forward_force_flush(self.next_processor, ms, sec)

    @staticmethod
    def _forward_force_flush(processor: Any, timeout_millis: int, timeout_sec: Optional[float]) -> bool:
        """Forward force_flush adapting timeout argument."""
        try:
            params = inspect.signature(processor.force_flush).parameters
            if "timeout" in params and "timeout_millis" not in params:
                res = processor.force_flush(timeout=timeout_sec)
                return True if res is None else bool(res)
            if "timeout_millis" in params:
                res = processor.force_flush(timeout_millis=timeout_millis)
                return True if res is None else bool(res)
        except (ValueError, TypeError):
            pass

        try:
            res = processor.force_flush(timeout_millis=timeout_millis)
            return True if res is None else bool(res)
        except TypeError:
            try:
                res = processor.force_flush(timeout=timeout_sec)
                return True if res is None else bool(res)
            except TypeError:
                res = processor.force_flush()
                return True if res is None else bool(res)

