"""Traccia callback handler for LangChain."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.messages import BaseMessage
except ImportError as e:
    raise ModuleNotFoundError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install traccia[langchain]"
    ) from e

from traccia import get_tracer
from traccia.tracer.span import SpanStatus


class TracciaCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that creates Traccia spans for LLM and chain runs.
    
    This handler integrates LangChain with Traccia's tracing system, creating spans
    for LLM calls with the same attributes used by Traccia's OpenAI instrumentation.
    
    Usage:
        ```python
        from traccia import init
        from traccia.integrations.langchain import CallbackHandler  # or TracciaCallbackHandler
        from langchain_openai import ChatOpenAI
        
        # Initialize Traccia
        init()
        
        # Create Traccia handler (no args)
        traccia_handler = CallbackHandler()
        
        # Use with any LangChain runnable
        llm = ChatOpenAI(model="gpt-4o-mini")
        result = llm.invoke(
            "Tell me a joke",
            config={"callbacks": [traccia_handler]}
        )
        ```
    
    Note:
        Requires langchain-core to be installed:
        ```bash
        pip install traccia[langchain]
        ```
    """
    
    def __init__(self):
        """Initialize the callback handler."""
        super().__init__()
        self.tracer = get_tracer("traccia.langchain")
        
        # Track active spans by run_id
        self._spans: Dict[UUID, Any] = {}
        self._context_tokens: Dict[UUID, Any] = {}
        
        # Track parent relationships
        self._parent_map: Dict[UUID, Optional[UUID]] = {}
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event."""
        self._parent_map[run_id] = parent_run_id
        
        try:
            # Extract attributes
            attributes = self._build_llm_attributes(
                serialized, prompts, None, kwargs, metadata
            )
            
            # Start span
            span = self.tracer.start_as_current_span(
                "llm.langchain.run",
                attributes=attributes
            )
            
            # Store span
            self._spans[run_id] = span
            self._context_tokens[run_id] = span
            
        except Exception as e:
            # Don't break LangChain execution
            import logging
            logging.getLogger(__name__).exception(f"Error in on_llm_start: {e}")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chat model start event."""
        self._parent_map[run_id] = parent_run_id
        
        try:
            # Convert messages to prompt format
            message_dicts = []
            for msg_list in messages:
                for msg in msg_list:
                    message_dicts.append(self._convert_message_to_dict(msg))
            
            # Extract attributes
            attributes = self._build_llm_attributes(
                serialized, None, message_dicts, kwargs, metadata
            )
            
            # Start span
            span = self.tracer.start_as_current_span(
                "llm.langchain.run",
                attributes=attributes
            )
            
            # Store span
            self._spans[run_id] = span
            self._context_tokens[run_id] = span
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Error in on_chat_model_start: {e}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM end event."""
        try:
            span = self._spans.pop(run_id, None)
            if span is None:
                return
            
            # Extract usage and output
            self._set_llm_response_attributes(span, response)
            
            # End span
            span.__exit__(None, None, None)
            
            # Clean up context
            self._context_tokens.pop(run_id, None)
            self._parent_map.pop(run_id, None)
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Error in on_llm_end: {e}")
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM error event."""
        try:
            span = self._spans.pop(run_id, None)
            if span is None:
                return
            
            # Record exception
            span._otel_span.record_exception(error)
            span.set_status(SpanStatus.ERROR, str(error))
            
            # End span
            span.__exit__(type(error), error, None)
            
            # Clean up
            self._context_tokens.pop(run_id, None)
            self._parent_map.pop(run_id, None)
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"Error in on_llm_error: {e}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain start event (optional Phase 2)."""
        self._parent_map[run_id] = parent_run_id
        # Phase 2: Can add chain spans here
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain end event (optional Phase 2)."""
        self._parent_map.pop(run_id, None)
        # Phase 2: Can end chain spans here
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle chain error event (optional Phase 2)."""
        self._parent_map.pop(run_id, None)
        # Phase 2: Can handle chain errors here
    
    def _build_llm_attributes(
        self,
        serialized: Dict[str, Any],
        prompts: Optional[List[str]],
        messages: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build LLM span attributes."""
        from traccia.integrations.langchain.utils import extract_model_name
        
        attributes: Dict[str, Any] = {}
        
        # Extract vendor
        vendor = self._extract_vendor(serialized)
        if vendor:
            attributes["llm.vendor"] = vendor
        
        # Extract model
        model = extract_model_name(serialized, kwargs, metadata)
        if model:
            attributes["llm.model"] = model
        
        # Set prompt
        if messages:
            # Chat messages
            prompt_text = self._format_messages_as_prompt(messages)
            if prompt_text:
                attributes["llm.prompt"] = prompt_text
            
            # Store messages as JSON (truncated)
            try:
                messages_json = json.dumps(messages)[:1000]
                attributes["llm.openai.messages"] = messages_json
            except Exception:
                pass
        elif prompts:
            # Text prompts
            if len(prompts) == 1:
                attributes["llm.prompt"] = prompts[0]
            else:
                attributes["llm.prompt"] = json.dumps(prompts)[:1000]
        
        return attributes
    
    def _set_llm_response_attributes(
        self,
        span: Any,
        response: LLMResult,
    ) -> None:
        """Set response attributes on span."""
        usage = self._parse_usage(response)
        if usage:
            span.set_attribute("llm.usage.source", "provider_usage")
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            if prompt_tokens is not None:
                span.set_attribute("llm.usage.prompt_tokens", int(prompt_tokens))
                span.set_attribute("llm.usage.prompt_source", "provider_usage")
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            if completion_tokens is not None:
                span.set_attribute("llm.usage.completion_tokens", int(completion_tokens))
                span.set_attribute("llm.usage.completion_source", "provider_usage")
            total = usage.get("total_tokens")
            if total is not None:
                span.set_attribute("llm.usage.total_tokens", int(total))

        # Extract completion (last generation text or message content)
        if response.generations and len(response.generations) > 0:
            last_gen = response.generations[-1]
            if last_gen and len(last_gen) > 0:
                chunk = last_gen[-1]
                completion = getattr(chunk, "text", None) or (
                    getattr(getattr(chunk, "message", None), "content", None)
                )
                if completion:
                    span.set_attribute("llm.completion", str(completion))
    
    def _extract_vendor(self, serialized: Dict[str, Any]) -> Optional[str]:
        """Extract vendor from serialized LLM config."""
        if not serialized or "id" not in serialized:
            return None
        
        id_list = serialized["id"]
        if not isinstance(id_list, list) or len(id_list) == 0:
            return None
        
        # Get last component (class name)
        class_name = id_list[-1].lower()
        
        # Map to vendor
        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name:
            return "anthropic"
        elif "cohere" in class_name:
            return "cohere"
        elif "huggingface" in class_name:
            return "huggingface"
        elif "vertexai" in class_name or "vertex" in class_name:
            return "google"
        elif "bedrock" in class_name:
            return "aws"
        
        return "langchain"
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert LangChain message to dict."""
        return {
            "role": getattr(message, "type", "unknown"),
            "content": str(message.content) if hasattr(message, "content") else str(message),
        }
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as a prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts) if parts else ""

    def _parse_usage(self, response: LLMResult) -> Optional[Dict[str, Any]]:
        """
        Extract token usage from LLMResult.
        Checks llm_output['token_usage'], llm_output['usage'], and generation_info.
        """
        usage = None
        if response.llm_output:
            for key in ("token_usage", "usage"):
                if key in response.llm_output and response.llm_output[key]:
                    raw = response.llm_output[key]
                    usage = self._normalize_usage(raw)
                    if usage:
                        break
        if not usage and response.generations:
            for gen_list in response.generations:
                for chunk in gen_list:
                    if getattr(chunk, "generation_info", None) and isinstance(
                        chunk.generation_info, dict
                    ):
                        raw = chunk.generation_info.get("usage_metadata")
                        if raw:
                            usage = self._normalize_usage(raw)
                            break
                    msg = getattr(chunk, "message", None)
                    if msg is not None:
                        meta = getattr(msg, "response_metadata", None) or {}
                        raw = meta.get("usage") if isinstance(meta, dict) else None
                        if raw:
                            usage = self._normalize_usage(raw)
                            break
                if usage:
                    break
        return usage

    @staticmethod
    def _normalize_usage(raw: Any) -> Optional[Dict[str, Any]]:
        """Normalize usage dict to prompt_tokens, completion_tokens, total_tokens."""
        if raw is None:
            return None
        if hasattr(raw, "__dict__"):
            raw = getattr(raw, "__dict__", raw)
        if not isinstance(raw, dict):
            return None
        # Map common keys to Traccia/OpenAI style
        prompt = raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("input")
        completion = raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("output")
        total = raw.get("total_tokens") or raw.get("total")
        if prompt is None and completion is None and total is None:
            return None
        out: Dict[str, Any] = {}
        if prompt is not None:
            out["prompt_tokens"] = int(prompt) if not isinstance(prompt, list) else sum(prompt)
        if completion is not None:
            out["completion_tokens"] = int(completion) if not isinstance(completion, list) else sum(completion)
        if total is not None:
            out["total_tokens"] = int(total) if not isinstance(total, list) else sum(total)
        return out if out else None
