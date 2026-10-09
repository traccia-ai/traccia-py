"""Gemini (google-genai) monkey patching for interactions.create."""

from __future__ import annotations

import time

from traccia.tracer.span import SpanStatus

_patched = False


def _compute_cost(model, prompt_tokens, completion_tokens):
    """Compute cost from token usage using pricing config."""
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    try:
        from traccia.pricing_config import load_pricing
        from traccia.processors.cost_engine import compute_cost as _compute

        return _compute(model, prompt_tokens, completion_tokens, load_pricing())
    except Exception:
        return None


def _record_llm_metrics(model, input_tokens, output_tokens, duration, cost):
    """Record LLM metrics if metrics are enabled."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder

        recorder = get_metrics_recorder()
        if not recorder:
            return
        attributes = {"gen_ai.system": "google_gemini"}
        if model:
            attributes["gen_ai.request.model"] = model
        try:
            from traccia import runtime_config as _rc

            _aid = _rc.get_agent_id()
            _aname = _rc.get_agent_name()
            _env = _rc.get_env()
            if _aid:
                attributes["agent.id"] = _aid
                attributes["agent_id"] = _aid
            if _aname:
                attributes["agent.name"] = _aname
            if _env:
                attributes["environment"] = _env
        except Exception:
            pass
        if input_tokens is not None or output_tokens is not None:
            recorder.record_token_usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                attributes=attributes,
            )
        if duration is not None:
            recorder.record_duration(duration, attributes=attributes)
        if cost is not None:
            recorder.record_cost(cost, attributes=attributes)
    except Exception:
        pass


def _safe_get(obj, attr, default=None):
    """Safely get a nested attribute or dict key (dot-separated path)."""
    cur = obj
    for part in attr.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def _extract_usage(resp):
    """Return (input, output, thought, cached, tool_use, total) tokens from a Gemini Interaction."""
    usage = _safe_get(resp, "usage")
    if usage is None:
        return None, None, None, None, None, None

    input_tokens = _safe_get(usage, "total_input_tokens")
    if input_tokens is None:
        input_tokens = _safe_get(usage, "input_tokens")
        
    output_tokens = _safe_get(usage, "total_output_tokens")
    if output_tokens is None:
        output_tokens = _safe_get(usage, "output_tokens")
        
    thought_tokens = _safe_get(usage, "total_thought_tokens")
    cached_tokens = _safe_get(usage, "total_cached_tokens")
    tool_use_tokens = _safe_get(usage, "total_tool_use_tokens")
    total_tokens = _safe_get(usage, "total_tokens")

    return (
        input_tokens,
        output_tokens,
        thought_tokens,
        cached_tokens,
        tool_use_tokens,
        total_tokens,
    )


def _extract_prompt(kwargs, args):
    """Return the request's `input` field as prompt text, if present."""
    obj = args[0] if args else None
    prompt = kwargs.get("input")
    if prompt is None:
        prompt = _safe_get(obj, "input")
    if prompt is None:
        return None
    return str(prompt)


def _populate_span(span, resp, model, t0):
    """Write all Interaction fields into the current span, then record metrics."""
    
    if not model:
        resp_model = _safe_get(resp, "model")
        if resp_model and "llm.model" not in span.attributes:
            span.set_attribute("llm.model", str(resp_model))
        model = resp_model or model

    input_tok, output_tok, thought_tok, cached_tok, tool_use_tok, total_tok = _extract_usage(resp)

    # Token attributes — OpenAI-compatible aliases so downstream processors work
    if input_tok is not None:
        span.set_attribute("llm.usage.prompt_tokens", input_tok)
        span.set_attribute("llm.usage.input_tokens", input_tok)
        span.set_attribute("llm.usage.prompt_source", "provider_usage")
    if output_tok is not None:
        span.set_attribute("llm.usage.completion_tokens", output_tok)
        span.set_attribute("llm.usage.output_tokens", output_tok)
        span.set_attribute("llm.usage.completion_source", "provider_usage")
    if thought_tok is not None:
        span.set_attribute("llm.usage.thought_tokens", thought_tok)
    if cached_tok is not None:
        span.set_attribute("llm.usage.cached_tokens", cached_tok)
    if tool_use_tok is not None:
        span.set_attribute("llm.usage.tool_use_tokens", tool_use_tok)

    # This is fallback if the provider didn't send one.
    if total_tok is not None:
        span.set_attribute("llm.usage.total_tokens", total_tok)
        span.set_attribute("llm.usage.source", "provider_usage")
    elif input_tok is not None and output_tok is not None:
        span.set_attribute("llm.usage.total_tokens", input_tok + output_tok)
        span.set_attribute("llm.usage.source", "provider_usage")

    # Output text (first 4 KB)
    output_text = _safe_get(resp, "output_text")
    if output_text:
        span.set_attribute("llm.completion", str(output_text)[:4096])

    # Interaction / request identifiers
    interaction_id = _safe_get(resp, "id")
    if interaction_id:
        span.set_attribute("llm.interaction_id", str(interaction_id))

    # Status (completed / failed / etc.)
    status = _safe_get(resp, "status")
    if status:
        span.set_attribute("llm.response.status", str(status))

    # Agent field — non-null means an agent ran the interaction
    agent = _safe_get(resp, "agent")
    if agent is not None:
        span.set_attribute("llm.is_agent_run", True)

    # Metrics
    duration_val = time.perf_counter() - t0
    cost_val = _compute_cost(model, input_tok, output_tok)
    _record_llm_metrics(
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        duration=duration_val,
        cost=cost_val,
    )


def _record_exception_metric(model):
    """Fire a single exception counter metric on error."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder

        rec = get_metrics_recorder()
        if rec:
            rec.record_exception(
                attributes={
                    "gen_ai.system": "google_gemini",
                    "gen_ai.request.model": model or "unknown",
                }
            )
    except Exception:
        pass


def _build_sync_wrapper(original_create):
    """Return a sync wrapper around the original Interactions.create."""

    def sync_wrapped(self, *args, **kwargs):
        tracer = _get_tracer("gemini")
        model = kwargs.get("model") or _safe_get(args[0] if args else None, "model")
        attributes = {"llm.vendor": "google_gemini"}
        if model:
            attributes["llm.model"] = model

        prompt_text = _extract_prompt(kwargs, args)
        if prompt_text:
            attributes["llm.prompt"] = prompt_text[:4096]

        previous_interaction_id = kwargs.get("previous_interaction_id") or _safe_get(
            args[0] if args else None, "previous_interaction_id"
        )
        if previous_interaction_id:
            attributes["llm.previous_interaction_id"] = str(previous_interaction_id)

        streaming = kwargs.get("stream") is True or (
            _safe_get(args[0] if args else None, "stream") is True
        )
        if streaming:
            attributes["llm.streaming"] = True

        t0 = time.perf_counter()
        with tracer.start_as_current_span(
            "llm.gemini.interaction", attributes=attributes
        ) as span:
            decision = None
            try:
                from traccia.governance.pep import enforce_llm_call, finish_llm_call

                decision = enforce_llm_call(kwargs)
                resp = original_create(self, *args, **kwargs)
                if not streaming:
                    _populate_span(span, resp, model, t0)
                finish_llm_call(decision)
                return resp
            except Exception as exc:
                try:
                    from traccia.governance.pep import finish_llm_call as _finish
                    _finish(decision, release=True)
                except Exception:
                    pass
                span.record_exception(exc)
                span.set_status(SpanStatus.ERROR, str(exc))
                _record_exception_metric(model)
                raise

    sync_wrapped._agent_trace_patched = True
    return sync_wrapped


def _build_async_wrapper(original_create):
    """Return an async wrapper around the original AsyncInteractions.create."""

    async def async_wrapped(self, *args, **kwargs):
        tracer = _get_tracer("gemini")
        model = kwargs.get("model") or _safe_get(args[0] if args else None, "model")
        attributes = {"llm.vendor": "google_gemini"}
        if model:
            attributes["llm.model"] = model

        prompt_text = _extract_prompt(kwargs, args)
        if prompt_text:
            attributes["llm.prompt"] = prompt_text[:4096]

        previous_interaction_id = kwargs.get("previous_interaction_id") or _safe_get(
            args[0] if args else None, "previous_interaction_id"
        )
        if previous_interaction_id:
            attributes["llm.previous_interaction_id"] = str(previous_interaction_id)

        streaming = kwargs.get("stream") is True or (
            _safe_get(args[0] if args else None, "stream") is True
        )
        if streaming:
            attributes["llm.streaming"] = True

        t0 = time.perf_counter()
        with tracer.start_as_current_span(
            "llm.gemini.interaction", attributes=attributes
        ) as span:
            decision = None
            try:
                from traccia.governance.pep import enforce_llm_call, finish_llm_call

                decision = enforce_llm_call(kwargs)
                resp = await original_create(self, *args, **kwargs)
                if not streaming: 
                    _populate_span(span, resp, model, t0)
                finish_llm_call(decision)
                return resp
            except Exception as exc:
                try:
                    from traccia.governance.pep import finish_llm_call as _finish
                    _finish(decision, release=True)
                except Exception:
                    pass
                span.record_exception(exc)
                span.set_status(SpanStatus.ERROR, str(exc))
                _record_exception_metric(model)
                raise

    async_wrapped._agent_trace_patched = True
    return async_wrapped


def _extract_generate_content_prompt(kwargs, args):
    """Return prompt text for models.generate_content."""
    contents = kwargs.get("contents")
    if contents is None:
        if len(args) >= 2:
            contents = args[1]
        elif len(args) == 1:
            contents = args[0]
    if contents is None:
        return None
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            else:
                txt = _safe_get(item, "text") or str(item)
                parts.append(txt)
        return "\n".join(parts)
    return str(contents)


def _populate_generate_content_span(span, resp, model, t0):
    """Write generate_content response attributes into span."""
    if not model:
        resp_model = _safe_get(resp, "model_version") or _safe_get(resp, "model")
        if resp_model and "llm.model" not in span.attributes:
            span.set_attribute("llm.model", str(resp_model))
        model = resp_model or model

    usage = _safe_get(resp, "usage_metadata")
    prompt_tok = _safe_get(usage, "prompt_token_count")
    if prompt_tok is None:
        prompt_tok = _safe_get(usage, "prompt_tokens")
    if prompt_tok is None:
        prompt_tok = _safe_get(usage, "input_tokens")

    completion_tok = _safe_get(usage, "candidates_token_count")
    if completion_tok is None:
        completion_tok = _safe_get(usage, "completion_tokens")
    if completion_tok is None:
        completion_tok = _safe_get(usage, "output_tokens")

    thought_tok = _safe_get(usage, "thoughts_token_count")

    total_tok = _safe_get(usage, "total_token_count")
    if total_tok is None:
        total_tok = _safe_get(usage, "total_tokens")

    if prompt_tok is not None:
        span.set_attribute("llm.usage.prompt_tokens", prompt_tok)
        span.set_attribute("llm.usage.input_tokens", prompt_tok)
        span.set_attribute("llm.usage.prompt_source", "provider_usage")
    if completion_tok is not None:
        span.set_attribute("llm.usage.completion_tokens", completion_tok)
        span.set_attribute("llm.usage.output_tokens", completion_tok)
        span.set_attribute("llm.usage.completion_source", "provider_usage")
    if thought_tok is not None:
        span.set_attribute("llm.usage.thought_tokens", thought_tok)
    if total_tok is not None:
        span.set_attribute("llm.usage.total_tokens", total_tok)
        span.set_attribute("llm.usage.source", "provider_usage")

    output_text = _safe_get(resp, "text")
    if output_text:
        span.set_attribute("llm.completion", str(output_text)[:4096])

    cost_val = _compute_cost(model, prompt_tok, completion_tok)

    duration_val = time.perf_counter() - t0
    _record_llm_metrics(
        model=model,
        input_tokens=prompt_tok,
        output_tokens=completion_tok,
        duration=duration_val,
        cost=cost_val,
    )


def _build_models_sync_wrapper(original_generate_content):
    """Return a sync wrapper around Models.generate_content."""

    def sync_wrapped(self, *args, **kwargs):
        tracer = _get_tracer("gemini")
        model = kwargs.get("model") or (args[0] if args else None)
        attributes = {"llm.vendor": "google_gemini"}
        if model:
            attributes["llm.model"] = str(model)

        prompt_text = _extract_generate_content_prompt(kwargs, args)
        if prompt_text:
            attributes["llm.prompt"] = prompt_text[:4096]

        t0 = time.perf_counter()
        with tracer.start_as_current_span(
            "llm.gemini.generate_content", attributes=attributes
        ) as span:
            decision = None
            try:
                from traccia.governance.pep import enforce_llm_call, finish_llm_call

                decision = enforce_llm_call(kwargs)
                resp = original_generate_content(self, *args, **kwargs)
                _populate_generate_content_span(span, resp, str(model) if model else None, t0)
                finish_llm_call(decision)
                return resp
            except Exception as exc:
                try:
                    from traccia.governance.pep import finish_llm_call as _finish
                    _finish(decision, release=True)
                except Exception:
                    pass
                span.record_exception(exc)
                span.set_status(SpanStatus.ERROR, str(exc))
                _record_exception_metric(str(model) if model else None)
                raise

    sync_wrapped._agent_trace_patched = True
    return sync_wrapped


def patch_gemini():
    """Patch google-genai interactions.create (sync+async) and models.generate_content; returns True if patched.

    Tries two SDK layouts in priority order:
      SDK >=2.x: google.genai._gaos.google_genai  (GeminiNextGenInteractions)
      SDK  <2.x: google.genai.resources.interactions  (Interactions)
    Also patches Models.generate_content (SDK models).
    """
    global _patched
    if _patched:
        return True
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False  # google-genai not installed

    import importlib
    patched_any = False

    _CANDIDATES = [
        ("google.genai._gaos.google_genai",
         "GeminiNextGenInteractions",
         "AsyncGeminiNextGenInteractions"),
        ("google.genai.resources.interactions",
         "Interactions",
         "AsyncInteractions"),
    ]

    for mod_path, sync_name, async_name in _CANDIDATES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue  # not present in this SDK version

        hit = False
        try:
            sync_cls = getattr(mod, sync_name, None)
            if sync_cls is not None:
                orig = getattr(sync_cls, "create", None)
                if orig and not getattr(orig, "_agent_trace_patched", False):
                    setattr(sync_cls, "create", _build_sync_wrapper(orig))
                    hit = True

            async_cls = getattr(mod, async_name, None)
            if async_cls is not None:
                orig = getattr(async_cls, "create", None)
                if orig and not getattr(orig, "_agent_trace_patched", False):
                    setattr(async_cls, "create", _build_async_wrapper(orig))
                    hit = True
        except Exception:
            continue

        if hit:
            patched_any = True
            break

    _MODELS_CANDIDATES = [
        ("google.genai.models", "Models"),
        ("google.genai.resources.models", "Models"),
    ]

    for mod_path, cls_name in _MODELS_CANDIDATES:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                orig = getattr(cls, "generate_content", None)
                if orig and not getattr(orig, "_agent_trace_patched", False):
                    setattr(cls, "generate_content", _build_models_sync_wrapper(orig))
                    patched_any = True
        except Exception:
            continue

    if patched_any:
        _patched = True
    return _patched


def _get_tracer(name):
    import traccia

    return traccia.get_tracer(name)
