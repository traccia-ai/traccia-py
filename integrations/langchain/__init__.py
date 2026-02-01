"""Traccia LangChain integration via callback handler."""

try:
    from traccia.integrations.langchain.callback import TracciaCallbackHandler
    
    # Convenience alias: from traccia.integrations.langchain import CallbackHandler
    CallbackHandler = TracciaCallbackHandler
    
    __all__ = ["TracciaCallbackHandler", "CallbackHandler"]
except ImportError as e:
    raise ModuleNotFoundError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install traccia[langchain]"
    ) from e
