"""Traccia integrations for frameworks like LangChain, CrewAI, LlamaIndex."""

__all__ = []

# Lazy imports for optional dependencies
def _import_langchain():
    try:
        from traccia.integrations.langchain import TracciaCallbackHandler
        return TracciaCallbackHandler
    except ImportError as e:
        raise ModuleNotFoundError(
            "LangChain integration requires langchain-core. "
            "Install with: pip install traccia[langchain]"
        ) from e


# Make available if imported
try:
    from traccia.integrations.langchain import TracciaCallbackHandler
    __all__.append("TracciaCallbackHandler")
except ImportError:
    pass
