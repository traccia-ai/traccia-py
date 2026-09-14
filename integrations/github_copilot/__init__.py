"""Traccia integration for GitHub Copilot (CLI + cloud coding agent hooks)."""

from typing import Optional

_installed = False
_capture_content = False


def install(enabled: Optional[bool] = None, capture_content: Optional[bool] = None) -> bool:
    """
    Enable the GitHub Copilot hooks integration.

    To wire Copilot up to Traccia, run the CLI command once per repo:
    
        traccia copilot install-hooks

    Args:
        enabled: If False, disable. If None, check config (default: enabled).
        capture_content: If True, capture (redacted) tool/prompt content
            instead of length-only metadata. If None, check config
            (default: False -- metadata only).

    Returns:
        True if enabled, False otherwise.
    """
    global _installed, _capture_content

    if enabled is False:
        _installed = False
        return False

    if enabled is None:
        from traccia import runtime_config
        if runtime_config.get_config_value("github_copilot") is False:
            _installed = False
            return False

    if capture_content is not None:
        _capture_content = bool(capture_content)

    _installed = True
    return True


__all__ = ["install"]
