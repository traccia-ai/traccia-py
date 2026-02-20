"""Runtime configuration state management."""

from typing import Optional, List, Any

# Global runtime configuration state
_config = {
    "auto_instrument_tools": False,
    "tool_include": [],
    "max_tool_spans": 1000,
    "max_span_depth": 100,
    "session_id": None,
    "user_id": None,
    "tenant_id": None,
    "project_id": None,
    "agent_id": None,
    "agent_name": None,
    "env": None,
    "debug": False,
    "attr_truncation_limit": 1000,
    "openai_agents": True,
    "crewai": True,
}


def set_auto_instrument_tools(value: bool) -> None:
    _config["auto_instrument_tools"] = value


def get_auto_instrument_tools() -> bool:
    return _config["auto_instrument_tools"]


def set_tool_include(value: List[str]) -> None:
    _config["tool_include"] = value


def get_tool_include() -> List[str]:
    return _config["tool_include"]


def set_max_tool_spans(value: int) -> None:
    _config["max_tool_spans"] = value


def get_max_tool_spans() -> int:
    return _config["max_tool_spans"]


def set_max_span_depth(value: int) -> None:
    _config["max_span_depth"] = value


def get_max_span_depth() -> int:
    return _config["max_span_depth"]


def set_session_id(value: Optional[str]) -> None:
    _config["session_id"] = value


def get_session_id() -> Optional[str]:
    return _config["session_id"]


def set_user_id(value: Optional[str]) -> None:
    _config["user_id"] = value


def get_user_id() -> Optional[str]:
    return _config["user_id"]


def set_tenant_id(value: Optional[str]) -> None:
    _config["tenant_id"] = value


def get_tenant_id() -> Optional[str]:
    return _config["tenant_id"]


def set_project_id(value: Optional[str]) -> None:
    _config["project_id"] = value


def get_project_id() -> Optional[str]:
    return _config["project_id"]


def set_agent_id(value: Optional[str]) -> None:
    _config["agent_id"] = value


def get_agent_id() -> Optional[str]:
    return _config["agent_id"]


def set_agent_name(value: Optional[str]) -> None:
    _config["agent_name"] = value


def get_agent_name() -> Optional[str]:
    return _config["agent_name"]


def set_env(value: Optional[str]) -> None:
    _config["env"] = value


def get_env() -> Optional[str]:
    return _config["env"]


def set_debug(value: bool) -> None:
    _config["debug"] = value


def get_debug() -> bool:
    return _config["debug"]


def set_attr_truncation_limit(value: int) -> None:
    _config["attr_truncation_limit"] = value


def get_attr_truncation_limit() -> int:
    return _config["attr_truncation_limit"]


def get_openai_agents() -> bool:
    """Get whether OpenAI Agents SDK integration is enabled."""
    return _config.get("openai_agents", True)


def set_openai_agents(value: bool) -> None:
    """Set whether OpenAI Agents SDK integration is enabled."""
    _config["openai_agents"] = value


def get_crewai() -> bool:
    """Get whether CrewAI integration is enabled."""
    return _config.get("crewai", True)


def set_crewai(value: bool) -> None:
    """Set whether CrewAI integration is enabled."""
    _config["crewai"] = value


def set_config_value(key: str, value: Any) -> None:
    """Set a runtime config value."""
    _config[key] = value


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a runtime config value."""
    return _config.get(key, default)
