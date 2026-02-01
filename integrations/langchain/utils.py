"""Utility functions for LangChain integration."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, cast


def extract_model_name(
    serialized: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Extract model name from LangChain serialized config, invocation params, or metadata.
    
    Args:
        serialized: LangChain's serialized component dict
        kwargs: Keyword arguments from the callback (contains invocation_params)
        metadata: Optional metadata dict
        
    Returns:
        Model name string or None
    """
    # Check metadata first
    if metadata:
        model_from_meta = _parse_model_name_from_metadata(metadata)
        if model_from_meta:
            return model_from_meta
    
    # Try known model paths by ID
    models_by_id = [
        ("ChatOpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("ChatOpenAI", ["invocation_params", "model"], "kwargs"),
        ("OpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("AzureChatOpenAI", ["invocation_params", "model"], "kwargs"),
        ("AzureChatOpenAI", ["invocation_params", "model_name"], "kwargs"),
        ("AzureChatOpenAI", ["invocation_params", "azure_deployment"], "kwargs"),
        ("ChatAnthropic", ["invocation_params", "model"], "kwargs"),
        ("ChatAnthropic", ["invocation_params", "model_name"], "kwargs"),
        ("ChatGoogleGenerativeAI", ["kwargs", "model"], "serialized"),
        ("ChatVertexAI", ["kwargs", "model_name"], "serialized"),
        ("BedrockChat", ["kwargs", "model_id"], "serialized"),
        ("ChatBedrock", ["kwargs", "model_id"], "serialized"),
    ]
    
    for model_name, keys, select_from in models_by_id:
        model = _extract_model_by_path_for_id(
            model_name,
            serialized,
            kwargs,
            keys,
            cast(Literal["serialized", "kwargs"], select_from),
        )
        if model:
            return model
    
    # Try common paths as catch-all
    common_paths = [
        ["invocation_params", "model_name"],
        ["invocation_params", "model"],
        ["kwargs", "model_name"],
        ["kwargs", "model"],
    ]
    
    for select in ["kwargs", "serialized"]:
        for path in common_paths:
            model = _extract_model_by_path(
                serialized, kwargs, path, cast(Literal["serialized", "kwargs"], select)
            )
            if model:
                return str(model)
    
    return None


def _parse_model_name_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """Extract model name from metadata if present."""
    if not isinstance(metadata, dict):
        return None
    return metadata.get("ls_model_name", None)


def _extract_model_by_path_for_id(
    id: str,
    serialized: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
    keys: List[str],
    select_from: Literal["serialized", "kwargs"],
) -> Optional[str]:
    """Extract model if the serialized ID matches."""
    if serialized is None and select_from == "serialized":
        return None
    
    if serialized:
        serialized_id = serialized.get("id")
        if (
            serialized_id
            and isinstance(serialized_id, list)
            and len(serialized_id) > 0
            and serialized_id[-1] == id
        ):
            result = _extract_model_by_path(serialized, kwargs, keys, select_from)
            return str(result) if result is not None else None
    
    return None


def _extract_model_by_path(
    serialized: Optional[Dict[str, Any]],
    kwargs: dict,
    keys: List[str],
    select_from: Literal["serialized", "kwargs"],
) -> Optional[str]:
    """Extract value by following a path in the dict."""
    if serialized is None and select_from == "serialized":
        return None
    
    current_obj = kwargs if select_from == "kwargs" else serialized
    
    for key in keys:
        if current_obj and isinstance(current_obj, dict):
            current_obj = current_obj.get(key)
        else:
            return None
        if not current_obj:
            return None
    
    return str(current_obj) if current_obj else None
