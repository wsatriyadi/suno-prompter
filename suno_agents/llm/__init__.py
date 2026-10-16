"""LLM transport: OpenAI-compatible client, runner protocol, mocks."""

from .client import LLMClient, StageRoutedClient, client_for
from .mock import MockRunner, json_responder, sequence
from .runner import ClientRunner, NullRunner, Runner
from .trace import AgentDecision, LLMExchange, TraceRecorder
from .wire import (
    ChatMessage,
    ChatResult,
    ChatUsage,
    LLMError,
    LLMHTTPError,
    LLMParseError,
    estimate_tokens,
    extract_json,
    fit_messages,
    normalise_messages,
    parse_chat_response,
    strip_reasoning,
)

__all__ = [
    "LLMClient",
    "StageRoutedClient",
    "client_for",
    "ClientRunner",
    "NullRunner",
    "Runner",
    "TraceRecorder",
    "LLMExchange",
    "AgentDecision",
    "MockRunner",
    "json_responder",
    "sequence",
    "ChatMessage",
    "ChatResult",
    "ChatUsage",
    "LLMError",
    "LLMHTTPError",
    "LLMParseError",
    "estimate_tokens",
    "extract_json",
    "fit_messages",
    "normalise_messages",
    "parse_chat_response",
    "strip_reasoning",
]
