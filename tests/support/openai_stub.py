"""A throwaway HTTP server that speaks the OpenAI chat-completions dialect.

Why this exists: the app's whole value depends on the custom-provider path
working over the wire, and that path cannot be verified against a real vendor in
a hermetic test. So this stub answers exactly like a real endpoint - including
the awkward behaviours real endpoints have:

* ``response_format`` support is optional (``/novj`` rejects it, like some
  llama.cpp builds do), so the client's parameter-dropping retry is exercised.
* ``stream=true`` returns proper SSE frames.
* ``/models`` returns the OpenAI list shape.
* An endpoint that returns a reasoning-prefixed reply, so reasoning extraction is
  tested against a real response body.
* A 429 on the first request of one route, so retry-after-rate-limit is tested.

Used by ``tests/test_live_provider.py``; not imported by the application.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

#: Sentinel policies per route prefix.
POLICIES = {
    "/v1": {"json_mode": True, "reasoning": False, "rate_limit_first": False},
    "/novj": {"json_mode": False, "reasoning": False, "rate_limit_first": False},
    "/think": {"json_mode": True, "reasoning": True, "rate_limit_first": False},
    "/flaky": {"json_mode": True, "reasoning": False, "rate_limit_first": True},
    "/streamonly": {"json_mode": True, "reasoning": False, "rate_limit_first": False, "require_stream": True},
    # Answers every request with an echo object, like a model that misread the
    # instructions. Used to prove the loop fails in a bounded, diagnosable way.
    "/garbage": {"json_mode": True, "reasoning": False, "rate_limit_first": False, "garbage": True},
    # Enforces the documented OpenAI precondition: json_object mode needs the word
    # "json" in the prompt, and with streaming on it is strict about it. A client
    # that forgets this gets a bare 400 that looks like a server bug.
    # Mimics a reasoning model that spends the whole completion budget on hidden
    # reasoning when the cap is small: content comes back empty, finish_reason is
    # "length", and reasoning_content holds everything. Found live on a real
    # endpoint, and it silently killed a stage.
    "/burner": {
        "json_mode": True,
        "reasoning": False,
        "rate_limit_first": False,
        "burn_below": 8192,
    },
    "/strictjson": {
        "json_mode": True,
        "reasoning": False,
        "rate_limit_first": False,
        "require_json_keyword": True,
    },
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OpenAIStub/1.0"

    # ---------------------------------------------------------------- plumbing
    def log_message(self, *_args) -> None:  # keep test output clean
        pass

    @property
    def policy(self) -> dict[str, Any]:
        for prefix, policy in POLICIES.items():
            if self.path.startswith(prefix):
                return policy
        return POLICIES["/v1"]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.endswith("/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "stub-model-large", "object": "model"},
                        {"id": "stub-model-small", "object": "model"},
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": f"no route for GET {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        policy = self.policy
        body = self._read_body()

        if not body.get("messages"):
            self._send(400, {"error": {"message": "messages is required"}})
            return

        # A server that only supports streaming: any non-stream request is refused.
        if policy.get("require_stream") and not body.get("stream"):
            self._send(
                400,
                {
                    "error": {
                        "message": "this endpoint only supports streaming",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        # The OpenAI json_object precondition, enforced the way real endpoints do.
        if policy.get("require_json_keyword") and body.get("stream"):
            fmt = body.get("response_format") or {}
            if isinstance(fmt, Mapping) and fmt.get("type") == "json_object":
                blob = " ".join(str(m.get("content", "")) for m in body["messages"]).lower()
                if "json" not in blob:
                    self._send(
                        400,
                        {
                            "error": {
                                "message": (
                                    "Prompt must contain the word 'json' in some form to "
                                    "use 'response_format' of type 'json_object'."
                                ),
                                "type": "invalid_request_error",
                                "param": "response_format",
                            }
                        },
                    )
                    return

        # A server that does not implement response_format, like older llama.cpp.
        if not policy["json_mode"] and body.get("response_format") is not None:
            self._send(
                400,
                {
                    "error": {
                        "message": (
                            "Unsupported parameter: 'response_format' is not supported "
                            "by this model."
                        ),
                        "type": "invalid_request_error",
                        "param": "response_format",
                    }
                },
            )
            return

        if policy["rate_limit_first"] and not self.server.saw_rate_limit:  # type: ignore[attr-defined]
            self.server.saw_rate_limit = True  # type: ignore[attr-defined]
            self._send(
                429,
                {"error": {"message": "Rate limit reached, retry shortly.", "type": "rate_limit"}},
            )
            return

        cap = int(body.get("max_tokens") or 0)
        threshold = policy.get("burn_below")
        if threshold and cap and cap < int(threshold):
            self._send(
                200,
                {
                    "id": "chatcmpl-burn",
                    "object": "chat.completion",
                    "model": body.get("model", "stub-model-large"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "weighing the options " * 60,
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": cap,
                        "total_tokens": 40 + cap,
                        "completion_tokens_details": {"reasoning_tokens": cap},
                    },
                },
            )
            return

        prompt = "\n".join(str(m.get("content", "")) for m in body["messages"])
        text = _compose_reply(prompt, policy)

        if body.get("stream"):
            self._stream(text)
            return

        self._send(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": body.get("model", "stub-model-large"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 84, "total_tokens": 126},
            },
        )

    def _stream(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for index in range(0, len(text), 24):
            chunk = text[index : index + 24]
            frame = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


#: A reasoning leak, exactly as real reasoning models emit it inline in
#: ``content``. The app must strip it from the visible answer while keeping it
#: available for debugging - a leaked block would otherwise be parsed as JSON
#: and fail every stage.
REASONING_PREFIX = "".join(["<", "think", ">weighing the options...", "<", "/think", ">", "\n"])

#: Maps a stage's request text to the payload builder that satisfies it.
#: Imported lazily so the stub stays importable without the app installed.
def _stage_payloads() -> dict[str, Callable[[], Any]]:
    from suno_agents.testing.demo import (
        DEMO_EXCLUDE_GOOD,
        _arrangement_payload,
        _blueprint_payload,
        _intake_payload,
        _lyric_payload,
        _prompt_payload,
        _reference_payload,
    )

    return {
        "CreativeBrief": _intake_payload,
        "ReferenceProfile": _reference_payload,
        "MusicBlueprint": _blueprint_payload,
        "LyricSheet": _lyric_payload,
        "Arrangement": lambda: _arrangement_payload(True),
        "SunoPrompt": lambda: _prompt_payload(DEMO_EXCLUDE_GOOD),
    }


def _passing_review(stage: str, dimensions: dict[str, float]) -> dict[str, Any]:
    return {
        "stage": stage,
        "verdict": "pass",
        "passed": True,
        "overall_score": 91.0,
        "dimensions": dimensions,
        "issues": [],
        "restart_from": "",
        "reasoning": "Stub review: the artifact satisfies the brief.",
        "strengths": ["clear front-loading of decisive descriptors"],
    }


def _compose_reply(prompt: str, policy: dict[str, Any]) -> str:
    """Answer like a model would, echoing back enough to be verifiable.

    When the prompt is one of the nine known stage requests, answer with a real,
    gate-passing artifact so an over-the-wire pipeline run exercises the whole
    loop rather than dying on the first parse. Otherwise reply with an echo
    object, which is what a confused model looks like.
    """
    prefix = REASONING_PREFIX if policy["reasoning"] else ""
    lowered = prompt.lower()
    if policy.get("garbage"):
        return prefix + json.dumps(
            {"echo": _echo_of(prompt), "note": "I did not understand the task"}, ensure_ascii=False
        )

    # If the caller sent our JSON-repair instruction, answer with clean JSON only.
    if "only one valid json object" in lowered:
        return prefix + json.dumps(
            {"repaired": True, "note": "fixed the JSON"}, ensure_ascii=False
        )
    if "reply with the single word" in lowered:
        return prefix + "ready"

    # Reviewers first: their prompts describe the artifact they are auditing.
    if "review the lyric sheet" in lowered:
        return prefix + json.dumps(
            _passing_review(
                "lyric_reviewer",
                {
                    "originality": 9.0, "imagery": 9.0, "singability": 9.0,
                    "emotional_truth": 9.0, "hook_strength": 9.0, "language_quality": 9.0,
                    "vocal_fit": 9.0, "brief_fidelity": 9.0,
                },
            ),
            ensure_ascii=False,
        )
    if "audit the complete song" in lowered:
        return prefix + json.dumps(
            _passing_review(
                "music_reviewer",
                {
                    "structure": 9.0, "harmony": 9.0, "melody_contour": 9.0,
                    "groove_and_rhythm": 9.0, "energy_arc": 9.0, "hook_strength": 9.0,
                    "lyrics_integration": 9.0, "character_and_identity": 9.0,
                    "producibility": 9.0, "lossless_readiness": 9.0,
                },
            ),
            ensure_ascii=False,
        )
    if "review the final suno prompt" in lowered:
        return prefix + json.dumps(
            _passing_review(
                "prompt_reviewer",
                {
                    "prompt_fidelity": 9.0, "style_clarity": 9.0, "descriptor_balance": 9.0,
                    "lyrics_format": 9.0, "exclusion_effectiveness": 9.0,
                    "structural_clarity": 9.0, "vocal_direction": 9.0, "title_hook": 9.0,
                    "render_risk": 9.0, "lossless_readiness": 9.0,
                },
            ),
            ensure_ascii=False,
        )

    # Production stages. Each agent frames its own request, so match the actual
    # wording each one uses rather than a single generic phrase.
    stage_markers = (
        ("reply plus the updated brief", "CreativeBrief"),
        ("interpret the measured analysis", "ReferenceProfile"),
        ("no audio reference was supplied", "ReferenceProfile"),
        ("design the complete musical blueprint", "MusicBlueprint"),
        ("write the full lyric set", "LyricSheet"),
        ("brief is instrumental", "LyricSheet"),
        ("map every lyric line onto the bar grid", "Arrangement"),
        ("produce the four suno fields", "SunoPrompt"),
    )
    for marker, artifact in stage_markers:
        if marker in lowered:
            builder = _stage_payloads()[artifact]
            return prefix + json.dumps(builder(), ensure_ascii=False)
    if "produce the " in lowered:
        for name, builder in _stage_payloads().items():
            if f"produce the {name.lower()}" in lowered:
                return prefix + json.dumps(builder(), ensure_ascii=False)

    if "answer" in lowered and "json" in lowered:
        return prefix + json.dumps({"answer": 42, "echo": _echo_of(prompt)}, ensure_ascii=False)
    return prefix + json.dumps({"echo": _echo_of(prompt), "len": len(prompt)}, ensure_ascii=False)


def _echo_of(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip()[:160]


class StubProvider:
    """Run the stub on a background thread; use as a context manager."""

    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.saw_rate_limit = False  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    # ------------------------------------------------------------------ api
    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def base_url(self, prefix: str = "/v1") -> str:
        return f"http://127.0.0.1:{self.port}{prefix}"

    @property
    def rate_limited_once(self) -> bool:
        return bool(getattr(self.httpd, "saw_rate_limit", False))

    def __enter__(self) -> "StubProvider":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


__all__ = ["StubProvider"]
