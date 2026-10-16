"""Agent 1 - the intake consultant.

Two jobs in one call: hold the conversation in the user's own language, and
maintain the structured brief behind it. A non-musician must be able to answer
"what do you want to hear" and end up with a real specification, without ever
being asked for a genre, a key, or a BPM.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..domain.graph import INTAKE
from ..domain.models import CreativeBrief, as_bool, as_str_list
from .base import Agent
from .prompts import (
    CONSTITUTION,
    JSON_DISCIPLINE,
    SCHEMA_FIELD_NAMES,
    NEED_REFERENCE,
    build_system,
    context_block,
    json_schema_hint,
    revision_block,
)
from .schemas import INTAKE_SCHEMA

INTAKE_CRAFT = """\
YOUR CRAFT: CONSULTING A NON-MUSICIAN
- You are a producer with 20 years in the room. The user is not. Never ask for
  "genre", "key", "tempo", or "arrangement" as if they should know. Ask in
  feelings, references, and scenes, then translate.
- Ask at most 2 questions per turn, and always attach 2-4 concrete, opinionated
  choices so the user only has to pick a number or say "the second one".
  Example: "Mau terasa seperti apa di bagian reff? (1) lega setelah lama
  bertahan (2) marah tapi dingin (3) jatuh cinta lagi".
- Interpret vague words as musical decisions. "Sedih tapi semangat" = minor key,
  100-118 BPM, driving kick, major lift in the chorus. Say your interpretation
  back in one plain-language line so the user can correct you cheaply.
- Detect and honour references: "kayak lagu X" or "vibes band Y" -> translate
  into era, instrumentation, vocal type, and production texture - never copy the
  artist's name into the deliverable.
- Mirror the user's language in `reply` (Indonesian in, Indonesian out; mixed
  language in, match the dominant one). Keep JSON field names in English.
- Stop asking as soon as you can defend the choices. A brief with a genre, a
  mood, a vocal direction, a tempo range, and a story is buildable. Say so in
  `ready_to_build`, summarise the plan in 3-5 plain lines, and end with what
  happens next ("aku akan mulai menyusun nadanya").
- Never invent a user preference: if it is not stated, propose it and label the
  field as your proposal in `reply`.
- If the user attached audio references, the measured analysis arrives in a
  separate block. Use the numbers, quote at most two, and do not lecture.

HARD RULES
- `reply` is displayed verbatim to a human: no JSON, no markdown tables, no
  stage directions, max ~140 words, warm but efficient.
- `brief` must be COMPLETE on every turn: carry forward everything decided
  earlier, then apply the newest answers. Never drop a field you already filled.
- Keep `open_questions` for unknowns you are deliberately leaving to the team;
  list them only when they matter.
- Set `confidence` to how sure you are that the brief is buildable (0-1).
"""


class IntakeAgent(Agent):
    stage = INTAKE
    schema = INTAKE_SCHEMA

    def needs(self, state) -> Sequence[str]:
        return (NEED_REFERENCE,)

    def system_prompt(self, state, revision_notes: Sequence[str] = ()) -> str:
        previous = state.brief
        carried = ""
        if previous is not None:
            carried = (
                "=== BRIEF SO FAR (carry every filled field forward) ===\n"
                + previous.summary()
            )
        expertise = getattr(previous, "user_expertise", "non-musician") if previous else "non-musician"
        return build_system(
            CONSTITUTION,
            INTAKE_CRAFT,
            JSON_DISCIPLINE,
            SCHEMA_FIELD_NAMES,
            json_schema_hint(INTAKE_SCHEMA),
            carried,
            context_block(state, self.needs(state)),
            f"User expertise level: {expertise}",
            revision_block(revision_notes, self.stage),
        )

    def user_message(self, state, revision_notes: Sequence[str] = ()) -> str:
        return (
            "Continue the intake conversation. Read the transcript above, reply to "
            "the user's latest message, and return the reply plus the updated brief."
        )

    def parse(self, payload: Any, state) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Intake reply must be a JSON object.")
        reply = str(payload.get("reply", "") or "").strip()
        if not reply:
            raise ValueError("Field 'reply' is required and was empty.")
        brief_payload = payload.get("brief")
        if not isinstance(brief_payload, Mapping):
            # Tolerate models that flatten the brief into the root object.
            brief_payload = {
                key: value
                for key, value in payload.items()
                if key
                not in {"reply", "ready_to_build", "missing_information", "suggested_choices"}
            }
        brief = CreativeBrief.from_dict(brief_payload)
        if state.brief is not None:
            brief = _merge_brief(state.brief, brief)
        if not brief.language:
            brief.language = state.brief.language if state.brief else "id"
        return {
            "reply": reply,
            "ready_to_build": as_bool(payload.get("ready_to_build"), False),
            "brief": brief,
            "missing_information": as_str_list(payload.get("missing_information")),
            "suggested_choices": as_str_list(payload.get("suggested_choices")),
        }

    def validate(self, artifact: Any) -> list[str]:
        problems = list(artifact["brief"].validate())
        # An incomplete brief is only a problem when the model claims it is ready.
        if artifact["ready_to_build"]:
            blocking = [
                p for p in problems if "no concept" in p.lower() or "neither genre" in p.lower()
            ]
            return blocking
        return [p for p in problems if "outside 40-220" in p]


def _merge_brief(previous: CreativeBrief, fresh: CreativeBrief) -> CreativeBrief:
    """Never let a later turn erase a decision made earlier."""
    merged = CreativeBrief.from_dict(previous.to_dict())
    fresh_data = fresh.to_dict()
    previous_data = previous.to_dict()
    for key, value in fresh_data.items():
        if isinstance(value, list):
            if value:
                merged.__dict__[key] = value
        elif isinstance(value, bool):
            # Booleans are only adopted when the fresh turn actually asserted them.
            if value or not previous_data.get(key):
                merged.__dict__[key] = value
        elif isinstance(value, (int, float)):
            if value:
                merged.__dict__[key] = value
        elif value:
            merged.__dict__[key] = value
    return merged


__all__ = ["IntakeAgent"]
