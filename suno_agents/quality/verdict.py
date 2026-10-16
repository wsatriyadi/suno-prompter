"""Turning a model's opinion plus deterministic gates into one binding verdict.

Order of authority, highest first:

1. **Hard gates.** A failed gate is a fact; the reviewer cannot override it.
2. **Dimension floor.** No dimension may sit below ``DIMENSION_FLOOR`` on a pass.
3. **Pass mark.** The weighted score must clear the stage's pass mark.
4. **The reviewer's taste.** It fills in scores and the issues gates cannot see.

The rollback target is derived here, not taken on faith: gate failures name the
stage that must own the fix, the reviewer's own ``restart_from`` is considered,
and the earliest legal stage wins because downstream defects root upstream. The
result is then clamped through :func:`suno_agents.domain.graph.resolve_restart`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..domain.graph import is_reviewer, legal_restarts, resolve_restart, stage_index
from ..domain.models import ReviewIssue, ReviewVerdict
from .gates import (
    ACCEPTANCE_MIN_ATTEMPT,
    ACCEPTANCE_SCORES,
    DIMENSION_FLOOR,
    GateResult,
    is_soft,
)

#: Weighted score = llm_weight * model score + gate_weight * gate score.
LLM_WEIGHT = 0.62
GATE_WEIGHT = 0.38

#: The canonical rule lives on the gate result: see
#: :meth:`suno_agents.quality.gates.GateResult.soft` and
#: :func:`suno_agents.quality.gates.is_soft`.

#: Below this the run is considered unsalvageable by review alone.
REJECT_FLOOR = 55.0

SEVERITY_WEIGHT = {"blocker": 1.0, "major": 0.6, "minor": 0.25, "polish": 0.1}


def gate_score(results: Sequence[GateResult]) -> float:
    """0-100 score from hard-gate outcomes, weighted by severity.

    Soft gates are excluded, because their cost is applied once as an explicit
    penalty in :func:`finalise_review`. Counting them here as well would deduct
    twice and, as a live run showed, could push a shippable 83.4 below a pass mark
    of 80 through arithmetic rather than judgement.
    """
    scored = [r for r in results if not is_soft(r)]
    if not scored:
        return 100.0
    total = 0.0
    earned = 0.0
    for result in scored:
        weight = SEVERITY_WEIGHT.get(result.severity, 0.5)
        total += weight
        if result.ok:
            earned += weight
    return round(100.0 * earned / total, 2) if total else 100.0


def finalise_review(
    stage: str,
    llm: ReviewVerdict | None,
    gates: Sequence[GateResult],
    *,
    pass_mark: float,
    attempt: int = 0,
    dimension_floor: float = DIMENSION_FLOOR,
) -> ReviewVerdict:
    """Merge machine gates and the model's review into the binding verdict."""
    assert is_reviewer(stage), f"{stage} is not a reviewer stage"
    all_failures = [g for g in gates if not g.ok]
    failures = [g for g in all_failures if not is_soft(g)]
    soft_failures = [g for g in all_failures if is_soft(g)]
    llm = llm or ReviewVerdict(stage=stage)

    # 1. fold gate failures in as issues, ahead of taste issues
    issues: list[ReviewIssue] = [g.to_issue() for g in all_failures]
    seen: set[tuple[str, str]] = {(i.code, i.message[:60]) for i in issues}
    for issue in llm.sorted_issues():
        key = (issue.code, issue.message[:60])
        if key in seen:
            continue
        seen.add(key)
        issues.append(issue)

    # 2. score
    model_score = _model_score(llm)
    machine_score = gate_score(gates)
    overall = round(LLM_WEIGHT * model_score + GATE_WEIGHT * machine_score, 2)

    # 3. dimension floor check
    # Only real submitted scores can fail a floor. When `dimensions` is absent the
    # review is scored from `overall_score` alone; inventing zeros here used to
    # manufacture a floor failure for every dimension at once.
    weak = {
        name: value
        for name, value in llm.dimensions.items()
        if isinstance(value, (int, float)) and value < dimension_floor
    }
    for name, value in weak.items():
        issues.append(
            ReviewIssue(
                code=f"LOW_{name.upper()}",
                severity="major",
                dimension=name,
                message=f"Dimension '{name}' scored {value:.1f}/10, below the {dimension_floor:.0f} floor.",
                fix=f"Rework the material with {name.replace('_', ' ')} as the priority.",
            )
        )

    # 4. decide. A soft gate never blocks, but it must cost score so a reviewer
    #    that ignores a real quality gap cannot pass on a technicality.
    penalty = min(6.0, 2.5 * len(soft_failures))
    if penalty:
        overall = max(0.0, round(overall - penalty, 2))
    gates_clean = not failures
    dimensions_clean = not weak
    model_passed = llm.verdict == "pass"
    passed = bool(gates_clean and dimensions_clean and model_passed and overall >= pass_mark)

    accepted_by_score = False
    if not passed and gates_clean and dimensions_clean:
        acceptance_mark = ACCEPTANCE_SCORES.get(stage, pass_mark + 4.0)
        if attempt >= ACCEPTANCE_MIN_ATTEMPT and overall >= acceptance_mark:
            passed = True
            accepted_by_score = True

    if passed:
        verdict = "pass"
    elif overall < REJECT_FLOOR and attempt >= 2:
        verdict = "reject"
    else:
        verdict = "revise"

    # 5. rollback target, by authority rather than by hule. See _select_restart.
    legal = legal_restarts(stage)
    restart = (
        ""
        if passed
        else _select_restart(
            gate_targets=[g.restart_from for g in failures if g.restart_from],
            verdict_target=llm.restart_from,
            blocker_targets=[i.restart_from for i in issues if i.severity == "blocker" and i.restart_from],
            other_targets=[i.restart_from for i in issues if i.severity != "blocker" and i.restart_from],
            legal=legal,
        )
    )

    reasoning = llm.reasoning.strip()
    gate_note = ""
    if failures:
        gate_note = (
            f"\n\n[Hard gates failed: {len(failures)}] "
            + "; ".join(f"{g.code}: {g.message}" for g in failures[:4])
        )
    if soft_failures:
        gate_note += (
            f"\n\n[Soft gates: {len(soft_failures)}, -{penalty:.1f} score] "
            + "; ".join(f"{g.code}: {g.message}" for g in soft_failures[:3])
        )
    if accepted_by_score:
        gate_note += (
            f"\n\n[Accepted on measured quality] The reviewer still asked for "
            f"revision, but every hard gate passed, no scored dimension fell below "
            f"{dimension_floor:.0f}, and the score {overall:.1f} cleared the acceptance "
            f"mark {acceptance_mark:.0f} after {attempt + 1} passes. Shipping this "
            "version; its remaining notes are recorded above as improvements."
        )
    score_note = (
        f"\n\n[Scores] model={model_score:.1f} ({_score_source(llm)}) "
        f"gates={machine_score:.1f} weighted={overall:.1f} "
        f"pass_mark={pass_mark:.0f} attempt={attempt + 1}"
    )

    return ReviewVerdict(
        stage=stage,
        passed=passed,
        verdict=verdict,
        overall_score=overall,
        dimensions=dict(llm.dimensions),
        gates_failed=[g.code for g in failures],
        issues=issues,
        restart_from=restart,
        reasoning=reasoning + gate_note + score_note,
        strengths=list(llm.strengths),
        raw=dict(llm.raw),
    )


def _score_source(llm: ReviewVerdict) -> str:
    """Say where the model's number came from, so a review is auditable.

    The distinction matters in practice: a reviewer that filled in its score sheet
    and one that only gave a headline number are not equally trustworthy, and the
    person reading the report deserves to know which they got.
    """
    if llm.dimensions:
        counted = sum(1 for v in llm.dimensions.values() if v)
        return f"dimensions x{counted}" if counted else "overall only"
    return "overall only"


def _model_score(llm: ReviewVerdict) -> float:
    """Prefer explicit overall_score; otherwise average the dimensions x10."""
    if llm.overall_score and 0 < llm.overall_score <= 100:
        return float(llm.overall_score)
    dims = [
        float(v)
        for v in llm.dimensions.values()
        if isinstance(v, (int, float)) and 0 <= float(v) <= 10
    ]
    if dims:
        return round(sum(dims) / len(dims) * 10.0, 2)
    return 0.0


def _select_restart(
    *,
    gate_targets: Sequence[str],
    verdict_target: str,
    blocker_targets: Sequence[str],
    other_targets: Sequence[str],
    legal: Sequence[str],
) -> str:
    """Choose the rollback target by authority, not by taking the most upstream.

    The hierarchy exists because the inputs are not equally trustworthy:

    1. **Gate failures** are measured facts, so the earliest gate-named stage wins
       - a real defect upstream invalidates everything downstream, and fixing the
       downstream artifact would just reproduce it.
    2. **The reviewer's consolidated ``restart_from``** is its deliberate
       diagnosis. It beats individual issues, because a single contrarian issue
       must not drag a lyricist problem back to the brief and throw away the
       composer's work along with it.
    3. **Blocker-severity issues** may escalate one level beyond the consolidated
       verdict, since a blocker genuinely blocks and usually roots upstream.
    4. **Non-blocker issues are ignored for routing.** They are advisory notes; the
       reviewer already accounted for them when it named its target.

    Returns '' when nothing usable was named, which tells the orchestrator to run
    its own escalation ladder instead of pinning every failure to one target.
    """
    if not legal:
        return ""
    resolved = _resolve_targets(gate_targets, legal)
    if resolved:
        return min(resolved, key=stage_index)

    if verdict_target:
        target = resolve_restart_for(verdict_target, legal)
        if target:
            return target
        # Named but untranslatable (a garbled stage id): clamp, do not escalate.
        return legal[0]

    blockers = _resolve_targets(blocker_targets, legal)
    if blockers:
        return min(blockers, key=stage_index)

    named_others = [str(t).strip() for t in other_targets if str(t or "").strip()]
    if named_others:
        # Only non-blocker hints exist. A majority vote is the honest reading of
        # several advisory notes, and it still respects the legal set.
        votes: dict[str, int] = {}
        for candidate in named_others:
            target = resolve_restart_for(candidate, legal)
            if target:
                votes[target] = votes.get(target, 0) + 1
        if votes:
            best = max(votes.items(), key=lambda kv: (kv[1], -stage_index(kv[0])))
            return best[0]
        return legal[0]
    return ""


def _resolve_targets(candidates: Sequence[str], legal: Sequence[str]) -> list[str]:
    """Resolve a list of requested targets into the legal ones."""
    resolved: list[str] = []
    for candidate in candidates:
        target = resolve_restart_for(str(candidate or ""), legal)
        if target:
            resolved.append(target)
    return resolved


def resolve_restart_for(candidate: str, legal: Sequence[str]) -> str:
    """Clamp a single requested target into the legal set ('' when impossible)."""
    for option in (candidate,):
        if not option:
            return ""
        from ..domain.graph import normalise_restart_name

        name = normalise_restart_name(option)
        if name and name in legal:
            return name
    return ""


def apply_explicit_restart(stage: str, verdict: ReviewVerdict, requested: str) -> ReviewVerdict:
    """Force a specific legal restart target onto a verdict (used by the UI)."""
    legal = legal_restarts(stage)
    target = resolve_restart(stage, requested)
    if not target or (legal and target not in legal):
        return verdict
    verdict.restart_from = target
    verdict.reasoning += f"\n\n[Restart target set to '{target}' by the user.]"
    return verdict


def revision_notes_from(
    verdict: ReviewVerdict,
    *,
    for_stage: str,
    history: Sequence[ReviewVerdict] = (),
) -> list[str]:
    """Actionable instructions for whichever stage is about to be re-run.

    ``history`` is every earlier verdict in the run. It is consulted for one
    specific failure mode: a stage that is being sent back for a problem it has
    already been told about. Without saying so, the rewrite tends to arrive with the
    same defect and the loop repeats - the model has no way to know that its previous
    interpretation of the note was rejected, because it never saw that they were
    connected.
    """
    notes: list[str] = []
    notes.append(
        f"Reviewer verdict: {verdict.verdict} (score {verdict.overall_score:.1f}/100). "
        f"Restart target: {verdict.restart_from or 'n/a'}."
    )
    if verdict.gates_failed:
        notes.append("Hard gate failures that MUST be fixed: " + ", ".join(verdict.gates_failed))
    repeats = _repeated_failures(for_stage, verdict, history)
    if repeats:
        notes.append(
            f"THIS IS A REPEAT. You have already been asked about "
            f"{len(repeats)} of these and the problem SURVIVED your previous rewrite: "
            + ", ".join(repeats)
            + ". Do not make another cosmetic change to those points. Change the "
            "approach that produced them: if the note asked for shorter lines, cut "
            "content rather than re-punctuating; if it asked for a lock on the vocal, "
            "write it explicitly instead of implying it."
        )
    for issue in verdict.sorted_issues()[:10]:
        line = f"[{issue.severity}] {issue.message}"
        if issue.fix:
            line += f" -> {issue.fix}"
        if issue.restart_from and issue.restart_from != for_stage:
            line += f" (root cause originates at '{issue.restart_from}')"
        notes.append(line)
    if verdict.strengths:
        notes.append("Keep these strengths: " + "; ".join(verdict.strengths[:5]))
    return notes


def _repeated_failures(
    for_stage: str,
    verdict: ReviewVerdict,
    history: Sequence[ReviewVerdict],
) -> list[str]:
    """Defects that were already reported and are still being reported.

    Matched on the machine-readable keys (failed gate codes and issue codes), never
    on prose: two attempts at describing the same defect rarely produce identical
    sentences, but they do produce the same gate code.
    """
    if not history:
        return []
    current = failure_keys(verdict)
    if not current:
        return []
    prior: dict[str, int] = {}
    for earlier in history:
        if earlier is verdict:
            continue
        # Only count earlier attempts at the same stage, so a fresh problem reported
        # after an upstream change is not mislabelled as a repeat.
        if earlier.restart_from and earlier.restart_from != for_stage:
            continue
        for key in failure_keys(earlier):
            prior[key] = prior.get(key, 0) + 1
    return sorted(key for key in current if prior.get(key))


def failure_keys(verdict: ReviewVerdict) -> set[str]:
    """Stable identifiers for what a verdict complained about."""
    keys = {f"gate:{code}" for code in verdict.gates_failed}
    for issue in verdict.issues:
        if issue.severity in {"blocker", "major"}:
            if issue.code:
                keys.add(f"issue:{issue.code}")
            elif issue.dimension:
                keys.add(f"dimension:{issue.dimension}")
    return keys


__all__ = [
    "LLM_WEIGHT",
    "GATE_WEIGHT",
    "is_soft",
    "REJECT_FLOOR",
    "finalise_review",
    "gate_score",
    "apply_explicit_restart",
    "failure_keys",
    "revision_notes_from",
]
