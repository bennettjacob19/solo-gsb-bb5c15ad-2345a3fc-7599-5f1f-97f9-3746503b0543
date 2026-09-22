"""Rollback-wave preview for a stored release plan.

Given the immutable stored plan and a request listing the services that are
currently live, compute the minimum number of ordered **rollback** waves:

* a service is rolled back strictly *later* than every live service that
  depends on it — downstream comes down before its prerequisites, i.e. the
  release dependency edges are scheduled in reverse;
* the stored per-wave concurrency cap still applies;
* members of one stored fault domain still cannot share a wave, but only the
  live members of each domain constrain the schedule;
* the same two-phase CP-SAT solve as release planning minimizes the wave
  count first and then canonicalizes placement, so identical inputs yield
  identical outputs.

The live set must be closed under the stored "depends-on" relation: a live
service cannot be rolled back while one of its prerequisites is not live.
A violation yields structured, localizable 422 issues naming both the live
service and each missing prerequisite. The stored plan is never modified —
the preview is computed in memory only.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .errors import PlanningError
from .graph import build_adjacency
from .planner import PlanOutput, schedule_minimum_waves
from .schemas import Issue


@dataclass(frozen=True)
class RollbackOutput:
    live_services: list[str]  # validated live set, sorted
    waves: list[list[str]]  # services of each rollback wave, sorted
    wave_count: int
    optimal: bool


def _validate_live_set(
    services: list[str],
    edges: list[tuple[str, str]],
    live_services: list[str],
) -> set[str]:
    """Validate the reported live set against the stored plan.

    Rejects duplicates and services that are not part of the plan, then
    enforces dependency closure: every prerequisite of a live service must
    itself be live. Returns the de-duplicated live set.
    """
    issues: list[Issue] = []

    seen: set[str] = set()
    live: set[str] = set()
    for s in live_services:
        if s in seen:
            issues.append(
                Issue(
                    code="duplicate_service",
                    message=f"Service '{s}' is listed more than once.",
                    constraint="live_services",
                    service=s,
                )
            )
            continue
        seen.add(s)
        live.add(s)

    known = set(services)
    for s in sorted(live - known):
        issues.append(
            Issue(
                code="unknown_service",
                message=(
                    f"Service '{s}' is not part of this plan and cannot be "
                    "reported as live in a rollback preview."
                ),
                constraint="live_services",
                service=s,
            )
        )

    if issues:
        raise PlanningError(
            "The rollback request references unknown services or duplicates.",
            issues,
        )

    adjacency = build_adjacency(services, edges)

    # Dependency closure: a live service requires all of its stored
    # prerequisites to be live too; otherwise a missing prerequisite can be
    # located directly from the issue.
    for service in services:
        if service not in live:
            continue
        missing = sorted(dep for dep in adjacency[service] if dep not in live)
        for dep in missing:
            issues.append(
                Issue(
                    code="rollback_closure_violation",
                    message=(
                        f"Service '{service}' is reported live, but its "
                        f"prerequisite '{dep}' is not; the live set must be "
                        "closed under dependencies so every prerequisite of "
                        "a live service is live as well."
                    ),
                    constraint="live_services",
                    service=dep,
                    related_services=[service],
                    context={
                        "live_service": service,
                        "missing_prerequisite": dep,
                    },
                )
            )

    if issues:
        raise PlanningError(
            "The reported live set violates dependency closure.",
            issues,
        )

    return live


def preview_rollback(
    services: list[str],
    edges: list[tuple[str, str]],
    fault_domains: list[dict],
    concurrency: int,
    live_services: list[str],
    settings: Settings,
) -> RollbackOutput:
    """Compute minimum rollback waves for the reported live set.

    ``services`` lists every service of the stored plan, ``edges`` its
    ``(service, depends_on)`` dependencies and ``fault_domains`` the stored
    ``{"domain", "services"}`` records. The stored plan is never modified.
    """
    live = _validate_live_set(services, edges, live_services)
    live_sorted = sorted(live)

    if not live_sorted:
        return RollbackOutput(
            live_services=[], waves=[], wave_count=0, optimal=True
        )

    # Reverse the release ordering for rollback: a prerequisite goes in a
    # strictly later wave than each of its live downstreams. Edges touching a
    # non-live endpoint are dropped — only live services are being rolled back.
    reverse_edges = [
        (dep, service)
        for service, dep in edges
        if service in live and dep in live
    ]

    # Anti-affinity applies only to the live members of each stored domain:
    # a clique with fewer than two live members cannot constrain anything.
    fault_cliques: list[list[str]] = []
    for fd in fault_domains:
        members = sorted(m for m in fd["services"] if m in live)
        if len(members) >= 2:
            fault_cliques.append(members)

    output: PlanOutput = schedule_minimum_waves(
        live_sorted, reverse_edges, fault_cliques, concurrency, settings
    )
    return RollbackOutput(
        live_services=live_sorted,
        waves=output.waves,
        wave_count=output.wave_count,
        optimal=output.optimal,
    )
