"""Pydantic request/response models.

Fault-domain semantics (important): every entry in ``fault_domains`` is an
*anti-affinity* zone. Two services sharing the same zone id must NOT be
released in the same wave, so listing one zone per physical fault domain
expresses "services in this domain cannot co-locate in a wave".
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---- Request models --------------------------------------------------------


class Dependency(BaseModel):
    service: str = Field(..., min_length=1, description="Dependending service")
    depends_on: str = Field(..., min_length=1, description="Required service")

    @field_validator("service", "depends_on")
    @classmethod
    def _strip_and_reject_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("service id must not be blank")
        return v


class FaultDomain(BaseModel):
    domain: str = Field(
        ...,
        min_length=1,
        description="Fault-domain / zone id; members of one domain cannot "
        "share a wave",
    )
    services: list[str] = Field(..., min_length=1)

    @field_validator("domain")
    @classmethod
    def _strip_domain(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("domain id must not be blank")
        return v

    @field_validator("services")
    @classmethod
    def _strip_services(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


class PlanCreate(BaseModel):
    name: str | None = Field(
        default=None,
        max_length=200,
        description="Optional human-readable label",
    )
    services: list[str] = Field(..., min_length=1)
    dependencies: list[Dependency] = Field(default_factory=list)
    """service -> depends_on edges; a dependency releases in an earlier wave."""
    max_concurrency: int = Field(..., ge=1, description="Services per wave cap")
    fault_domains: list[FaultDomain] = Field(default_factory=list)
    max_waves: int | None = Field(
        default=None,
        ge=1,
        description="Optional release-window cap; error 422 with reasons if "
        "the optimum needs more waves",
    )

    @field_validator("services", "name")
    @classmethod
    def _strip_service_ids(cls, v: Any) -> Any:
        if isinstance(v, list):
            cleaned = [s.strip() for s in v]
            if any(not s for s in cleaned):
                raise ValueError("service id must not be blank")
            return cleaned
        if isinstance(v, str):
            return v.strip()
        return v


class PlanSimulate(BaseModel):
    """What-if constraint changes applied on top of a stored plan.

    Nothing is persisted. Removals must name constraints that exist in the
    stored plan; additions create new ones. To modify a fault domain, remove
    its id and re-add it with the new member list in the same request. The
    stored ``max_waves`` window is not applied: the simulation reports the
    resulting wave-count delta instead.
    """
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description="New per-wave concurrency cap; omit to keep the plan's value",
    )
    add_dependencies: list[Dependency] = Field(default_factory=list)
    remove_dependencies: list[Dependency] = Field(default_factory=list)
    add_fault_domains: list[FaultDomain] = Field(default_factory=list)
    remove_fault_domains: list[str] = Field(
        default_factory=list, description="Ids of fault domains to drop"
    )

    @field_validator("remove_fault_domains")
    @classmethod
    def _strip_domain_ids(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("domain id must not be blank")
        return cleaned


class RollbackPreviewRequest(BaseModel):
    """Live services reported for a rollback preview.

    ``live_services`` lists the services currently live. Rollback waves are
    generated only for those services, in reverse dependency order
    (downstream first), under the stored plan's per-wave concurrency cap and
    fault-domain anti-affinity. The set must be closed under the stored
    dependencies — otherwise the response is a localizable 422. The stored
    plan itself is never modified.
    """

    live_services: list[str] = Field(
        ...,
        description="Services currently live; rollback waves cover exactly these",
    )

    @field_validator("live_services")
    @classmethod
    def _strip_service_ids(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


class GateRequest(BaseModel):
    """Execution progress reported against a stored plan.

    ``completed_services`` have already finished successfully and
    ``failed_services`` have failed in this rollout. The gate decides which
    of the remaining services can still be executed and which are blocked,
    strictly following the stored wave assignment — the plan itself is never
    modified.
    """

    completed_services: list[str] = Field(default_factory=list)
    failed_services: list[str] = Field(default_factory=list)

    @field_validator("completed_services", "failed_services")
    @classmethod
    def _strip_service_ids(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


# ---- Response models -------------------------------------------------------


class Wave(BaseModel):
    index: int = Field(..., ge=1, description="1-based wave number")
    services: list[str] = Field(..., description="Sorted service ids in wave")


class Issue(BaseModel):
    """A single, localizable reason that makes planning impossible."""

    code: str
    message: str
    constraint: str
    service: str | None = None
    related_services: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    id: int
    name: str | None
    services: list[str]
    waves: list[Wave]
    wave_count: int
    max_concurrency: int
    max_waves: int | None
    dependencies: list[dict[str, str]]
    fault_domains: list[dict[str, Any]]
    optimal: bool
    created_at: str


class PlanSummary(BaseModel):
    id: int
    name: str | None
    wave_count: int
    service_count: int
    max_concurrency: int
    optimal: bool
    created_at: str


class PlanList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PlanSummary]


class PlanDeleted(BaseModel):
    id: int
    deleted: bool = True


class ServiceWaveChange(BaseModel):
    service: str
    from_wave: int = Field(..., ge=1, description="Wave in the stored plan (1-based)")
    to_wave: int = Field(..., ge=1, description="Wave in the simulated plan (1-based)")


class PlanSimulation(BaseModel):
    """Outcome of a what-if simulation; never persisted."""

    plan_id: int
    name: str | None
    waves: list[Wave]
    wave_count: int
    original_wave_count: int
    wave_delta: int = Field(
        ..., description="wave_count - original_wave_count (positive: more waves)"
    )
    max_concurrency: int = Field(..., description="Effective per-wave cap used")
    optimal: bool = Field(
        ...,
        description="True when both the wave count and the number of moved "
        "services are proven minimal",
    )
    assignments: list[ServiceWaveChange] = Field(
        ..., description="Every service's old and new wave, sorted by service id"
    )
    moved_services: list[str] = Field(
        ..., description="Services whose wave changed, sorted"
    )
    affected_services: list[str] = Field(
        ...,
        description="Services directly referenced by newly added constraints, "
        "sorted",
    )


class GateBlockedItem(BaseModel):
    """A not-yet-reported service that cannot be executed right now."""

    service: str
    wave: int = Field(..., ge=1, description="Wave of the service in the stored plan")
    reasons: list[Issue] = Field(
        ..., description="Localizable blocking reasons, one per offending dependency"
    )


class GateWaveStatus(BaseModel):
    """Per-wave breakdown of the gate decision (stored plan unchanged)."""

    index: int = Field(..., ge=1)
    executable: list[str] = Field(..., description="Executable members, sorted")
    blocked: list[str] = Field(..., description="Blocked members, sorted")
    completed: list[str] = Field(..., description="Reported-success members, sorted")
    failed: list[str] = Field(..., description="Reported-failed members, sorted")


class GateResult(BaseModel):
    """Outcome of the execution gate for one rollout progress report."""

    plan_id: int
    name: str | None
    wave_count: int = Field(..., description="Stored plan wave count (unchanged)")
    current_wave: int | None = Field(
        ...,
        description="Earliest stored wave that still has an executable service; "
        "null when every service is completed or blocked",
    )
    executable_services: list[str] = Field(
        ...,
        description="Remaining services whose dependencies are all completed, "
        "grouped by wave then sorted by service id",
    )
    blocked_services: list[GateBlockedItem] = Field(
        ...,
        description="Blocked remaining services and their localizable reasons; "
        "empty when nothing is blocked",
    )
    completed_services: list[str] = Field(..., description="Reported successes, sorted")
    failed_services: list[str] = Field(..., description="Reported failures, sorted")
    waves: list[GateWaveStatus] = Field(
        ..., description="Executable/blocked/completed/failed split for every stored wave"
    )


class RollbackPreviewResult(BaseModel):
    """Minimum-wave rollback schedule for the reported live set; never persisted."""

    plan_id: int
    name: str | None
    live_services: list[str] = Field(
        ..., description="Validated, de-duplicated live set the waves cover, sorted"
    )
    waves: list[Wave] = Field(
        ...,
        description="Rollback waves; wave 1 is rolled back first and contains "
        "the (most downstream) live services nothing live depends on",
    )
    wave_count: int = Field(..., description="Number of rollback waves (0 for an empty live set)")
    max_concurrency: int = Field(
        ..., description="Stored plan's per-wave cap, still applied to rollback waves"
    )
    optimal: bool = Field(
        ...,
        description="True when the rollback wave count is proven minimal",
    )


# ---- Error envelope --------------------------------------------------------


class ErrorBody(BaseModel):
    error: Literal["planning_failed", "not_found", "validation_error", "internal_error"]
    message: str
    issues: list[Issue] = Field(default_factory=list)
