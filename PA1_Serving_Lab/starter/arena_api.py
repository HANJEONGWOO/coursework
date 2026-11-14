"""Read-only observations available to a competition policy.

Start at OnlineSnapshot, then inspect OnlineRequest and CompletedRequest.
Frozen dataclasses reject field reassignment. Tuples hold fixed collections.
Active requests hide their final output length. Completed results describe
events that already happened and may reveal the observed output count.
RecentStats is a convenience window, not the class-balanced competition score.
"""

from dataclasses import dataclass

RECENT_WINDOW_SIZE = 32


@dataclass(frozen=True)
class CompletedRequest:
    """Observed results, published only after the request has completed."""

    id: str
    arrival: float
    completed_at: float
    output_tokens: int
    ttft: float
    mean_tpot: float
    p95_itl: float
    max_itl: float
    ttft_slo: float
    itl_slo: float
    slo_met: bool


@dataclass(frozen=True)
class RecentStats:
    """Last 32 completions, equally weighted by request. Empty means None."""

    count: int = 0
    mean_ttft: float | None = None
    mean_tpot: float | None = None
    mean_p95_itl: float | None = None
    ttft_miss_rate: float | None = None
    itl_miss_rate: float | None = None
    slo_attainment: float | None = None


def summarize_recent(completed):
    """The caller supplies the bounded completion window, in completion order."""
    if not completed:
        return RecentStats()
    n = len(completed)
    return RecentStats(
        count=n,
        mean_ttft=sum(r.ttft for r in completed) / n,
        mean_tpot=sum(r.mean_tpot for r in completed) / n,
        mean_p95_itl=sum(r.p95_itl for r in completed) / n,
        ttft_miss_rate=sum(r.ttft > r.ttft_slo for r in completed) / n,
        itl_miss_rate=sum(r.p95_itl > r.itl_slo for r in completed) / n,
        slo_attainment=sum(r.slo_met for r in completed) / n,
    )


@dataclass(frozen=True)
class OnlineRequest:
    """An active request visible to plan_custom, with no future output length."""

    id: str
    prompt: int
    remaining: int
    generated: int
    order: int
    arrival: float
    last_output_time: float | None
    ttft_slo: float
    itl_slo: float

    @property
    def phase(self):
        """Identify which kind of action this active request can accept."""
        return "prefill" if self.remaining else "decode"


@dataclass(frozen=True)
class OnlineSnapshot:
    """Inputs to one competition planner call. No future arrivals are included.

    completed_since_last_call delivers each completion once at the next call.
    recent_stats retains up to 32 completions even when the new tuple is empty.
    There is no extra call after the case ends. Defaults support empty history.
    """

    requests: tuple[OnlineRequest, ...]
    now: float
    waiting_count: int
    completed_since_last_call: tuple[CompletedRequest, ...] = ()
    recent_stats: RecentStats = RecentStats()
