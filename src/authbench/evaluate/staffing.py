"""How many analysts a daily alert budget stands for (S6-29).

A budget is only meaningful against the people who would work it. The model is deliberately
small and every input is published with its source and its weakness:

- **Triage time per analyst-day: 2.7 hours.** Vectra (2023), a survey of 2,000 IT security
  analysts at organisations of more than 1,000 employees, run by Sapio Research in
  March-April 2023: "Analysts spend nearly 3 hours (2.7) each day manually triaging alerts".
  Self-reported, vendor-commissioned.
- **Minutes per alert: 10 to 20, 5 to 30 as a sensitivity range.** Li et al. (2026, section
  4.5, figure 14c): half of 10 practitioners of one SOC answered 10 to 20 minutes to
  investigate an alert. Self-reported, small, one SOC, a preprint. Published industry figures
  run from about 5 to 40 minutes depending on what "triage" covers.

The same survey reports 4,484 alerts a day per SOC, of which analysts could not deal with
67%: about 1,480 handled a day across every tool a SOC runs, the ceiling for any one detector.
"""

from __future__ import annotations

from dataclasses import dataclass

TRIAGE_MINUTES_PER_ANALYST_DAY = 2.7 * 60
MINUTES_PER_ALERT = {"low": 5.0, "central_low": 10.0, "central_high": 20.0, "high": 30.0}
SOC_ALERTS_HANDLED_PER_DAY = 4484 * (1 - 0.67)


@dataclass(frozen=True)
class Staffing:
    budget: int
    #: Analysts whose daily triage time the budget fills, from the fastest to the slowest triage.
    analysts_fastest: float
    analysts_central: tuple[float, float]
    analysts_slowest: float
    #: Share of the alerts a typical SOC handles a day across all of its tools.
    share_of_soc_capacity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "analysts_at_5_min": round(self.analysts_fastest, 2),
            "analysts_at_10_to_20_min": [round(a, 2) for a in self.analysts_central],
            "analysts_at_30_min": round(self.analysts_slowest, 2),
            "share_of_soc_daily_capacity": round(self.share_of_soc_capacity, 4),
        }


def staffing(budget: int) -> Staffing:
    """Analysts needed to triage `budget` alerts a day, under each minutes-per-alert assumption."""

    def analysts(minutes: float) -> float:
        return budget * minutes / TRIAGE_MINUTES_PER_ANALYST_DAY

    return Staffing(
        budget=budget,
        analysts_fastest=analysts(MINUTES_PER_ALERT["low"]),
        analysts_central=(
            analysts(MINUTES_PER_ALERT["central_low"]),
            analysts(MINUTES_PER_ALERT["central_high"]),
        ),
        analysts_slowest=analysts(MINUTES_PER_ALERT["high"]),
        share_of_soc_capacity=budget / SOC_ALERTS_HANDLED_PER_DAY,
    )


def staffing_table(budgets: list[int]) -> list[dict[str, object]]:
    return [staffing(k).to_dict() for k in budgets]
