"""Human-readable alert cards (US-130) — what a SOC analyst reads instead of model internals."""

from __future__ import annotations

from dataclasses import dataclass, field

from authbench.explain.shap_wrap import FeatureContribution


@dataclass
class AlertCard:
    event_id: int
    src_user: str
    src_computer: str
    dst_computer: str
    time: int
    score: float
    rank_in_day: int
    top_features: list[FeatureContribution]
    mitre_techniques: list[str] = field(default_factory=list)

    def render_markdown(self) -> str:
        lines = [
            f"### Alert — event {self.event_id} (rank #{self.rank_in_day} today)",
            f"- **User:** {self.src_user}",
            f"- **Source computer:** {self.src_computer}",
            f"- **Destination computer:** {self.dst_computer}",
            f"- **Time:** {self.time} (raw LANL epoch seconds)",
            f"- **Score:** {self.score:.4f}",
            "",
            "**Top contributing features:**",
        ]
        for feat in self.top_features:
            lines.append(
                f"- `{feat.feature_name}` = {feat.value:.4g} "
                f"(usual for this user: {feat.typical_value:.4g}), "
                f"contribution {feat.contribution:+.4g}"
            )
        if self.mitre_techniques:
            lines.append("")
            lines.append(f"**MITRE ATT&CK:** {', '.join(self.mitre_techniques)}")
        return "\n".join(lines)


def build_alert_card(
    *,
    event_id: int,
    src_user: str,
    src_computer: str,
    dst_computer: str,
    time: int,
    score: float,
    rank_in_day: int,
    top_features: list[FeatureContribution],
    mitre_techniques: list[str] | None = None,
) -> AlertCard:
    return AlertCard(
        event_id=event_id,
        src_user=src_user,
        src_computer=src_computer,
        dst_computer=dst_computer,
        time=time,
        score=score,
        rank_in_day=rank_in_day,
        top_features=top_features,
        mitre_techniques=mitre_techniques or [],
    )
