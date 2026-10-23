"""The feature configuration states what the feature store computes, and nothing it ignores."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

from authbench.pipeline.build_features import CONF_DIR, check_supported


def _features(*overrides: str):  # type: ignore[no-untyped-def]
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides)).features


@pytest.mark.parametrize("split", ["temporal", "harmonised", "demo_temporal"])
def test_every_shipped_configuration_is_supported(split: str) -> None:
    check_supported(_features(f"split={split}"))


@pytest.mark.parametrize(
    "override",
    [
        "features.f2_history.windows_hours=[1,24,168]",
        "features.f2_history.entities=[src_user,src_computer,dst_computer]",
        "features.f1_event.enabled=false",
        "features.f4_temporal.night_window_calibration=fixed",
    ],
)
def test_a_setting_the_store_would_ignore_is_refused(override: str) -> None:
    with pytest.raises(ValueError, match="Unsupported feature settings"):
        check_supported(_features(override))


def test_unimplemented_feature_families_are_refused() -> None:
    with pytest.raises(ValueError, match="f5_graph.enabled"):
        check_supported(_features("features=graph"))
