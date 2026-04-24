from __future__ import annotations

from pathlib import Path

import pandas as pd

from thinkrouter.adapters import default_model_configs
from thinkrouter.experiments.policy_utils import add_sample_id, summarize_selection
from thinkrouter.routers import (
    LogRegJointRouter,
    MLPFactorizedRouter,
    ThresholdRouter,
    UncertaintyAwareRouter,
    load_factorized_artifact,
    load_logreg_joint_artifact,
)


def parse_router_spec(spec: str) -> tuple[str, str | None]:
    if "=" in spec:
        router_name, model_path = spec.split("=", 1)
        return router_name.strip(), model_path.strip()
    return spec.strip(), None


def _single_or_mixed(selected: pd.DataFrame, column: str) -> object:
    if column not in selected.columns:
        return pd.NA
    values = selected[column].dropna()
    if values.empty:
        return pd.NA
    unique = values.astype(str).unique().tolist()
    if len(unique) == 1:
        value = unique[0]
        if column == "selected_budget":
            try:
                return int(float(value))
            except ValueError:
                return value
        return value
    return "mixed"


def summarize_phase2_selection(policy: str, router_name: str, selected: pd.DataFrame) -> dict[str, object]:
    summary = summarize_selection(policy, selected)
    if selected.empty:
        return {
            **summary,
            "policy_family": f"phase2_{router_name}",
            "router_name": router_name,
            "selected_model": pd.NA,
            "selected_model_provider": pd.NA,
            "selected_model_tier": pd.NA,
            "selected_model_alias": pd.NA,
            "selected_budget": pd.NA,
            "avg_route_confidence": pd.NA,
            "fallback_rate": 0.0,
        }
    confidence = pd.to_numeric(selected.get("route_confidence"), errors="coerce")
    fallback = selected.get("fallback_triggered")
    fallback_rate = 0.0 if fallback is None else float(pd.Series(fallback).fillna(False).astype(bool).mean())
    return {
        **summary,
        "policy_family": f"phase2_{router_name}",
        "router_name": router_name,
        "selected_model": _single_or_mixed(selected, "selected_model"),
        "selected_model_provider": _single_or_mixed(selected, "selected_model_provider"),
        "selected_model_tier": _single_or_mixed(selected, "selected_model_tier"),
        "selected_model_alias": _single_or_mixed(selected, "selected_model_alias"),
        "selected_budget": _single_or_mixed(selected, "selected_budget"),
        "avg_route_confidence": float(confidence.dropna().mean()) if confidence.notna().any() else pd.NA,
        "fallback_rate": fallback_rate,
    }


def replay_router(
    csv_path: str,
    router_name: str,
    model_path: str | None = None,
    confidence_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_sample_id(pd.read_csv(csv_path))
    models = list(default_model_configs().values())
    if router_name == "threshold":
        router = ThresholdRouter(models)
    elif router_name == "logreg_joint":
        if not model_path:
            raise ValueError("--model is required for logreg_joint")
        router = LogRegJointRouter(models, artifact=load_logreg_joint_artifact(model_path))
    elif router_name == "mlp_factorized":
        if not model_path:
            raise ValueError("--model is required for mlp_factorized")
        router = MLPFactorizedRouter(models, artifact=load_factorized_artifact(model_path))
    elif router_name == "uncertainty_aware":
        if not model_path:
            raise ValueError("--model is required for uncertainty_aware")
        router = UncertaintyAwareRouter(
            models,
            artifact=load_factorized_artifact(model_path),
            confidence_threshold=0.55 if confidence_threshold is None else float(confidence_threshold),
        )
    else:
        raise ValueError(f"Unsupported router: {router_name}")

    selected_rows: list[pd.Series] = []
    for _, group in df.groupby("sample_id", sort=False):
        first = group.iloc[0]
        decision = router.route(str(first["query"]), str(first["task_type"]))
        exact = group[
            (group["selected_model"].astype(str) == decision.model_id)
            & (group["selected_budget"].astype(int) == int(decision.budget))
        ]
        candidates = exact if not exact.empty else group.sort_values(["cost_usd", "latency_s", "selected_budget"], ascending=[True, True, True])
        selected = candidates.iloc[0].copy()
        selected["policy"] = f"phase2_{router_name}"
        selected["route_confidence"] = decision.route_confidence
        selected["fallback_triggered"] = decision.fallback_triggered
        selected["fallback_reason"] = decision.fallback_reason
        selected["router_name"] = decision.router_name
        selected_rows.append(selected)
    selected_df = pd.DataFrame(selected_rows)
    summary = pd.DataFrame([summarize_phase2_selection(f"phase2_{router_name}", router_name, selected_df)])
    return summary, selected_df


def replay_router_specs(csv_path: str, router_specs: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries: list[pd.DataFrame] = []
    selected_outputs: dict[str, pd.DataFrame] = {}
    for spec in router_specs:
        router_name, model_path = parse_router_spec(spec)
        summary, selected = replay_router(csv_path, router_name, model_path=model_path)
        summaries.append(summary)
        selected_outputs[router_name] = selected
    if not summaries:
        return pd.DataFrame(), selected_outputs
    return pd.concat(summaries, ignore_index=True), selected_outputs


def write_selected_outputs(out_dir: str | Path, selected_outputs: dict[str, pd.DataFrame]) -> list[Path]:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for router_name, selected in selected_outputs.items():
        out_path = base / f"{router_name}_selected.csv"
        selected.to_csv(out_path, index=False)
        written.append(out_path)
    return written
