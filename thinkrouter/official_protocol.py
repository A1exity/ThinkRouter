from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficialBenchmark:
    benchmark: str
    task_type: str
    source: str
    train_count: int = 60
    dev_count: int = 20
    test_count: int = 20


@dataclass(frozen=True)
class OfficialUtility:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02


@dataclass(frozen=True)
class OfficialProtocol:
    version: str
    model_pool: tuple[str, ...]
    budgets: tuple[int, ...]
    benchmarks: tuple[OfficialBenchmark, ...]
    baselines: tuple[str, ...]
    routers: tuple[str, ...]
    utility: OfficialUtility
    default_router: str
    report_fields: tuple[str, ...]
    semantic_backend: str
    semantic_model_name: str
    appendix_prefixes: tuple[str, ...]

    def benchmark_by_name(self, benchmark: str) -> OfficialBenchmark:
        for item in self.benchmarks:
            if item.benchmark == benchmark:
                return item
        raise KeyError(f"Unsupported official benchmark: {benchmark}")

    def data_path(self, benchmark: str) -> Path:
        return Path("data") / "splits" / f"official_{benchmark}.jsonl"

    def trace_db_path(self, benchmark: str, split: str) -> Path:
        return Path("results") / "traces" / f"official_{benchmark}_{split}.sqlite"

    def grid_csv_path(self, benchmark: str, split: str) -> Path:
        return Path("results") / "tables" / f"official_{benchmark}_{split}_grid.csv"

    def router_prefix(self, benchmark: str) -> str:
        return str(Path("results") / "official" / benchmark / f"{benchmark}_official")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["benchmarks"] = [asdict(item) for item in self.benchmarks]
        payload["utility"] = asdict(self.utility)
        return payload


OFFICIAL_PROTOCOL = OfficialProtocol(
    version="official_v1",
    model_pool=("qwen-flash", "qwen-plus", "qwen-max"),
    budgets=(0, 256, 1024),
    benchmarks=(
        OfficialBenchmark(benchmark="gsm8k", task_type="gsm8k", source="gsm8k"),
        OfficialBenchmark(benchmark="math500", task_type="math", source="math500"),
        OfficialBenchmark(benchmark="humaneval", task_type="humaneval", source="humaneval"),
    ),
    baselines=(
        "fixed_model_budget",
        "model_only",
        "budget_only",
        "joint_aggregate_utility",
        "joint_safe_fallback",
    ),
    routers=("threshold", "logreg_joint", "mlp_factorized", "uncertainty_aware"),
    utility=OfficialUtility(alpha=1.0, beta=5.0, gamma=0.02),
    default_router="uncertainty_aware",
    report_fields=(
        "benchmark",
        "policy",
        "policy_family",
        "accuracy",
        "avg_cost",
        "p95_latency",
        "utility",
        "cost_reduction_vs_strongest_fixed",
        "accuracy_drop_vs_strongest_fixed",
    ),
    semantic_backend="sentence-transformers",
    semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
    appendix_prefixes=("day1", "dev5", "dev10", "dev20", "smoke", "seed"),
)


def official_router_model_env(router_name: str) -> str | None:
    mapping = {
        "logreg_joint": "THINKROUTER_LOGREG_JOINT_MODEL_PATH",
        "mlp_factorized": "THINKROUTER_FACTORIZED_ROUTER_MODEL_PATH",
        "uncertainty_aware": "THINKROUTER_FACTORIZED_ROUTER_MODEL_PATH",
    }
    return mapping.get(router_name)
