# ThinkRouter Phase 2 Closeout

Phase 2 is complete in the current repository state.

The strongest Phase 2 evidence comes from the real Qwen 3.5 pool GSM8K `dev20` run:

- candidate grid: `flash / plus / max` x `0 / 256 / 1024`
- traces: `180`
- integrated Phase 2 summary: `results/qwen35_pool_gsm8k_dev20_baseline_phase2_summary.csv`
- ranked report: `results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md`

On that slice, all four Phase 2 routers replayed successfully through the same evaluation flow:

| policy | accuracy | avg cost | avg latency | avg route confidence |
| --- | ---: | ---: | ---: | ---: |
| `phase2_threshold` | 0.950 | 0.000246 | 6.744s | 0.6381 |
| `phase2_logreg_joint` | 0.950 | 0.000246 | 6.744s | 0.7785 |
| `phase2_mlp_factorized` | 0.950 | 0.000246 | 6.744s | 0.9873 |
| `phase2_uncertainty_aware` | 0.950 | 0.000246 | 6.744s | 0.9873 |

The utility winner on GSM8K `dev20` still remained `qwen-max @ budget 0`. That is an experimental conclusion, not an implementation gap. The router stack is complete enough to measure that result honestly.

The code-task side is also closed out at the Phase 2 level. The existing real Humaneval `dev2` Qwen pool grid was replayed through the same Phase 2 stack offline, producing:

- `results/qwen35_pool_humaneval_dev2_budget256_phase2_baseline_phase2_summary.csv`
- `results/reports/qwen35_pool_humaneval_dev2_budget256_phase2_report.md`

That slice is tiny and all routes remain incorrect, but it proves that the code-task benchmark path goes through the same Phase 2 machinery: feature extraction, router replay, uncertainty metadata, integrated summaries, Pareto outputs, and ranked reports.

Phase 2 deliverables now present in the repo:

- extensible feature pipeline
- `threshold`, `logreg_joint`, `mlp_factorized`, `uncertainty_aware`
- utility-aware training and replay
- integrated baseline plus Phase 2 summary flow
- utility-ranked Phase 2 report generation
- Streamlit display of router name, confidence, and fallback state
- real Qwen pool GSM8K Phase 2 artifacts
- real Qwen pool code-task Phase 2 replay artifacts

The next unfinished work belongs to Phase 3 and Phase 4, not Phase 2.
