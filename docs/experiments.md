# Experiments

This repository now distinguishes between:

1. the frozen official protocol
2. historical appendix slices

Only the official protocol should be used for future formal results.

## Official Entry Point

```powershell
.\scripts\run_official_pipeline.ps1
```

Equivalent staged commands:

```bash
python -m thinkrouter.experiments.run_official_pipeline --stage prepare-data
python -m thinkrouter.experiments.run_official_pipeline --stage grids
python -m thinkrouter.experiments.run_official_pipeline --stage routers
python -m thinkrouter.experiments.run_official_pipeline --stage report
```

## Frozen Official Protocol

- model pool: `qwen-flash,qwen-plus,qwen-max`
- budgets: `0,256,1024`
- benchmarks: `gsm8k`, `math500`, `humaneval`
- split sizes: `60 train / 20 dev / 20 test`
- default runtime router: `uncertainty_aware`

## Output Layout

Official rerun outputs are written under:

- `results/official/gsm8k/`
- `results/official/math500/`
- `results/official/humaneval/`

Top-level official report targets are:

- `results/tables/final_official_results.csv`
- `results/figures/final_official_pareto.png`
- `results/tables/final_official_failures.csv`
- `results/reports/final_official_report.md`

## Historical Appendix

Older `dev5/dev10/dev20`, smoke, and seed slices remain in the repository for debugging only. They should not be cited as the official benchmark protocol.
