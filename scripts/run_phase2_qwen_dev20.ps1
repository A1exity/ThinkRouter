$ErrorActionPreference = "Stop"

python -m thinkrouter.experiments.run_grid `
  --input data/splits/gsm8k.jsonl `
  --task gsm8k `
  --split dev `
  --limit 20 `
  --budgets 0,256,1024 `
  --models qwen-flash,qwen-plus,qwen-max `
  --db results/traces/qwen35_pool_gsm8k_dev20_phase2.sqlite `
  --out results/tables/qwen35_pool_gsm8k_dev20_grid.csv `
  --resume

python -m thinkrouter.experiments.run_phase2_eval `
  results/tables/qwen35_pool_gsm8k_dev20_grid.csv `
  --out-prefix results/qwen35_pool_gsm8k_dev20

python -m thinkrouter.experiments.make_phase2_report `
  results/qwen35_pool_gsm8k_dev20_baseline_phase2_summary.csv `
  --summary-out results/tables/qwen35_pool_gsm8k_dev20_phase2_ranked.csv `
  --markdown-out results/reports/qwen35_pool_gsm8k_dev20_phase2_report.md
