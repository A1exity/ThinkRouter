# ThinkRouter Project Brief

ThinkRouter is a routing system for reasoning workloads that selects `model + budget` jointly and records cost, latency, correctness, and route metadata for offline replay.

## Final State

- official protocol is frozen and completed
- Phase 2 router stack is the default online route
- semantic features use `sentence-transformers`
- deterministic `GSM8K`, `MATH-500`, and `HumanEval` evaluation paths are complete
- one-command official rerun pipeline exists
- final official results and final official report are present

## Final Main Result

The key acceptance result is:

- on official `GSM8K`, the learned router `phase2_logreg_joint` beats both the strongest fixed baseline and the aggregate baseline

The repository also contains official negative results:

- on `MATH-500`, the learned router does not win
- on `HumanEval`, the learned router does not win

## Official Entry Point

```powershell
.\scripts\run_official_pipeline.ps1
```
