# ThinkRouter Project Brief

ThinkRouter is a routing system for reasoning workloads that selects `model + budget` jointly and records cost, latency, correctness, and route metadata for offline replay.

## Current State

- official protocol is frozen
- Phase 2 router stack is the default online route
- semantic features now use `sentence-transformers`
- deterministic GSM8K, MATH, and HumanEval evaluation paths exist
- one-command official rerun pipeline exists

## Current Main Committed Reference Result

The main committed multi-model reference slice is GSM8K `dev20` on the Qwen pool:

- strongest fixed point: `qwen-max @ 0`
- learned routers: operational, but not yet better on utility in the committed slice

## Not Yet Committed

The final official protocol rerun across `GSM8K`, `MATH-500`, and `HumanEval` has not yet been committed, so the repository does not yet include the definitive official result table and final official report artifacts.

## Official Entry Point

```powershell
.\scripts\run_official_pipeline.ps1
```
