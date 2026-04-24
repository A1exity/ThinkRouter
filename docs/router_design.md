# Router Design

ThinkRouter Phase 2 compares four router families:

- `threshold`
- `logreg_joint`
- `mlp_factorized`
- `uncertainty_aware`

## Feature pipeline

The feature stack is intentionally small and extensible:

- surface features
- semantic hash features
- cheap-probe difficulty / confidence / consistency

## Decision structures

Two routing structures are implemented:

1. joint classification over `(model, budget)`
2. factorized prediction over `model` and `budget`

The uncertainty-aware router wraps the factorized router and can fall back to a simpler policy when confidence is too low or the primary artifact is missing.

## Current conclusion

The router stack is complete and measurable on real Qwen pool slices, but the strongest baseline can still win on utility. That is a model-selection outcome, not an implementation gap.
