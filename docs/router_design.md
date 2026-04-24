# Router Design

ThinkRouter compares four router families:

- `threshold`
- `logreg_joint`
- `mlp_factorized`
- `uncertainty_aware`

## Feature Stack

The active feature stack is:

- surface features
- sentence-transformer semantic features
- cheap-probe features

The earlier semantic-hash placeholder is no longer the main semantic path.

## Decision Structures

Two learned decision structures are implemented:

1. joint `(model, budget)` classification
2. factorized `model` head plus `budget` head

`uncertainty_aware` wraps the factorized router and falls back when confidence is below threshold.

## Runtime Position

The Phase 2 router stack is now the default online path. The old `JointPolicyEngine` remains only as a legacy baseline.

## Current Conclusion

The router stack is implemented and reportable. The current committed historical GSM8K reference slice still favors the strongest fixed baseline on utility, so the remaining gap is experimental outcome, not missing router infrastructure.
