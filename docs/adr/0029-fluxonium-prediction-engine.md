---
status: accepted
---

# Fluxonium Prediction engine owns simulation policy outside GUI adapters

Fluxonium Prediction lives in `zcu_tools.simulate.fluxonium` as a production
simulation-facing engine, not inside GUI services. GUI dispersive tuning,
measure-gui predictor/session code, and notebook helpers adapt this engine but
do not own prediction policy.

The engine owns the value-to-flux affine mapping, typed `PredictionResolution`,
dispersive fast-path plus scqubits fallback, fallback provenance, and axis-bound
cache identity. Resolution is injectable for notebooks, tests, and batch
prediction, while GUI adapters keep their fixed app defaults and do not expose
resolution knobs. Fallback from fast dressed-state labeling to scqubits is a
normal engine policy: GUI normal paths do not catch `DressedLabelingError`, but
prediction results expose lightweight provenance such as the backend used.

Axis-bound prediction cache belongs to the engine. A caller creates a prediction
session for one `(params, flux axis, resolution)` combination and reuses it for
slider-style `(g, bare_rf, return_dim)` calls; controllers rebuild the session
when fit inputs or preprocessed axes change instead of constructing their own
cache keys.

`FluxoniumPredictor` remains a stable facade for existing measure-gui and
notebook callers. It keeps its public API, MHz frequency contract, `from_file()`,
`calculate_bias()`, and `clone()`, and may delegate internal computation to the
prediction engine incrementally.
