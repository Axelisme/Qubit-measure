"""Standalone fluxonium flux-dependence analysis GUI.

The app owns its State, services, interaction policy and native remote adapter.
Transport, events and plotting mechanisms use the shared GUI packages. Entry
selection and native source/dialog mechanics belong to analysis_project.

The pipeline loads spectra, aligns half/integer lines, samples and filters
points, then searches a fluxonium database and presents diagnostics. Export
saves spectrums v2 checkpoints and independently reports predictor v2 and
component P1 outcomes while recording analysis in the entry ledger.
"""
