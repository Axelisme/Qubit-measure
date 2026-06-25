---
status: accepted
---

# Arbitrary waveform delete and rename do not scan ModuleLibrary references

Deleting an arbitrary waveform asset removes the qubit-scoped asset without scanning all `ModuleLibrary` waveform and module entries for references. Renaming an arbitrary waveform asset moves the asset to a new data key without migrating existing `ModuleLibrary` references. Reference scanning is broad, expensive, and risky because arbitrary waveform data is referenced through nested experiment configuration shapes. If a remaining `style="arb"` waveform references missing data, the using path fails when it loads the asset. Create/import operations fail on key collision unless the caller explicitly requests overwrite, rename never overwrites an existing key, and formula update overwrites an existing asset's playback arrays and embedded recipe.
