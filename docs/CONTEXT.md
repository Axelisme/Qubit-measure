# Qubit Measurement Context

This context defines the domain language for measurement workflows, persisted
measurement data, and analysis handoffs in ZCU-Tools.

## Language

**Experiment Run**:
A single execution of an experiment against a specific configuration and hardware
state. An experiment run produces one experiment result.
_Avoid_: job, task run, measurement job

**Experiment Result**:
The complete measurement outcome of one experiment run, including all measured
data needed to analyze that run as a whole.
_Avoid_: output blob, save payload

**Complete Experiment Result**:
An experiment result whose required dataset roles are all present. Normal
analysis and loading operate on complete experiment results.
_Avoid_: partial result, best-effort result

**Experiment Result Identity**:
The shared identity that ties persisted measurement data back to one experiment
result. It belongs to the experiment result, not to any individual dataset.
_Avoid_: primary file identity, sidecar identity

**Labber Dataset**:
A single persisted measurement dataset with axes and measured values in the
Labber-style data model used by the project.
_Avoid_: HDF5 blob, data file

**Experiment Data File**:
The canonical persisted file for one experiment result. If the experiment result
has multiple dataset roles, they belong in the same experiment data file.
_Avoid_: sidecar file, companion file, workaround file

**Legacy Measurement Artifact**:
A persisted measurement artifact written before the current experiment data file
language. Legacy artifacts may be migrated, but they are not a normal loading
format.
_Avoid_: supported old format, compatibility path

**Dataset Role**:
The semantic role of a member dataset inside a grouped experiment dataset, as
defined by that experiment result. Each dataset role may have its own measurement
shape; shared experiment result identity, not shared shape, is what makes the
members a group.
_Avoid_: file suffix, member name

**Grouped Experiment Dataset**:
A set of Labber datasets that together represent one experiment result. The
group, not any individual member, is the canonical persisted measurement result;
all member datasets are peers under the same experiment result identity and each
member has a dataset role.
_Avoid_: grouped persistence, multi-file workaround, sidecar artifact

**Flux-Dependence Analysis**:
The analysis workflow that uses flux-dependent spectra, selected spectral
features, and flux alignment to produce a fluxonium fit handoff for later
measurement and dispersive analysis.
_Avoid_: fluxdep widget, spectrum editor, point-picking UI

**Fluxonium Prediction**:
The simulation-backed calculation that maps fitted fluxonium parameters and
value-to-flux alignment into predicted transition frequencies, matrix elements,
or dispersive resonator frequencies. GUI services and notebook helpers may adapt
it, but they are not the prediction model itself.
_Avoid_: GUI predictor service, dispersive tuning widget, plot overlay

**Arbitrary Waveform Asset**:
A qubit-scoped stored waveform sample asset file that can be referenced by an
arbitrary waveform configuration. It always has playback data, and it may also
embed an editable formula recipe.
_Avoid_: waveform cfg, formula file, arbitrary pulse, asset folder

**Arbitrary Waveform Data Key**:
The path-safe identity of an arbitrary waveform asset, used as the asset filename
stem and as the reference string in arbitrary waveform configurations.
_Avoid_: filename, module name, waveform name

**Formula Recipe**:
An optional editable source definition for an arbitrary waveform asset. User and
agent-created formula waveforms store a formula recipe, while imported waveform
assets may exist as playback data only. A formula recipe is stored inside the
asset file and may contain one or more formula segments.
_Avoid_: source of truth, cache metadata, formula cfg

**Formula Segment**:
An ordered part of a formula recipe with its own duration and formula
expression. Formula segments concatenate to form one arbitrary waveform asset;
their formulas are written against segment-local time. Segment rendering uses
half-open time intervals, with the final segment including the waveform endpoint.
Total waveform duration is the sum of segment durations.
_Avoid_: piece, section, pulse part

## Example Dialogue

Developer: "This experiment result has multiple measured views. Is that several
experiment runs?"

Domain expert: "No. It is one experiment run. Persist it as a grouped experiment
dataset so the datasets stay tied to the same result."

Developer: "Can one member be a sidecar artifact in another storage format?"

Domain expert: "No. If it is part of the canonical experiment result, model it as
a Labber dataset inside the grouped experiment dataset."
