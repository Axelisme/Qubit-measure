---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: zcu-tools (3.13.11.final.0)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.13.undefined
---

# Sample Merge

**Readers:** operator, lab-author. Merge record JSONL into the canonical `samples.csv` used by the T1 and T2 notebooks.

Run this notebook before `T1_curve.md` / `T2_curve.md` when raw sample files come
from different source flux frames or device units. The curve notebooks should
only read the canonical `samples.csv` produced here.

Sources are record JSONL histories, with the last whole row kept for each
`run_id`. Each row uses explicit `flux`, or the selected
`devices[device_name].value` with `flux_int` / `flux_period` or a declared
`fallback_frame`. Only fallback resolution needs a matching device `value_unit`.
Unused NaN, extra fields and metadata do not block merging. Missing required
fields report the source file and field names. The output remains a flat
`samples.csv` with target coordinates followed by the source measurement fields.

For legacy `calibrated mA` / `Flux` CSVs, follow the formal
[Sample CSV → JSONL migration procedure](../../migrations/README.md#sample-csv--jsonl)
with explicit source units and column mapping. The merge does not read CSV,
recognize legacy aliases, infer units or supply missing device facts. Before
merging migrated rows, check that each row has explicit `flux` or the selected
device value and a usable frame.

# Import

```python
%load_ext autoreload

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display

%autoreload 2
import zcu_tools.notebook.analysis.fit_tools as zfit
```

# Project

```python
target_result_dir = "../../result/Q12_2D[7]/Q4"

samples_output = f"{target_result_dir}/samples.csv"
report_output = f"{target_result_dir}/samples_merge_report.csv"
figure_output = f"{target_result_dir}/samples_merge_f01_diagnostics.png"
```

# Target frame

The target frame is the single authoritative merge identity. `dev_unit` is the
target device's A/V unit (current source -> `"A"`, voltage source -> `"V"`); the
persisted params carry no unit metadata, so it is declared here explicitly.

```python
target_frame = zfit.FluxFrame.from_result_dir(
    target_result_dir,
    dev_unit="A",
    label="Q4 target",
)
```

# Sources

Each source is record JSONL. Set `device_name` to the captured device used for
rows without explicit `flux`; missing required device values fail without a
flat-column fallback. `fallback_frame` resolves rows that carry
neither explicit `flux` nor a row frame. `integer_flux_offset` aligns a source
to an equivalent integer flux branch (e.g. `-0.5` -> `+1` to match `0.5`); the
merge never infers a branch on its own.

```python
sources = (
    zfit.SampleSource(
        device_name="bias_device",  # Select a captured device explicitly.
        path=f"{target_result_dir}/records1.jsonl",
        label="samples1",
    ),
    zfit.SampleSource(
        device_name="bias_device",  # Select a captured device explicitly.
        path=f"{target_result_dir}/records2.jsonl",
        label="samples2",
        integer_flux_offset=1,
        fit_batch_flux_offset=True,
        batch_flux_offset_objective="soft_l1",
        max_abs_batch_flux_offset=0.03,
    ),
    zfit.SampleSource(
        device_name="bias_device",  # Select a captured device explicitly.
        path=f"{target_result_dir}/records3.jsonl",
        label="samples3",
        fallback_frame=zfit.FluxFrame.from_result_dir(
            "../../result/2DQ12/Q4",
            dev_unit="A",
            label="2DQ12/Q4 migrated",
        ),
        fit_batch_flux_offset=True,
        batch_flux_offset_objective="soft_l1",
        max_abs_batch_flux_offset=0.03,
    ),
)
```

# Merge

```python
merge = zfit.merge_sample_sources(
    target_frame=target_frame,
    sources=sources,
)
```

```python
display(merge.summary_table)
display(merge.merged.head(10))
```

# Diagnostics

```python
fig, _ = zfit.plot_sample_merge_f01_diagnostics(merge)
fig.savefig(figure_output, dpi=160, bbox_inches="tight")
plt.show()
```

# Write Output

The output is an analysis artifact, not raw adapter records. Original devices
remain source evidence and are not rewritten to match target-frame coordinates.
Do not feed this output back into raw-record APIs.

The output path is caller-owned and must never overwrite a source CSV; the
write helpers refuse to do so.

```python
source_paths = {str(Path(source.path).resolve()) for source in sources}
assert str(Path(samples_output).resolve()) not in source_paths, (
    "output path must not overwrite a source CSV"
)

samples_path = zfit.write_merged_samples(merge, samples_output)
report_path = zfit.write_sample_merge_report(merge, report_output)

print(f"samples.csv = {samples_path}")
print(f"merge report = {report_path}")
print(f"diagnostic figure = {figure_output}")
```
