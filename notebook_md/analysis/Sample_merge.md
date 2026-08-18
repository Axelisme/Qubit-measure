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

Run this notebook before `T1_curve.md` / `T2_curve.md` when raw sample files come
from different source flux frames or device units. The curve notebooks should
only read the canonical `samples.csv` produced here.

Sources must be flat v2 CSVs (`dev_value` / `dev_unit`, optional `flux` /
`flux_int` / `flux_period`). Legacy `calibrated mA` / `Flux` tables must be
migrated explicitly with `script/migrate_sample_table_v2.py` first — the merge
never guesses units or integer flux branches.

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

Each source is a v2 CSV. `fallback_frame` resolves migrated rows that carry
neither explicit `flux` nor a row frame. `integer_flux_offset` aligns a source
to an equivalent integer flux branch (e.g. `-0.5` -> `+1` to match `0.5`); the
merge never infers a branch on its own.

```python
sources = (
    zfit.SampleSource(
        path=f"{target_result_dir}/samples1.csv",
        label="samples1",
    ),
    zfit.SampleSource(
        path=f"{target_result_dir}/samples2.csv",
        label="samples2",
        integer_flux_offset=1,
        fit_batch_flux_offset=True,
        batch_flux_offset_objective="soft_l1",
        max_abs_batch_flux_offset=0.03,
    ),
    zfit.SampleSource(
        path=f"{target_result_dir}/samples3.csv",
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
