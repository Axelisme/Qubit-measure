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
    display_name: zcu-tools
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
    version: 3.13.11
---

# Sample Merge

Run this notebook before `T1_curve.md` / `T2_curve.md` when raw sample files come
from mixed units or source flux frames. The curve notebooks should only read the
canonical `samples.csv` produced here.

# Import

```python
%load_ext autoreload

import matplotlib.pyplot as plt
from IPython.display import display

%autoreload 2
import zcu_tools.notebook.analysis.fit_tools as zfit
```

# Project

```python
target_result_dir = "../../result/Q12_2D[7]/Q4"
source_2dq12_result_dir = "../../result/2DQ12/Q4"

samples_output = f"{target_result_dir}/samples.csv"
report_output = f"{target_result_dir}/samples_merge_report.csv"
figure_output = f"{target_result_dir}/samples_merge_f01_diagnostics.png"
```

# Sources

```python
sources = (
    zfit.SampleSource(
        path=f"{target_result_dir}/samples1.csv",
        unit="mA",
        source_result_dir=target_result_dir,
        label="samples1",
        fit_batch_flux_offset=False,
    ),
    zfit.SampleSource(
        path=f"{target_result_dir}/samples2.csv",
        unit="A",
        source_result_dir=source_2dq12_result_dir,
        label="samples2",
        fit_batch_flux_offset=True,
        batch_flux_offset_reference="target",
        batch_flux_offset_objective="soft_l1",
        max_abs_batch_flux_offset=0.03,
    ),
    zfit.SampleSource(
        path=f"{target_result_dir}/samples3.csv",
        unit="A",
        source_result_dir=source_2dq12_result_dir,
        label="samples3",
        fit_batch_flux_offset=True,
        batch_flux_offset_reference="target",
        batch_flux_offset_objective="soft_l1",
        max_abs_batch_flux_offset=0.03,
    ),
)
```

# Merge

```python
merge = zfit.merge_sample_sources(
    target_result_dir=target_result_dir,
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

```python
samples_path = zfit.write_merged_samples(merge, samples_output)
report_path = zfit.write_sample_merge_report(merge, report_output)

print(f"samples.csv = {samples_path}")
print(f"merge report = {report_path}")
print(f"diagnostic figure = {figure_output}")
```
