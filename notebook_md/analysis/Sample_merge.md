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

# Sample merge

**Readers:** operator, lab-author. Flatten record JSONL histories into a CSV for curve analysis.

Each source keeps the last whole row per `run_id`. The merge concatenates those
rows in source order and joins nested object keys with dots. For example,
`devices.bias_device.value` holds the captured device value.
The merge preserves source values. It does not calculate flux, require physical
fields, apply corrections or select a target frame. Extra metadata and unused
NaN values do not block the merge.

For old CSV files, use the formal
[Sample CSV → JSONL migration procedure](../../migrations/README.md#sample-csv--jsonl).
Declare source units and column mappings there. This notebook only reads JSONL
and does not interpret old field aliases.

# Imports

```python
%load_ext autoreload

from pathlib import Path

import pandas as pd
from IPython.display import display

%autoreload 2
from zcu_tools.notebook.analysis.fit_tools import (
    merge_sample_sources,
    write_merged_samples,
)
from zcu_tools.resources.record_table import RecordFluxFrame, flux_from_frame
```

# Sources and output

The output directory only selects where to write the CSV. It supplies no model
or calibration to the merge.

```python
result_dir = Path("../../result/Q12_2D[7]/Q4")
sources = (
    result_dir / "records1.jsonl",
    result_dir / "records2.jsonl",
)
samples_output = result_dir / "samples.csv"
```

# Merge and write

`write_merged_samples` creates the output directory and replaces an existing
output CSV. It refuses to overwrite any source JSONL, including empty sources.

```python
merge = merge_sample_sources(sources=sources)
display(merge.merged.head(10))

samples_path = write_merged_samples(merge, samples_output)
print(f"samples.csv = {samples_path}")
```

# Author-selected flux for analysis

Flux conversion is a separate step. Copy the examples below into T1 or T2
analysis and uncomment the call you need, or build your own flux array.
Pass that array to the analysis loader in the same row order. These examples
do not run during merging, so records without physical fields can still merge.
The helper does not change the table or CSV.

Set `device_name` to the captured device whose values define flux. The helper
uses finite explicit `flux` first. Otherwise it uses
`devices.<device_name>.value` with the row's `flux_int` and `flux_period`.
A row with only one frame value fails with its row label and missing field.
A valid explicit flux ignores unused frame fields.

```python
# samples = pd.read_csv(samples_path)
# flux = flux_from_frame(samples, device_name="bias_device")
# display(pd.DataFrame({"flux": flux}))
```

If rows lack both frame values and explicit flux, pass a named frame yourself.
The following example declares volts, a zero-flux value of 0.1 V and a period
of 0.8 V. Replace those numbers with your calibration. If calibration lives in
a file, read it before constructing `RecordFluxFrame`; the helper accepts frame
data, not a path. Fallback requires the selected device's `value_unit` to match.
No unit conversion occurs.

```python
# Use this call instead of the one above when a fallback is needed.
# fallback = RecordFluxFrame(dev_unit="V", flux_int=0.1, flux_period=0.8)
# flux = flux_from_frame(samples, device_name="bias_device", fallback_frame=fallback)
```

Call the separate f01 correction tool if your analysis needs it, then pass the
chosen flux array to T1 or T2. Neither merging nor loading chooses a correction
for you.
