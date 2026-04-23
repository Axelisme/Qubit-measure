---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

```python
import zcu_tools.experiment.v2 as ze
```

```python
%matplotlib widget
exp = ze.FakeExp()
_ = exp.run()
```

```python
# %matplotlib inline
fig = exp.analyze()
```

```python

```
