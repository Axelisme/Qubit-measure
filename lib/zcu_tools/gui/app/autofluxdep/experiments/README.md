**Last updated:** 2026-07-11 — experiment authoring boundary

# autofluxdep experiments

這個 package 是 autofluxdep-gui 量測實驗的使用者修改入口。每個實驗以一個同名 Python 檔案作唯一 authoritative definition，並在檔尾 export `EXPERIMENT = <Builder>()`。Builder/Node/RunEnv 契約保持在 `../nodes/`，執行順序仍由使用者 workflow 決定。

## Ownership

```text
experiments/
├── qubit_freq.py
├── lenrabi.py
├── ro_optimize.py
├── t1.py
├── t2ramsey.py
├── t2echo.py
├── mist.py
├── catalog.py
└── _support/
```

- `<name>.py` 擁有單一實驗的 cfg/schema、短生命週期 Node、acquire/fit/Patch policy、Result/Plotter factory，以及只屬於該實驗的 domain policy。concrete experiment 互不 import。
- `catalog.py` 是唯一會同時 import 七個 `EXPERIMENT` 的 composition root。catalog 的顯式順序只控制 GUI 新增選單，不改變 persisted workflow order。
- `__init__.py` 只有 package docstring，不 import catalog 或 concrete experiment。需要 catalog 的 caller 明確從 `experiments.catalog` import；只使用 `_support` 或單一 experiment 時不會啟動完整 composition root。
- `_support/` 只擁有至少兩個實驗共用的 mechanics；它不 import concrete experiment 或 catalog。詳細邊界見 `_support/README.md`。
- `../nodes/` 只擁有 ADR-0018 execution contracts 與 pure-compute predictor；量測實驗不放回該 package。

## 修改既有實驗

直接修改對應的 `<name>.py`，並保留 Builder/Node 介面、declaration keys、persistence/wire shape 與 Result/Patch 語意。只被該實驗使用的 helper 留在同一檔案；確實被至少兩個實驗共用時才移入 `_support/`。

## 新增實驗 checklist

1. 新增 `experiments/<name>.py`，使檔名 stem 與 `Builder.name` 完全相同。
2. 在同一檔案定義 cfg/schema、Node、Builder 與 experiment-specific policy，檔尾 export `EXPERIMENT`。
3. 在 `catalog.py` 顯式 import 該 singleton，並在 `_DECLARATIONS` 中放到預期的選單位置；不要使用 filesystem discovery、decorator side effect 或 plugin scan。
4. declaration tuple 內不得重複，experiment name 必須非空且唯一。unknown placement 維持 `KeyError`。
5. 不修改 `Builder.build_node()` / `Node.produce()` seam，不讓 orchestrator 理解 experiment-specific key，也不把 predictor 加進 catalog。
6. 在 `tests/autofluxdep_gui/experiments/` 增加該實驗的 cfg/acquire/fit/Result/Patch 測試，並讓 production-wide contract tests 明確從 `experiments.catalog` 的 `builders()` 取得 Builder。
7. 執行 catalog/import architecture tests、該實驗 targeted tests，以及相關 cfg/persistence/workflow tests。

跨模組 runtime contract 見 ADR-0018、ADR-0036、ADR-0043；本 package 的高層執行語言見 `../CONTEXT.md`。
