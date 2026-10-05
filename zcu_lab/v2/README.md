# v2 experiments

**Last updated:** 2026-10-05，通用契約與共用 helper 測試

每個實驗有自己的資料夾。`core.py` 擁有 cfg、Result、量測與分析政策。`gui.py` 是可選的 measure-gui adapter。`autofluxdep.py` 是可選的 Autofluxdep Builder／Node 附件。Leaf 的 `__init__.py` 只提供 core exports，不載入 GUI。

`v2` 延續 program/v2 的版本脈絡，不表示 GUI framework 的第二版。通用 runtime、量化、SNR 與 tracker 工具留在 `zcu_tools.experiment.v2`。Core 不依賴 GUI，import-linter C15 檢查 core 到 GUI framework 的直接與間接依賴。

## Layout

```text
zcu_lab/
├── definitions.py        # explicit measure catalog
├── roles.py              # startup-only role catalog
├── autofluxdep_catalog.py
└── v2/
    ├── <family>/<experiment>/
    │   ├── core.py
    │   ├── gui.py         # optional measure adapter
    │   └── autofluxdep.py # optional Builder / Node attachment
    ├── _support/         # shared domain helpers
    ├── autofluxdep/       # flux executor and its leaves
    └── overnight/         # repeated-measurement executor and its leaves
```

`definitions.register_all` 明列 measure adapters，`roles.register_all_roles` 組合 roles。組合根注入 caller-owned catalogs，import 不註冊或操作硬體。Reload 重建 leaf、catalog 與 core-typed Notebook helper，保留 measure／autofluxdep／singleshot shared support、workflow support、roles 與框架 identity。修改固定依賴需要重啟。

## Reading routes

- [Experiment authoring](experiment-authoring.md) 說明核心 records、資料形狀與 run 慣例。
- [Measure authoring](measure-authoring.md) 說明 cfg defaults、operator guides、writeback 與 reload。
- [GUI contracts](gui-contracts.md) 說明 concrete adapter hooks 與 capability。
- [Autofluxdep](autofluxdep/README.md) 分別連到 executor 與 GUI Node 的 authoring。兩套 workflow 不共用接口或 reload catalog。
- [Measure support](_support/measure/README.md) 與 [Autofluxdep support](_support/autofluxdep/README.md) 各自提供共享 domain mechanics。
- [Framework runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 擁有 Schedule、buffer、acquire、stop 與 executor lifecycle。

測試位於 `tests/zcu_lab/`。套件層的通用契約由 registry 選出每個 adapter；`tests/zcu_lab/v2/` 保留共用 helper、role defaults 與使用者要求的 AllXY、ZigZag、ZigZagScan 特例，不逐實驗測 cfg、fit 或 writeback 政策。Core-only caller 明確 import `core`；GUI callers 明確選擇其附件。
