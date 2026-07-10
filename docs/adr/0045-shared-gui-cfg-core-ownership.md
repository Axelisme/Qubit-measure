---
status: accepted
---

# 0045 — Shared GUI cfg core ownership

**狀態：** accepted（2026-07-10）。
**關聯：** [[0009]]、[[0010]]、[[0011]]、[[0012]]、[[0043]]。

## 背景

measure-gui 與 autofluxdep-gui 共用同一套 Spec/Value tree、完整 value tree
繼承規則與 persistence codec，但實作原本位於 measure app package。autofluxdep
因此必須從 `gui.app.main` 取得 generic model/codec，使純資料 ownership 與 app runtime
ownership 混在一起。

linked Module/Waveform reference 的 finished-cfg validation/lowering 仍需即時查詢
`ModuleLibrary`，並透過 measure-owned module/waveform conversion 判斷 concrete shape。
直接把這段一起搬進 shared 會造成 `gui.cfg -> gui.app.main` 反向依賴，或需要先設計新的
resolver port；兩者都超出本 slice。

## 決策

`zcu_tools.gui.cfg` 擁有以下 app-independent core：

- `model.py`：Spec/Value 型別、`CfgSchema` 的 `spec`/`value` data carrier；
- `inheritance.py`：default value、locked literal alignment、inheritance；
- `codec.py`：session/node cfg raw ↔ live value tree codec 與 `SessionCodecError`；
- package `__init__`：只 re-export上述 public names。

`gui.cfg` 不 import `gui.app.*`、`experiment.*`、Qt 或 `meta_tool`。`CfgSchema` 不保存
environment callback、resolver registration 或 app runtime dependency。

measure adapter package 暫時 re-export shared cfg public names，而且 identity 必須相同；它
不保留舊 model/inheritance/codec wrapper implementation。finished-cfg entry point 留在
`gui.app.main.adapter.lowering`：

- `validate_schema(schema, ml)`；
- `validate_dynamic_schema(schema, md, ml)`；
- `schema_to_raw_dict(schema, md, ml)`。

`schema_to_raw_dict` 依序執行 static validation、可選 dynamic validation、再 lower。
linked-reference resolution 與 module/waveform conversion 仍由 measure app 擁有。

autofluxdep 的 model、inheritance 與 codec import 直接指向 `gui.cfg`。它的
`NodeCfgSchema.logical_paths`、generation persistence reshape、`OverridePlan`、node builder
與 pulse/readout spec factory仍由 app/domain layer 擁有。autofluxdep 在本 slice 仍顯式呼叫
measure lowering，且 module/pulse/readout conversion 與 Qt form seam 仍是精確列出的 app
coupling；本決策不宣稱已消除全部 autofluxdep → measure app dependency。

## 後果

- shared cfg package 可在 fresh process 中不載入 app、experiment、Qt 或 `meta_tool`。
- measure 與 autofluxdep 使用同一組 class/function identity與相同 codec wire shape。
- finished-cfg caller 改用 app-local free function，observable validation/lowering順序不變。
- 下一 phase 若要下沉 generic lowering，需先定義窄的 expression evaluator 與
  linked-reference resolver seam；本 ADR 不預先宣告該 phase 已完成。

## 拒絕的替代方案

- **shared method runtime import measure app**：破壞 shared → app 單向依賴。
- **process-global resolver registration**：引入 import-order side effect與隱性 global state。
- **把 broad environment port 塞進 `CfgSchema`**：純 data carrier 承擔 app runtime責任。
- **只信 linked ref embedded snapshot**：會改變 missing library key 與 unsupported shape 的
  既有 validity/error contract。
