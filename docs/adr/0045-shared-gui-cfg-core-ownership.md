---
status: accepted
---

# 0045 — Shared GUI cfg core ownership

**狀態：** accepted（2026-07-10）。
**關聯：** [[0009]]、[[0010]]、[[0011]]、[[0012]]、[[0043]]、[[0046]]。

**演化：** finished-cfg lowering暫留 measure app的安排由 [[0046]] 取代；本 ADR 的
shared Spec/Value、inheritance、codec ownership與 import boundary繼續生效。

## 背景

measure-gui 與 autofluxdep-gui 共用同一套 Spec/Value tree、完整 value tree
繼承規則與 persistence codec，但實作原本位於 measure app package。autofluxdep
因此必須從 `gui.app.main` 取得 generic model/codec，使純資料 ownership 與 app runtime
ownership 混在一起。

linked Module/Waveform reference 的 finished-cfg validation/lowering需要即時查詢
app-owned library shape policy。[[0046]]以窄resolver port提供此能力，不讓shared core
反向依賴app runtime。

## 決策

`zcu_tools.gui.cfg` 擁有以下 app-independent core：

- `model.py`：Spec/Value 型別、`CfgSchema` 的 `spec`/`value` data carrier；
- `inheritance.py`：default value、locked literal alignment、inheritance；
- `codec.py`：session/node cfg raw ↔ live value tree codec 與 `SessionCodecError`；
- `lowering.py`：[[0046]] 定義的 generic finished-cfg algorithm、三個窄 callable
  ports與 validation/lowering operations；
- package `__init__`：re-export上述 app-independent core public names，包含 [[0046]] 的
  `ExpressionResolver`、`ReferenceResolver`、`RangeFactory`、`validate_finished_cfg`與
  `lower_finished_cfg`。

`gui.cfg` 不 import `gui.app.*`、`experiment.*`、Qt 或 `meta_tool`。`CfgSchema` 不保存
environment callback、resolver registration 或 app runtime dependency。

measure adapter package re-export shared cfg public names，而且identity必須相同；它
不保留舊 model/inheritance/codec wrapper implementation。finished-cfg generic algorithm與
三個窄runtime ports由 [[0046]] 擁有；measure保留caller-facing `validate_schema`與
`schema_to_raw_dict`，在adapter內組app-owned ports。

autofluxdep 的 model、inheritance 與 codec import 直接指向 `gui.cfg`。它的
`NodeCfgSchema.logical_paths`、generation persistence reshape、`OverridePlan`、node builder
與 pulse/readout spec factory仍由 app/domain layer 擁有。autofluxdep的local lowering與
module conversion由 [[0046]] 定義；Qt form seam仍是精確列出的app coupling。

## 後果

- shared cfg package 可在 fresh process 中不載入 app、experiment、Qt 或 `meta_tool`。
- measure 與 autofluxdep 使用同一組 class/function identity與相同 codec wire shape。
- finished-cfg caller 改用 app-local free function，observable validation/lowering順序不變。
- generic lowering的ownership與ports由 [[0046]] 定義。

## 拒絕的替代方案

- **shared method runtime import measure app**：破壞 shared → app 單向依賴。
- **process-global resolver registration**：引入 import-order side effect與隱性 global state。
- **把 broad environment port 塞進 `CfgSchema`**：純 data carrier 承擔 app runtime責任。
- **只信 linked ref embedded snapshot**：會改變 missing library key 與 unsupported shape 的
  既有 validity/error contract。
