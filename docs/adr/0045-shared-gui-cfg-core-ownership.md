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

`zcu_tools.gui.cfg` 擁有以下 app-independent Qt-free core：

- `model.py`：Spec/Value 型別、`CfgSchema` 的 `spec`/`value` data carrier；
- `inheritance.py`：default value、locked literal alignment、inheritance；
- `codec.py`：session/node cfg raw ↔ live value tree codec 與 `SessionCodecError`；
- `lowering.py`：[[0046]] 定義的 generic finished-cfg algorithm、三個窄 callable
  ports與 validation/lowering operations；
- package `__init__`：re-export上述 app-independent core public names，包含 [[0046]] 的
  `ExpressionResolver`、`ReferenceResolver`、`RangeFactory`、`validate_finished_cfg`與
  `lower_finished_cfg`。

`zcu_tools.gui.widgets.cfg`擁有shared Qt cfg renderer：`CfgFormWidget`、field widgets、
decoration presentation contract與renderer registry。每個form持有一個
`FrozenFieldRendererRegistry`；default factory每次建立新builder，為`LiteralField`、
`ScalarField`、`SweepField`、`CenteredSweepField`、`SectionField`與`ReferenceField` exact註冊
固定`FieldRenderer(field, FieldRenderContext)` factory後freeze。immutable context只保存path、
top-level、label width、decoration resolver、text enhancer與同一registry，不保存
controller/service/app/runtime data。root、section child與reference subtree都由registry
`render()`建構，不使用consumer-side constructor分支、module-global mapping、decorator、
string key或inheritance fallback。

registry在registration boundary用call signature bind拒絕錯誤factory；frozen render boundary
驗證回傳值同時符合`QWidget`與field-widget protocol。`CfgFormWidget.attach()`先完成root build再
訂閱caller-owned draft；build失敗不留下draft callbacks。detach使用stable Python callback解除
change/validity subscriptions並刪除Qt tree，但不close draft。

shared widget只import Qt、`gui.cfg`、`gui.cfg.binding`與`gui.widgets.spinbox`，不import
`gui.app.*`、experiment、`meta_tool`、session、EventBus、device或notebook。它接受generic
text-input enhancer與decoration provider ports；measure的ValueSource enhancer與autofluxdep的
`OverridePlan` provider留在各自app。

`gui.cfg` 不 import `gui.app.*`、`experiment.*`、Qt 或 `meta_tool`。`CfgSchema` 不保存
environment callback、resolver registration 或 app runtime dependency。

measure adapter package re-export shared cfg public names，而且identity必須相同；它
不保留舊 model/inheritance/codec wrapper implementation。finished-cfg generic algorithm與
三個窄runtime ports由 [[0046]] 擁有；measure保留caller-facing `validate_schema`與
`schema_to_raw_dict`，在adapter內組app-owned ports。

autofluxdep 的 model、inheritance、codec與Qt form imports直接指向shared owners。它的
`NodeCfgSchema.logical_paths`、generation persistence reshape、`OverridePlan`、node builder
與 pulse/readout spec factory仍由 app/domain layer 擁有。autofluxdep的local lowering與
module conversion由 [[0046]] 定義；production不import measure app。

## 後果

- shared cfg package 可在 fresh process 中不載入 app、experiment、Qt 或 `meta_tool`。
- shared Qt cfg widget可重用同一份draft renderer而不載入任何app/session policy。
- renderer註冊是instance-owned、fixed-factory、exact且freeze後不可變，不受import order影響。
- measure 與 autofluxdep 使用同一組 class/function identity與相同 codec wire shape。
- finished-cfg caller 改用 app-local free function，observable validation/lowering順序不變。
- generic lowering的ownership與ports由 [[0046]] 定義。

## 拒絕的替代方案

- **shared method runtime import measure app**：破壞 shared → app 單向依賴。
- **process-global resolver/renderer registration**：引入 import-order side effect、
  inheritance fallback與隱性 global state。
- **把 broad environment port 塞進 `CfgSchema`**：純 data carrier 承擔 app runtime責任。
- **只信 linked ref embedded snapshot**：會改變 missing library key 與 unsupported shape 的
  既有 validity/error contract。
