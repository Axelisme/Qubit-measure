---
status: accepted
---

# 0045 — Shared GUI cfg core ownership

**狀態：** accepted（2026-07-10）。
**關聯：** [[0009]]、[[0010]]、[[0011]]、[[0012]]、[[0043]]、[[0046]]。

## 背景

measure-gui 與 autofluxdep-gui 共用同一套 Spec/Value tree、完整 value tree
繼承規則與 persistence codec。這些純資料與演算法屬於shared GUI cfg core；app package
只擁有各自的runtime policy、adapter與domain API。

linked Module/Waveform reference 的 finished-cfg validation/lowering需要即時查詢
app-owned library shape policy。[[0046]]以窄resolver port提供此能力，shared core不
反向依賴app runtime。

## 決策

`zcu_tools.gui.cfg` 擁有以下 app-independent Qt-free core：

- `model.py`：Spec/Value 型別、`CfgSchema` 的 `spec`/`value` data carrier；
- `inheritance.py`：default value、locked literal alignment、inheritance；
- `tree.py`：section/reference-aware Spec resolve與existing-leaf Value read/replace；
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

measure adapter package只暴露framework contract、request/result/writeback/analyze params與
protocol signature需要的`gui.session.types` vocabulary；不import或forward shared cfg public
names。generic consumer直接依賴shared owner，model/inheritance/codec implementation只由shared
core擁有。finished-cfg generic algorithm與三個窄runtime ports由 [[0046]] 擁有；measure保留
caller-facing `validate_schema`與`schema_to_raw_dict`，在adapter內組app-owned ports。

autofluxdep caller直接從shared owner匯入generic model、inheritance、codec names與Qt form。
`zcu_tools.gui.app.autofluxdep.cfg` package barrel只暴露`NodeCfgSchema`、OverridePlan/policy、
module reference spec helpers與其它app-owned API，不forward `zcu_tools.gui.cfg` public names。
`cfg.module_adapter`擁有pulse/readout/waveform conversion與spec functions；
`NodeCfgSchema.logical_paths`、generation persistence reshape、`OverridePlan`、node builder與
pulse/readout spec factory由app/domain layer擁有。autofluxdep local lowering與module
conversion遵循 [[0046]]，production不import measure app。

## 後果

- shared cfg package 可在 fresh process 中不載入 app、experiment、Qt 或 `meta_tool`。
- shared Qt cfg widget可重用同一份draft renderer而不載入任何app/session policy。
- renderer註冊是instance-owned、fixed-factory、exact且freeze後不可變，不受import order影響。
- measure 與 autofluxdep 使用同一組 class/function identity與相同 codec wire shape。
- generic cfg consumer直接依賴shared owner；measure adapter與autofluxdep local barrel只提供
  app-owned API。
- finished-cfg caller使用app-local free function，observable validation/lowering順序不變。
- generic lowering的ownership與ports由 [[0046]] 定義。

## 拒絕的替代方案

- **shared method runtime import measure app**：破壞 shared → app 單向依賴。
- **process-global resolver/renderer registration**：引入 import-order side effect、
  inheritance fallback與隱性 global state。
- **把 broad environment port 塞進 `CfgSchema`**：純 data carrier 承擔 app runtime責任。
- **只信 linked ref embedded snapshot**：會改變 missing library key 與 unsupported shape 的
  既有 validity/error contract。
