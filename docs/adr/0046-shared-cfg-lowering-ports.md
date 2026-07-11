---
status: accepted
---

# 0046 — Shared cfg lowering ports

**狀態：** accepted（2026-07-10）。
**關聯：** [[0011]]、[[0043]]、[[0045]]、[[0051]]。

## 背景

measure-gui 與 autofluxdep-gui 使用相同的 finished-cfg static validation、dynamic
validation 與 lowering 語意，但 expression evaluation、ModuleLibrary shape policy 與
`SweepCfg` construction 分別屬於 app/runtime。把整個 `MetaDict`、`ModuleLibrary` 或 broad
environment object 放進 shared cfg core，會讓純 Spec/Value package 反向依賴 app policy；
讓 autofluxdep 呼叫 measure adapter則會形成 app-to-app coupling。

[[0045]] 先下沉 model/inheritance/codec，並把 finished-cfg lowering 暫留 measure app。
本決策取代該暫留安排；[[0045]] 的純 data ownership與 import boundary仍然有效。

## 決策

`zcu_tools.gui.cfg` 的 reference model只有一組 app-neutral type：frozen
`ReferenceSpec(kind, allowed, label, optional)` 與 mutable
`ReferenceValue(chosen_key, value, is_overridden)`。`kind`是required、non-empty的app-local
opaque id；shared traversal只轉送或保存它，不定義合法值。`allowed`維持non-empty，spec
override與value `with_field`各只有一份generic實作。

`zcu_tools.gui.cfg.lowering` 擁有 generic finished-cfg algorithm，public surface只有三個
callable ports與三個操作：

- `ExpressionResolver(expr) -> int | float`：解析單次 lowering綁定的 current context；
- `ReferenceResolver(kind, key) -> str | None`：回傳 module/waveform concrete
  `CfgSectionSpec.label`，`None` 精確表示 live key missing；
- `RangeFactory(start, stop, *, expts) -> object`：建立 app runtime range object；
- `validate_reference_kinds(schema, allowed_kinds)`：以 caller-owned collection驗證opaque
  kind，不在shared定義合法值；
- `validate_finished_cfg(schema, *, resolve_reference)`：執行 static validation；
- `lower_finished_cfg(...)`：固定執行 static → optional dynamic → lower。

shared core負責既有 path、`RuntimeError` text、first-error Fast Fail、numeric coercion、
centered sweep contract、snapshot precedence與 drift warning。傳入 expression resolver時才
執行 dynamic validation；沒有 resolver時，只有已有 snapshot的 `EvalValue` 可 lower。

linked custom ref只按 embedded custom label選 allowed shape，不呼叫 reference port。linked
library ref每個 validation/lowering stage都即時呼叫 resolver，不快取結果；live key不存在時
維持 missing error，同名 key重現時重新 relink。resolver只決定 live key與 concrete shape，
lowered內容仍取 embedded snapshot。app-owned conversion exception不包裝。

`gui.cfg` 不 import `gui.app.*`、`experiment.*`、`meta_tool.*`、Qt、`notebook`或`device`，
也不提供 `LoweringEnv`、process-global resolver registry或 environment lookup。

measure adapter保留 `validate_schema(schema, ml)` 與
`schema_to_raw_dict(schema, md, ml)` caller interface，並在呼叫內組 expression/reference/range
ports；measure與autofluxdep adapter在validation/lowering前各自套用app-owned合法kind集合。
autofluxdep使用獨立 local lowering adapter與 local pulse/readout/waveform conversion；
defaults conversion只接受 `pulse`與`readout/pulse`，reference shape resolver透過 [[0051]] 的
closed catalog辨識所有合法 measure module discriminator，使合法但不允許的 shape由 shared core
回報 unsupported spec。
兩個app的pulse/waveform factory都顯式建立`ReferenceSpec(kind="module")`或
`ReferenceSpec(kind="waveform")`，resolver/converter依`spec.kind`分派；shared core不import
這些app policy。既有persisted `module_ref`/`waveform_ref` discriminator、payload shape與
missing/relink semantics保持不變。

`NodeCfgSchema.logical_paths`、generation persistence、`OverridePlan`、Qt form/live model與
node builders不屬於本決策，維持 [[0043]] dataflow。

## 後果

- measure與autofluxdep共用一份 validation/lowering algorithm，app只提供窄 runtime policy。
- autofluxdep production lowering/module conversion不 import measure app；目前跨 app seam只剩
  `cfg/form.py` 的 Qt/live-model reuse。
- missing → invalid、same-name relink、embedded snapshot lowering與`SweepCfg` output維持不變。
- shared model不再以平行module/waveform class表達同構reference state；新增kind不需擴張shared
  type hierarchy。
- `CfgSchema`繼續是純 spec/value data carrier，沒有 runtime dependency或global registration。

## 拒絕的替代方案

- **把 `MetaDict` / `ModuleLibrary` 放進 shared core**：破壞 import purity與 app ownership。
- **建立 broad `LoweringEnv`**：把三種獨立能力綁成難以驗證的 environment object。
- **process-global resolver registry**：引入 import-order side effect與跨 session hidden state。
- **只信 embedded reference snapshot**：破壞 live missing/relink與unsupported-shape contract。
- **autofluxdep沿用 measure conversion**：保留 app-to-app dependency，讓兩個 app policy無法獨立演化。
