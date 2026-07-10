---
status: accepted
---

# Measure cfg definition — context-free single-pass authoring

**狀態：** accepted（已實作）。
**關聯：** Spec/Value 與 locked literal 見 [[0009]]；完整 value tree 見 [[0010]]；成品驗證見
[[0011]]；resolve-once source 見 [[0037]]；shared cfg core ownership 見 [[0045]]。

## 脈絡

measure adapter 的 cfg 同時需要 static shape、module role、locked leaf、MetaDict-linked scalar、
ModuleLibrary adoption、sweep 與 fallback。若作者分別維護 spec 與 default value 兩棵宣告，field
order、lock value、role shape 與 fallback 很容易漂移；但把 role、MetaDict 或 ModuleLibrary policy
下沉到 shared cfg core，又會破壞 app-independent boundary。

adapter 是 user-maintained experiment flow。authoring surface 必須由上到下可讀，讓 field type、
label、default source、role、lock 與 override 保持 locality；同時 static shape 不應因 fresh cfg 的
runtime context 改變。

## 決策

### 1. 資料模型分離，authoring 集中

Spec tree 與 Value tree 仍是兩棵責任不同的資料樹：Spec 描述 static contract，Value 保存 mutable
draft。adapter 作者只維護一份 context-free `MeasureCfgDefinition`：

```python
@classmethod
def cfg_definition(cls) -> MeasureCfgDefinition:
    return (
        MeasureCfgBuilder()
        .reset("reset", optional=True)
        .pulse("pi_pulse", role_id="pi_pulse", label="Pi Pulse")
        .readout("readout")
        .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
        .sweep("length", label="Delay (us)", default=...)
        .reps(1000)
        .rounds(100)
        .build()
    )
```

`MeasureCfgBuilder` 不接收 `ExpContext`。它按呼叫順序保存 immutable ordered declarations，並在
`build()` 時固定 static spec。`MeasureCfgDefinition.instantiate(ctx)` 只能解析 fresh default，不能
增減 field、改型別、label、order 或 lock；materialized spec 必須精確等於 definition spec，否則
Fast Fail。

`BaseAdapter.make_default_cfg(ctx)` 是 framework-facing fresh-cfg 入口：instantiate definition，接著
以 `validate_schema(schema, ctx.ml)` 驗證成品。framework protocol 不暴露 static spec query；
`cfg_definition()` 是 `BaseAdapter` 的 authoring hook。

### 2. deferred seed vocabulary 小而 typed

measure domain 以 `Seed[T]` 表達「definition 現在描述來源，fresh cfg 建立時才解析」。public
vocabulary 維持最小：

- `literal(value)`：authoring-time snapshot；raw default 由 builder 自動 lift；
- `md(key, fallback, expr?)`：有 MetaDict key 時保留 live `EvalValue`，否則使用明文 fallback；
- `scaled_md(key, factor, fallback_value)`：明確區分 missing 時的最終值；
- `value_source(key, target_type, fallback?)`：透過 [[0037]] 的 read-only lookup resolve once；
- `custom(resolve, description)`：低頻純 context lookup escape hatch；
- `SweepDefault`：只組裝 typed sweep edges，不引入另一套 resolver algebra。

常用 range policy可提供具名 seed factory。不得加入 map/zip、conditional AST、priority graph 或依
field name 猜數值的全域 resolver。`custom` callback 必須無副作用並回傳精確 carrier；restore 與使用者
編輯不會重跑 seed。

### 3. measure builder 擁有 domain verbs

`MeasureCfgBuilder` 擁有高頻且穩定的 measure vocabulary：`pulse`、`readout`、`reset`、`sweep`、
`relax_delay`、`reps`、`rounds`、`device`，以及 generic scalar/`field` escape hatch。module 與 sweep
verb自動落到 `modules.<name>` / `sweep.<name>`；低頻 experiment knob 使用完整 path，不為單一
adapter 增加新 verb。

module declaration 同地保存 `role_id`、initialization、`blank_overrides`、`overrides` 與 `locked`：

- `ModuleInit.SMART`：required role adopt calibrated library entry or blank；optional role adopt or
  `None`；
- `ModuleInit.INLINE`：永不查 library，使用 fresh role blank；
- `ModuleInit.DISABLED`：只允許 optional ref，永遠 materialize `None`；
- `blank_overrides` 只修改 inline/fallback custom value；`overrides` 同時作用於 custom 與 adopted
  snapshot；
- `locked` 把 static leaf變成 `LiteralSpec`；不得與 override path重疊，materialization由 shared
  assembler 對齊 literal value。

role alias priority、shape 與 md-linked seed的 SSOT仍是 [[0009]] 的 `ROLE_TABLE` / `ROLE_FACTORIES`。
builder只消費這份資料，不掃描任意 library name，也不複製 role policy。

### 4. shared assembler 只共用 mechanics

`zcu_tools.gui.cfg.CfgSchemaAssembler` 同步宣告 paired Spec/Value tree，負責 duplicate/parent conflict、
default wrapping、choice binding、locked alignment、deep-copy snapshot與 one-shot build。它不認識
`ExpContext`、Seed、role、MetaDict、ModuleLibrary、logical key或 generation policy（[[0045]]）。

measure definition在 instantiate 後把 resolved defaults交給 assembler；autoflux
`NodeSchemaBuilder`也使用同一 mechanics，但保留自己的 logical path、generation與section label policy。
兩個 domain builder不共用 DSL。

## 取捨與被拒絕方案

- **採用 context-free definition，而非 `MeasureSchemaBuilder(ctx)`**：多一個 typed deferred carrier，
  換得 spec shape by construction。cross-context tests是 regression gate，不是結構保證的替代品。
- **採用 single-pass authoring，而非 split spec/default hooks**：Spec/Value資料模型仍分離，但作者不再
  重複描述相同 field，消除 drift source。
- **採用小型 seed vocabulary，而非通用 expression DSL**：特殊 policy用具名 factory或單一 typed
  `custom` escape hatch，避免 framework programming滲入 adapter data。
- **只共用 assembler，而非統一 measure/autoflux builders**：兩者只共享 tree mechanics；domain
  vocabulary、runtime dependency與logical projection維持各自 ownership。

## 後果

- 每個 concrete adapter只實作一個 `cfg_definition()` authoring block。
- fresh default deterministic；library priority與fallback在 role/seed declaration可追蹤。
- locked/override path會先完整 preflight，再 materialize，失敗不留下 partial tree。
- definition與每次 instantiate都隔離 mutable spec/value aliases，可安全重用。
- framework刪除 static spec forwarding；需要 cfg 的 caller一律取得 validated `CfgSchema`。
