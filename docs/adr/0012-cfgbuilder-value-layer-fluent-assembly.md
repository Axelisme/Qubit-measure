---
status: accepted
---

# CfgBuilder — value 層引入組裝 builder（對照 0009 spec 層不用 builder）

## 脈絡

adapter 的 `make_default_value(self, ctx)`（~20 個各寫，未來 `experiment/v2/` 全部實驗都做 adapter
後將是 50+）現況是手拼一棵巢狀 `CfgSectionValue(fields={...})` literal，並手動把各 per-role L2
factory（`make_<role>_default(ctx)` / `make_<role>_ref_default(ctx, optional=)`）穿進去（如
`twotone/freq.py` 的 17 行）。重複的痛點：頂層 scalar 每次手填 `DirectValue(...)`、modules 那坨 ref
要 import 一串 factory 再拼、sweep 要手建 `SweepValue`。用戶要一個降低 adapter 實作難度的工具。

grill 過程定錨的事實（推翻 HANDOFF 對 backlog 2 的舊描述）：

- backlog 2 設想的「`default_value_from(ctx, spec)` + 明文 dict + type 字串比對 ref」**主體已被
  ADR-0009 的 `shared/defaults/` per-role factory 取代**；HANDOFF 是舊快照。
- 預設值是**角色相關**非 field-name 相關（同名 `gain` 在 qub_probe=0.05、readout=0.1），扁平 dict 解
  不了；領域默認**大量是公式**（`trig_offset=timeFly+0.05`、`post_delay=5/(2π·rf_w)`），純資料表裝
  不下，必須是代碼（現 L2 factory 即是）。
- 「角色」已 first-class：`RoleCatalog`+`RoleEntry`，Inspect 建 module/waveform 用 L2 factory（第二
  消費者）→ L2 簽名鎖定，**不能被 builder 收編成私有 step**。

## 決策

### 1. 三層分工：L1 blank / L2 role default / L3 CfgBuilder

- **L1** `make_default_value(spec)`（gui 框架層，`inheritance.py`，不動）：吃 spec，回結構完整、值中性
  骨架（scalar→0/0.0/""、optional ref→None、required ref→`allowed[0]` blank）。
- **L2** `make_<role>_default(ctx)` / `make_<role>_ref_default(ctx, *, optional=)`（領域層
  `shared/defaults/`，**簽名不動**）：單角色 ModuleRefValue/WaveformRefValue + 領域默認（含公式
  EvalValue + library 查找）。RoleCatalog/Inspect 與 CfgBuilder 是**同一批 L2 函數的兩個平行消費者**。
- **L3** `CfgBuilder`（新增 `shared/cfg_builder.py`，**領域層**）：flat-path fluent 組裝。起手 = L1 blank
  骨架（完整性 by construction），`.role()` 經 `ROLE_FACTORIES` 表調 L2，`.scalars()/.set()` 走
  scalar leaf，`.sweep()/.set_sweep()` 掛 SweepValue，`.build()` 回 `CfgSectionValue`。

### 2. value 層引入 builder —— 與 ADR-0009 spec 層「不用 builder」並行不悖

ADR-0009 決策 2 否決了 **spec 層** 的可變 builder（理由：spec 是純結構、鎖定是離散「一條條點名 leaf
→ LiteralSpec」、無聚合邏輯，frozen+replace 回同型已天然鏈式，builder 的 `.build()` 是多餘氣味）。

**這條否決理由在 value 層不成立**：value 的組裝**有領域聚合邏輯**——「reps 該是 100」「readout 該查
library 的 readout_dpm」「freq 該是 EvalValue(q_f)」是有內容、有策略、跨多欄位協調的決策，spec 層沒有
的東西 value 層有。所以「spec 不用 builder」推不出「value 不用 builder」。

更關鍵：這堆領域邏輯（role / ctx / library 查找）**不能下沉到框架層的 `CfgSectionValue`**。若給
`CfgSectionValue` 加 `with_role(path, role_id, ctx, ...)`，則 gui 框架層的純數據容器要懂 qubit role +
RoleCatalog + library，且簽名一半要 ctx 一半不要（`with_field` 不要）——跨層污染 + API 不一致。
**Builder 住在領域層，正是這堆邏輯的正確歸屬**：領域知識聚在 Builder，`CfgSectionValue` 保持框架層純結構
（只懂 fields dict + scalar-leaf path setter）。這比 `CfgSectionValue.with_role` **更**符合 0009 的
spec/value 分層、責任明確精神。`.build()` 在此是「領域組裝器 → 純數據產物」的合法邊界，與 spec 層那個
「無聚合邏輯、沒必要 build」的情況性質不同。

**被否決**：`default_value_from(ctx, spec)` 回 `CfgSectionValue` + 在其上加 `with_role` 鏈式 method。
否決理由：`with_role` 把領域邏輯塞進框架層數據容器（上述跨層污染）。

### 3. Builder 不走 RoleCatalog，領域層直調 L2 + 自己的 ROLE_FACTORIES 表

事實：adapter 經 `Registry.create()` 無參構造、`make_default_value(self, ctx)` 只收 ctx，
`ExpContext` **不帶 RoleCatalog**（catalog 只在 Controller）；且 catalog factory **永不回 None**
（表達不了 optional→None，那能力只在未註冊的 `_ref_default`）。→ Builder 在領域層直接 import L2 函數，
經單一 `ROLE_FACTORIES` 表（`defaults/role_factories.py`，registry + builder 共用 source）取
`{role_id: (blank, ref)}`。Inspect/catalog 用 `.blank`；Builder 的 `.role()` 用 `.ref`（library
aware）或 `prefer_blank=True` 強制 `.blank`（守 ignore_library_readout）或 `optional=True` 走
`.ref(ctx, optional=True)`（library miss→None）。

### 4. Builder 零鎖定（但自動「填」鎖定值）、value in-place、不 validate（守既有 ADR）

- 零鎖定 vs 自動填：**鎖定的「宣告」100% 在 `cfg_spec().lock_literal`（ADR-0009 決策 5），Builder 不宣告任何鎖**；但 Builder **替 adapter 填鎖定值**——`build()` 遍歷 spec 樹，把每個 `LiteralSpec` leaf 的 value 對齊成 `spec.value`（穿透已掛載 ref 的 chosen shape，復用框架層 `find_allowed_spec`）。動機：L1 blank 已對齊頂層 literal，但 `.role()` 掛的 L2 factory value 不懂 ref shape 內的鎖（如 onetone/freq 在 `pulse_readout` shape 內鎖 `pulse_cfg.freq=0.0`，但 L2 產 `freq=r_f`），覆蓋後 locked leaf 帶錯值 → build 對齊修正。配套 **`.set` 碰 locked path 直接 raise（C-raise）**：locked 值的唯一 source 是 spec，adapter 既不該也不需手設（消掉舊手拼裡 `freq=0.0`/`ro_freq=0.0` 與 spec 鎖值的重複）。這與 ADR-0011 的 validate（事後查 value==spec.value）互補——build 是事前保證。**lower 本就不看 locked value（`lowering` 對 LiteralSpec 直接用 `spec.value`），所以填對齊只為過 validate + 語意一致**。
- value in-place（ADR-0009 決策 4）：Builder 持一棵 in-place mutate 的 value，每個 method 只覆寫已存在
  節點、不增刪 key（守 ADR-0010 完整性）。
- `.build()` 不 validate：驗證留在 cfg 邊界（`make_default_cfg` / `to_raw_dict`，ADR-0011）。
- 全方法 Fast-Fail（spec-aware）：bad path / 非 ref spec / kind 不匹配 / 對 required ref 用 optional /
  `.sweep` 收到 EvalValue 邊界 → 立即 raise，指向出錯的呼叫。

### 5. 掛整節點能力 = Builder 私有 `_mount_node`

`.role()/.sweep()/.set_sweep()` 要把整節點（ModuleRefValue/WaveformRefValue/SweepValue/None）塞進
value 樹的 path 位置，但框架層 `CfgSectionValue.with_field` **只設 scalar leaf**（強制包 Direct/Eval）。
→ Builder 自帶私有 `_mount_node`（下鑽 path 父節點塞整節點），**不擴 `with_field`**——框架層容器的
scalar-only 契約保持不變（目前無第二消費者要「按 path 換整節點」，下沉是 YAGNI）。

## 後果

- 新檔 `shared/cfg_builder.py`（`CfgBuilder`）+ `shared/defaults/role_factories.py`（`ROLE_FACTORIES`
  單一 source）。`registry.py` 的 `ROLE_ENTRIES` 改從表取 blank factory（消除 factory 重複 source，
  catalog 仍持有自己的 label/dropdown 順序）。
- `ScalarLeafInput` 加進 `gui/app/main/adapter/__init__.py` 導出（`.set/.scalars` 的公開入參型別）。
- adapter `make_default_value` 漸進遷移（舊手拼寫法仍合法，產出同樣的 `CfgSectionValue`，無 big-bang、無
  compat shim）。pilot：`twotone/freq`、`lookback`、`amp_rabi`。
- 映射規則：`make_<role>_default` → `.role(path, role, prefer_blank=True)`；`make_<role>_ref_default(ctx)`
  → `.role(path, role)`；`make_<role>_ref_default(optional=True)` → `.role(path, role, optional=True)`；
  無條件禁用的 optional ref（lookback 的 init_pulse/reset）→ 不寫（L1 blank 已給 None）。

## 替代方案（綜述）

| 維度 | 選擇 | 否決 |
| --- | --- | --- |
| 組裝形狀 | 領域層 CfgBuilder（持 ctx+spec+value，`.build()`） | `default_value_from` 回 value + `CfgSectionValue.with_role`（跨層污染框架容器） |
| role 解析 | 領域層直調 L2 + ROLE_FACTORIES 表 | 走 Controller-held RoleCatalog（adapter 夠不著、表達不了 optional→None） |
| 掛整節點 | Builder 私有 `_mount_node` | 框架層 `CfgSectionValue.with_node` 公開（擴框架容器、無第二消費者、YAGNI） |
| sweep API | `.sweep(path, start: float, stop, expts)` 純字面 + `.set_sweep(path, SweepValue)` 逃生口 | `.sweep` 也收 EvalValue 邊界（與 set_sweep 能力重疊、調用者要糾結用哪個） |
| scalar 默認 | `.scalars()` 純顯式無表 | 內建 `DOMAIN_SCALAR_DEFAULTS` 表（各 adapter relax_delay/rounds 本就不同，表藏意圖） |
