---
status: accepted
---

# ModuleLibrary / MetaDict 內容寫入的唯一權威 = ContextService（經窄 write port）

## 脈絡

ml/md 的內容寫入曾有三條並存路徑：(1) `ContextService.set_ml_*_from_raw` / `set_md_attr`（canonical：has_context 守衛 + factory 反序列化 + `version.bump("context")` + emit `ML/MD_CHANGED`）；(2) `WritebackService.apply_tab_writeback` **直寫** `ctx.md`/`ctx.ml`（`setattr` / `register_module`）並內聯 `schema_to_dict` + `ModuleCfgFactory.from_raw`，自己 bump+emit；(3) `inspect_dialog._on_save` 在 UI 內 `schema_to_dict`（且漏傳 md）後呼 `set_ml_*_from_raw`。

(2)(3) 是「為達效果繞過 owning abstraction」的反模式（與先前移除的 device `set_value` 同族）：重造寫入邏輯、lowering 散落多處且語義分歧（Writeback 傳 md、Inspect 漏 md），違反 ADR-0005 違規3（aggregate/infra 寫入未經 port）。

## 決策

**所有 ml/md 內容寫入必經單一權威 `ContextService`，對外以一個窄 write port 暴露。** ContextService 是 md/ml aggregate 的事實 owner（已集中 has_context 守衛、version bump、event emit）。

- 採**選項 A**（單一權威），否決選項 B（「任何帶 bump+emit 的 sink 皆可」——B 允許多寫入實作，正是要消滅的分裂）。
- port 是 ContextService 對外的**窄寫入契約**（interface segregation：只列 consumer 實呼的方法），**背後唯一實作就是 ContextService**。Writeback / Inspect 依賴此 port 型別，不再自己 `setattr`/`register_*`/`from_raw`。
- 違反此不變式（任何 service/UI 直接碰 `ctx.ml`/`ctx.md` 寫入）視為架構違規。

## 統一職責線（lowering 與 register 的歸屬）

**來源端只到「未-lower 的 CfgSchema 快照」為止；CfgSchema→concrete + 寫 ml/md 一律 ContextService。**

- **來源端**（CfgEditorSession / WritebackItem model / Inspect form）：把 LiveModel draft 快照成 `CfgSchema(spec, value)`（含 EvalValue，**未 lower**）。
- **ContextService**：收 `CfgSchema` → `schema_to_dict(schema, ml, **md**)` lower（傳 md 解析 EvalValue + drift 檢查）→ `Factory.from_raw` → `register_*` → `bump("context")` → emit。**lowering 收進 ContextService**（否則 md 洩漏給呼叫端管，正是 Inspect 漏 md 的根）。

### 對 ADR-0008 的修正（partial supersede）

ADR-0008 原述「commit 時 aggregate（CfgEditorSession）lowers + registers itself through the ML write port」。**本 ADR 修正其中的 register（寫 ml）**：register 不再屬 editor aggregate，移到 ContextService。

- `CfgEditorSession.commit` 不再呼 `set_ml_*_from_raw`、不再 lower。它只交出 **CfgSchema 快照**。
- `CfgEditorService.commit(editor_id, name)` 改為：取 session 的 CfgSchema → 呼 ContextWritePort 寫入。
- `ModuleLibraryWritePort` 的**寫入面萎縮**：editor 只需 `get_current_ml`/`get_current_md`（seed `from_name` + 不再需要的 lowering 已移走），移除 `set_ml_*_from_raw`。
- 不變：ADR-0008 的 LiveModel service-owned / attach-detach / gc 分流 / seeded teardown-only 全保留。

### port 方法集（窄）

`ContextWritePort` 暴露：`set_ml_module_from_schema(name, schema)` / `set_ml_waveform_from_schema(name, schema)` / `set_md_attr(key, value)` / batch 入口。三來源端（editor commit / writeback apply / inspect save）依賴此 port 型別，唯一實作 = ContextService。`set_ml_*_from_raw` 移除後，`from_schema` 內部 = `schema_to_dict(schema, ml, md)` lower → `Factory.from_raw`(內部反序列化，非 public 寫入入口) → register → bump → emit，唯一寫入路徑。

### Batch 寫入（Writeback apply）

Writeback 一次 apply 套多 item（md + 多筆 ml）必須是 **1 次 `bump("context")` + 各型別最多 1 次 emit**（ML_CHANGED/MD_CHANGED 的消費者是粗粒度全刷；逐筆 N 次 = N 次無謂全刷 + 版本亂跳）。

- ContextService 提供 **batch 寫入入口**（選項 i：如 `apply_context_writes(...)` / `with batch_write()` 風格），「一批 = 一次 bump + 一次 emit」是 ContextService 內部保證。
- **編排留 WritebackService**（它懂 item 的 selected / type 分派 / session_id 記錄）；它收集「要寫什麼」交一次 batch 呼叫。bump/emit 時機不洩漏給呼叫端（否決 defer 旗標 / 逐筆 N 次）。

### F2 Inspect = committable session（非 seeded）

Inspect 編輯既有 ml entry 的本質**就是** agent「`open(from_name)` → edit → `commit`」ml-entry 編輯流的 UI 版。當初用 `open_seeded`（拒 commit）+ UI 自建 schema + UI lower + `from_raw` 是偏差（seeded 較好開的偷懶）。

- Inspect 改用 `open(item_kind, from_name=name)`（已支援從 ml 直接載入既有 entry，`_initial_schema` from_name 分支）；存檔走 `commit_cfg_editor(editor_id, name)`。
- 消除 `_schema_from_cfg` / `module_cfg_to_value` / `waveform_cfg_to_value` 自建 schema、UI `schema_to_dict`、`set_ml_*_from_raw`。與 agent 路徑共用、WYSIWYG 自然成立。
- `editor.commit` 在本 ADR 後 = 取 session CfgSchema → ContextWritePort 寫入；故 Inspect/agent/未來 ml-entry 編輯三者寫入完全同一條 port。

### Port 拆分（讀寫分離，命名對稱）

`ModuleLibraryWritePort`（讀+寫混合）拆為：

- **`ContextReadPort`**：`get_current_ml()`（CfgEditorService seed `from_name` 用）。lowering 移走後 editor 不再需 `get_current_md`，故 read port 只列 `get_current_ml`（interface segregation；ContextService lowering 讀自己的 ml/md 不經 port）。
- **`ContextWritePort`**：`set_ml_module_from_schema` / `set_ml_waveform_from_schema` / `set_md_attr` / batch 寫入入口。

`CfgEditorService → ContextWritePort` 是合法 **Command 邊**（[[0004]]）：已查證 ContextService 不反依賴 CfgEditor，天然單向不成環。

### 移除 raw 入口，ml 寫入路徑徹底單一化（user/agent 同路）

原以為 `set_ml_*_from_raw`（收已-lower raw dict）有正當 consumer 須保留。深查後**推翻**：四個 caller 全部其實持有或應持有 CfgSchema —

1. **agent `context.set_ml_module`/`waveform` RPC** —— 與 editor session 路徑（`create_from_role`→`editor.open(from_name)`→`set_field`(含 EvalValue)→`commit`）**功能重疊且較差**：無法表達 EvalValue（raw 須 concrete，閹割了 ml entry「值跟 md 走」的核心價值）、不經 LiveModel 驗證、逼 agent 自組與 spec 耦合的巢狀 dict。判定為早期偷懶遺留（與 device set_value 同源）。→ **移除 `context.set_ml_module`/`set_ml_waveform` RPC + tool**；agent 統一走 editor session。
2. **`create_from_role`**（controller.py）—— 它**已持有 CfgSchema** 卻先在 UI 端 `schema_to_dict(...,ml,md)` lower 再 from_raw（第四個 UI-lowering 現場）。→ 改走 `from_schema`。
3. **editor commit** → from_schema（CfgEditorService.commit 經 ContextWritePort）。
4. **Writeback / Inspect** → from_schema / editor.commit。

**結論：四 caller 全走 CfgSchema → `set_ml_*_from_raw` 整個移除，ml 寫入唯一入口 = `set_ml_*_from_schema`。** 達成「user（Inspect→editor.commit）與 agent（editor session）寫 ml 同一條路徑」。

**md 寫入**（`set_md_attr`）不受影響：scalar 值，無 lowering，保留。

### 改動面（三來源端收斂到同一寫入權威）

| 來源端 | 改動 |
| --- | --- |
| agent ml RPC | **移除** `context.set_ml_module`/`set_ml_waveform`（RPC + method_spec + mcp tool + `_h_context_set_ml_*` handler）；agent 改走 editor session |
| create_from_role | controller.py 內 `schema_to_dict`+`set_ml_*_from_raw` → `set_ml_*_from_schema`（它已持 CfgSchema） |
| editor commit（agent + Inspect + 未來 ml-entry 編輯） | `CfgEditorSession` 不再 lower/register，只交 CfgSchema；`CfgEditorService.commit` 呼 ContextWritePort |
| writeback apply | 移除 `_resolve_*_item`（內聯 lower+from_raw）+ 直寫 `ctx.ml`/`ctx.md`；改收集 writes → ContextService batch 入口 |
| inspect save | `open_seeded`→`open(item_kind, from_name)`；`_on_save`→`commit_cfg_editor`；刪 `_schema_from_cfg`/`module_cfg_to_value`/UI `schema_to_dict`/`set_ml_*_from_raw` |
| ContextService | 刪 `set_ml_*_from_raw`（public）；新增 `set_ml_*_from_schema` + batch 入口；`from_raw` 降為內部反序列化細節 |

與 [[0005]]（service 經 port、aggregate 不貧血）、[[0008]]（LiveModel service-owned、UI 不自己 lower；本 ADR partial-supersede 其 commit-register 歸屬）一致。
