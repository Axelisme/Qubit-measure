# gui/form_adapter — 從 core cfg 生成 Form

**Last updated:** 2026-10-13，持久 Spec 表單與 module 欄位屏蔽政策

本 package 是 GUI 的 Qt-free 前端框架。它把 core cfg 宣告與作者寫的起始樹翻譯成 `gui.form` 的 `FormModel`，再把表單結果轉回驗證過的 core cfg。它供 measure、workflows、library 與 writeback 的表單定義共用；目前的 caller 是 measure 的註冊、protocol 與分頁 Form resource。

公開 root 有三個名字：

- `Form(core_cfg, tree, *, extras=None, lower=None)`：生成 FormModel 子類別、defaults hook、choices 與 `to_cfg`。Form 的 cfg 結果型別具協變性，具體 `Form[SampleCfg]` 可交給 generic-free host，`to_cfg` 仍回傳 SampleCfg 或 FormErrors。Core annotations 決定形狀與驗證；樹決定起始值、鎖定、隱藏欄位與 module 預設。Expr 初值欄位在生成的 model 中不帶靜態預設，算不出時保持未填。Range 的 Expr start／stop 也沿用這個規則，不回落 LinRange 的 0／1。
- `module_form(spec_type)`：為一個具體持久 module 或 waveform Spec class 建 Form。生成的 model 可當 FormState root，也可當 `ModuleInline` 的 module 欄位；host 的選項沿用自己的 `choices(env)`。`to_cfg` 回傳同一個 Spec，child library key 保持字串，不讀 library、asset，也不轉 native cfg。`Form.input_from_spec(spec, env)` 把既有 Spec（library entry 或 writeback proposal）投影成 input；它只接受這類 Form 的同型 Spec。
- `ModuleValues`：`Form.read_module` 回傳的單一 module 欄位實際值、模式與來源。

生成器擁有兩條翻譯政策，host 與 `gui.form` 不重做：

- `FieldMeta` 的 label、unit、decimals、description、group 轉成 `Display`，core 欄位與 module 欄位共用同一個翻譯。被鎖定的欄位是單值 Literal，不帶 decimals。Sweep 的 decimals 只傳到生成 LinRange 的 start／stop float 欄位，不套外層 model、n 或 ExplicitList 整列。
- 生成 module 欄位時，選填的 `desc`、`mask`、`outsel`、`ro_ch` 與 `mux_*` 不在 model、input、schema 與寫入路徑。`to_cfg` 驗證成 Spec 時由 Spec 預設補回。從 library 複製成 inline 或 `input_from_spec` 時，這些欄位的原值被丟棄，所以編輯後會變回預設。必填且沒有預設的欄位，例如 `DirectReadoutCfg.ro_ch`，照常可編輯。範本 seed 這些欄位時，註冊入口拒絕，訊息寫出 template id 與 seed 路徑；`to_cfg` 只由 Spec 預設補回。其他欄位的字串 seed 照舊是字面值。目前沒有露出覆寫機制。

Template／Slot 節點在 host 呼叫 `Form.bind_templates(catalog, adapter=...)` 後才解析。Host 的註冊入口負責呼叫它，例如 measure `Registry.register(form=...)`。範本目錄的格式在 `cfg_model`，真實範本表在 `zcu_lab`。

Spec Form 的 `template_input(Template(id), catalog, env)` 直接建立範本的 fresh input，不需先 bind。它沿用 Expr 初值、literal lock 與 Slot fallback 規則。已綁定 Slot 保留 child library key，`to_cfg` 回持久 Spec，不展開 native cfg。Core cfg Form 拒絕這個入口。

回傳的 `input.model` 可以是 `Form.model` 的專用 subclass，讓不同範本的 required 欄位與 lock 互不影響。用回傳 input 建新 `FormState`。同一範本重算初值用 `input.model.default_input(env)`，換範本則建新 state。原 Form.model、範本目錄與 captured sources 不變。

Module、seed、Display 與 annotation 翻譯放在私有模組 `_modules.py`、`_seeds.py`、`_display.py`、`_types.py`，只由 `form.py` 組合。

## 依賴方向

本 package 依賴 `cfg_model`、`program`、`resources`、`gui.form` 與 `gui.session.parameters`。它不 import `gui.app`、widgets、remote、experiment、mcp、Qt 或 `zcu_lab`；`.importlinter` 的 `form-adapter-no-host` contract 與 C16 執行這條邊界。

`gui.form` 是通用 Pydantic 輸入核心，不認得 core cfg 推導。推導與 module materialization 放在本 package，不放回 `gui.form`。

設計背景見 [measure 表單推導 ADR 草稿](../../../../docs/adr/draft/measure-form-derivation.md) 與 [`gui/form` README](../form/README.md)。
