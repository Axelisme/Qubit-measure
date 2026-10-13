# gui/form_adapter — 從 core cfg 生成 Form

**Last updated:** 2026-10-13，從 measure adapter 框架搬入

本 package 是 GUI 的 Qt-free 前端框架。它把 core cfg 宣告與作者寫的起始樹翻譯成 `gui.form` 的 `FormModel`，再把表單結果轉回驗證過的 core cfg。它供 measure、workflows、library 與 writeback 的表單定義共用；目前的 caller 是 measure 的註冊、protocol 與分頁 Form resource。

公開 root 只有兩個名字：

- `Form(core_cfg, tree, *, extras=None, lower=None)`：生成 FormModel 子類別、defaults hook、choices 與 `to_cfg`。Core annotations 決定形狀與驗證；樹決定起始值、鎖定、隱藏欄位與 module 預設。Expr 初值欄位在生成的 model 中不帶靜態預設，算不出時保持未填。
- `ModuleValues`：`Form.read_module` 回傳的單一 module 欄位實際值、模式與來源。

Template／Slot 節點在 host 呼叫 `Form.bind_templates(catalog, adapter=...)` 後才解析。Host 的註冊入口負責呼叫它，例如 measure `Registry.register(form=...)`。範本目錄的格式在 `cfg_model`，真實範本表在 `zcu_lab`。

Module、seed 與 annotation 翻譯放在私有模組 `_modules.py`、`_seeds.py`、`_types.py`，只由 `form.py` 組合。

## 依賴方向

本 package 依賴 `cfg_model`、`program`、`resources`、`gui.form` 與 `gui.session.parameters`。它不 import `gui.app`、widgets、remote、experiment、mcp、Qt 或 `zcu_lab`；`.importlinter` 的 `form-adapter-no-host` contract 與 C16 執行這條邊界。

`gui.form` 是通用 Pydantic 輸入核心，不認得 core cfg 推導。推導與 module materialization 放在本 package，不放回 `gui.form`。

設計背景見 [measure 表單推導 ADR 草稿](../../../../docs/adr/draft/measure-form-derivation.md) 與 [`gui/form` README](../form/README.md)。
