# gui/form — Qt-free Pydantic input editing

**Last updated:** 2026-10-08，F1 leaf 與 choices 契約

FormModel 的實例只承載普通值。類別上的 defaults、choices 與 input_schema 提供輸入契約。FormInput 保存未填、null、expression 與 complex tags；FormState 擁有 draft/base、batch write、branch switch、結果、差異與 compatible restore，不擁有 Engine revision 或提交 policy。

Scalar、整個 sequence 與整個 typed mapping 都是 leaf。普通 nested model 只編輯子欄位，nullable 區段支援 object／null 啟停。Choices 使用 model-field leaf paths，驗證與 schema 共用同一份當次選項，不影響重用同型 nested model 的其他欄位。

FormEnv 借用 host 捕捉的 md、library 與 native device infos。工具不寫來源、不 refresh 或 poll。ModuleRef/ModuleInline 使用公開 library Spec 讀取；to_cfg 的 materialization 與 range 展開歸具體表單。

Display 只承載欄位呈現資訊。順序依欄位宣告，分組依 nested model；collapsed 只用於子 model 欄位。Editor、host 與 MCP 共用 core 行為，但不反向成為 core dependency。

跨模組契約見 docs/adr/draft/pydantic-form-tool.md。這是新表單工具，不改 CfgSchema，不接 Measure。
