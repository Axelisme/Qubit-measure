# gui/form — Qt-free Pydantic input editing

**Last updated:** 2026-10-13，巢狀 tagged union 展開成具體 model branch

FormModel 的實例只承載普通值。類別上的 defaults、choices 與 input_schema 提供輸入契約。FormInput 保存未填、null、expression 與 complex tags；FormState 擁有 draft/base、batch write、branch switch、結果、差異與 compatible restore，不擁有 Engine revision 或提交 policy。

Numeric、complex 與所有字串 value leaves 只對顯式 `{"expr": str}` 求值，包括 Literal[str] 與 str Enum。普通字串，包括 `$` 開頭的內容，仍是 literal。Core 使用共用 evaluate_scalar_expr，再遵守 choices 與 Pydantic 約束；missing／unbound alias 指名失敗，不以預設值代替。Input schema 與 compatible restore 保留 expression 原文，GUI 與 MCP 共用這份輸入契約。

Input schema 的 tagged oneOf branch 以 x-form-discriminator-values 提供保留 JSON 型別的 intrinsic tags。這份 identity 不隨 dynamic choices 改變。Model 欄位可達的每個 branch 都是一個具體 model：union member 若是同一 tag key 的具名 union（例如 `ModuleSpec` 內的 `ReadoutSpec`），schema 把它展開成各 member 的 branch，與 core 對 annotation 的 union 展開一致。Choices 仍限制當次可接受的 leaf 值，不切換目前的 input branch。Discriminator 是決定子 model 形狀與 branch defaults 的結構 selector，schema 只提供普通 typed tag，不提供 expression envelope。

Scalar、整個 sequence 與整個 typed mapping 都是 leaf。普通 nested model 只編輯子欄位，nullable 區段支援 object／null 啟停。Choices 使用 model-field leaf paths，驗證與 schema 共用同一份當次選項。Parent 與 child 宣告同一 leaf 時，最深 nested FormModel 的 local choices 優先，包含空選項。非重疊的 parent 選項保持有效，重用同型 nested model 的欄位各自隔離。

Compatible restore 以原文欄位集合與型別選擇未標 tag 的 model union。恰好一個 member 相容才保留 subtree，合法 expression envelope 依欄位能力參與比對，不求值。零個或多個 member 相容就丟棄，不猜第一個 branch。Tagged branch 的 selector 與 defaults 契約不變。還原後 captured source 檢查失敗的 leaf（例如 library 沒有的 ref 名稱，或沒有 captured library 時的任何 ref）保留目前 input，目前沒有就留空，路徑和其他 dropped paths 一起回傳。不是由還原 leaf 造成的 source 失敗仍然 raise；一般 write、default_input 與 result 的 source 語意不變。

FormEnv 借用 host 捕捉的 ParameterContext、library 與 native device infos。沒有 parameter capture 時仍可編輯 ordinary input 與純數學 expression；參數引用失敗會落在消費欄位的 error path。工具不寫來源、不 refresh 或 poll。ModuleRef/ModuleInline 使用公開 library Spec 讀取；to_cfg 的 materialization 與 range 展開歸具體表單。

FormState.preview 提供隔離的 ordinary leaf values 與完整 errors，供 view 在 model-invalid draft 中顯示其他欄位的新求值結果。失敗的 expression 或含失敗子值的 whole container 不提供值。成功時包含 Pydantic defaults 與 coercions，不把它們寫回 input。Preview values 不代表可提交；host 仍檢查 errors 與 view 的 local parse errors。

Display 只承載欄位呈現資訊。順序依欄位宣告，分組依 nested model；collapsed 只用於子 model 欄位。Editor、host 與 MCP 共用 core 行為，但不反向成為 core dependency。

跨模組契約見 docs/adr/draft/pydantic-form-tool.md。這是新表單工具，不改 CfgSchema，不接 Measure。
