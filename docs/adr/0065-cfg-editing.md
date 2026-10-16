---
status: accepted
---

# ADR-0065 — Cfg 編輯模型與使用邊界

## 問題與決策

Qt、remote 與 Run 必須使用同一份 tab cfg。若 widget 持有唯一可編輯的樹，headless caller 就依賴 view 的存在。若 Run 重新解析 live source，caller 看到的值與實際執行值可能不同。

Measure 的 `TabFormResource` 擁有一份 Qt-free `FormState` 與固定來源。`TabFormResources` 管理建立、查找、操作禁令與撤銷。Qt、remote 與 Run 都使用同一資源。Frontend 只讀單層 description、input、preview 與 errors，寫入走 resource 的公開入口。Form 決定 defaults、形狀、驗證與 core cfg 轉換，wire caller 不重建這些規則。

Library、Inspect 與 writeback 也使用 FormState／FormEditor，但各自持有 state 與提交政策，不借用 tab resource。Library 寫入持久 Spec，writeback 的 opaque draft 與 selected Apply 見 [[0008]]。舊 CfgSchema、gui.cfg、widgets.cfg 與 experiment.cfg_editing 已移除。

## 共用機制與領域 owner

共用 `gui.form` 擁有 input、單層 description、FormState、preview、result 與錯誤。`widgets/form` 的 FormEditor 依 description 顯示欄位與選項，不解讀 library 或實驗語義。`gui.form_adapter` 從 core cfg 與持久 Spec 生成 Form，集中 FieldMeta 呈現投影、key-or-inline wrappers 與屏蔽欄位政策。Form core 不讀 FieldMeta。

Library entry 表單由 session 擁有，固定 root 的具體 Spec family。初始 input 與提交都保留 child key，不能把既有 pulse 切成 readout；換 family 要新建 entry。Optional None 維持缺席。生成器略過選填的 desc，提交時以 Spec 預設補回，不保留原值；沒有預設的必填欄位仍可編輯。

None、library keys 與各 Inline kind 在同一個合併下拉。Key 只顯示名稱，不展開唯讀 descendants。普通選 Inline 使用 fresh defaults，不繼承舊分支；明確的 to_inline 指令才從固定 library 複製內容。Advanced 群組預設折疊。Expression 只適用於文字欄位，下拉與勾選欄位不求值。

實驗 adapter 宣告 `Form(core_cfg, tree)`。Core annotations 定義可編輯形狀，tree 定義起始值、Slot、Template 與 lock。Open 與 Reset 計算 defaults。Memento 保存 Form input，不保存來源覆寫。Make_form、範本與固定來源的契約見 [[0073]]。Reset 可在同一資源內更換形狀與初值；成功後舊 editor state 不再承接寫入。

## Input 與宿主提交

FormInput 保存未填、null 與 expression 原文，FormModel 實例只承載求值後的普通值。FormState 擁有 draft／base，不持有 Engine revision。FormEditor 擁有尚未 parse 的文字與 local parse errors；宿主提交前檢查這些錯誤，再取得 Form result，不能把 preview 當成可提交結果。

Host 決定 capture、Apply、Discard 與 input 保存時機。Library／writeback 提交持久 Spec；workflow host 交普通 plan／tunables 給 Engine，只有 Engine 成功 publication 後才接受 input／base，見 [[0062]]。Expression 留在 FormInput，不進 Engine journal。Host 保留 input 的責任不表示它擁有另一套 evaluator、validator 或 codec。

共用核心不依賴 app、widgets、remote、experiment 或 Qt，editor 不依賴 app、remote、experiment 或 MCP。既有 `.importlinter` contracts `form-core-no-ui-policy` 與 `form-editor-no-host` 投影這兩個方向。使用者於 2026-10-16 以「都採用建議」確認將表單草稿的這些穩定邊界併入本篇及 [[0062]]，方法與欄位細節留在 owner 文件。

## Measure tab 的 input 與使用

Input 保存 ordinary JSON、顯式 `{expr: text}`、complex tags 與 model field names。普通字串保持 literal。Preview 提供 ordinary leaf values，不代表可以 Run。Form errors 包含帶路徑的診斷，合法但尚未完成的 input 保持可編輯。結構或來源錯誤拒絕整批，不保留成功前綴。

`tab.get_cfg`、`tab.edit_cfg` 與 `tab.reset_cfg` 回傳 input、values、errors，不附 schema。Values 是 path/value 葉清單，errors 是 path/message 清單。Tab 以字串陣列指定 model-field leaf paths，range 分別寫 start、stop、n。Wire 不提供 CfgSchema kind、ref 子樹或 Valid/Invalid/Unavailable publication。

`tab.describe_cfg(path)` 以 model-field 字串陣列查目標與直接 children，不產生整棵 JSON Schema。回覆同時包含局部 input_present、input、value 與 Form preview 定向 errors，不 materialize 或呼叫 to_cfg。Lowered cfg errors 用 get_cfg 或 Run admission 檢查。Unknown、inactive 與 whole-leaf 子路徑回 invalid_params／form_path，data.form_path 保留請求路徑。其他 ValueError 維持 controller_error。Writeback 的 observe_entry 同樣只回 input、values、errors。

使用者 2026-10-15：「好，先照這個規劃開始」。共用描述直接讀宣告與目前 input，options 只列 label、value 與 command name／label，不帶 fields。查 active path 才列 children；單一普通 model wrapper 攤平一層，完整 path 仍保留 wrapper。can_expand 只表示結構可展開，不執行 child hooks 或保證 children 非空。selection_fixed 表示本層 annotation 只有一個非 null 分支，不凍結內部欄位；fixed 表示完整值固定。Recipe lock 取兩者的 or。

FormEditor 首次展開才查詢並建立 descendants，收起保留 widgets 與未提交文字，不偷寫 input。換選項只替換該子樹。折疊錯誤從完整 preview 聚合到可見祖先。Measure、workflows、Inspect、New Entry 與 writeback 共用此機制，保留各宿主的固定 capture、Apply／Cancel、guard 與 lifetime。

Read、preview、result 與 accept 不捕捉來源。通知期間允許讀取，禁止 mutation、accept、Run 及 close。Subscriber 故障記錄 traceback，不回滾已提交 input，也不阻止其他 subscriber 收到通知。

Run 只接受目前有效的 Form result，不收 cfg ref 或 expected revision，也不回退上次有效 cfg。Accepted config 深層隔離執行值與來源 provenance。後續 Edit、全域來源變更與 Reset 都不改已接受 Run。控制身份見 [[0070]]，hardware guard、lease 與 operation lifecycle 見 [[0066]]。

Tab 建立時就能 headless 編輯。Qt attach/detach 只管理 view 與訂閱。Idle close 撤銷舊 handle，重建同名 tab 建立新資源。Active Run 禁止同 tab 的人工 Edit、Reset、replacement 與 close，其他 tab 仍可編輯。Load 只載既有結果及分析來源，保留表單原輸入與固定來源。有效、Invalid 與未完成的 draft 都不被改寫，Load 不要求 current 有效，也不從 native cfg 推回 extras。Wire 回覆保留結果與分析準備 facts，不提供回填狀態。原始 cfg 保存在 native 資料檔，透過公開 datafile.load_run_data 離線讀取為 StoredRun，GUI 不提供檢視入口，見 [[0063]]。

## 固定來源與 Form library key

Open 與 Reset 捕捉一次 point、library、alias 表、cached device infos 與已發布 value sources。Edit、parameter_read、preview、result 與 Run 使用這份 capture，直到下一次 Reset。全域來源事件不刷新既有 tab。捕捉不輪詢硬體。

公開 `read_context(aliases=None)` 固定 point、library、storage 與全表。非空 aliases 只覆寫這次 capture 的具名列，保留 dependent authored links，不改全域表。省略、None 與 {} 使用全域列。EMPTY 且無覆寫時回 None，保留 context-free defaults，非空覆寫仍拒絕 no_context。

`tab.reset_cfg(aliases=...)` 是 tab 的單次覆寫入口，重新捕捉來源並恢復 defaults。覆寫不存入 alias 表或 memento。`tab.parameter_read` 只讀 resource.env.parameters，不收 aliases。`tab.run_start` 在任何 focus 或 operation 效果前拒絕非空 aliases，None 與 {} 使用畫面上的值與固定 capture。

RecipeSession.open_tab 的公開 aliases/reuse 用法不變。新 tab 與 reuse tab 都先 Reset，再設定參數與 Run。Target 對到 Reset 的 aliases，而非 preview 或 Run 的臨時解析。Recipe 的參數 schema 保持手寫，不從 Form 或 Python signatures 生成。

Form Slot 使用公開 ModuleRef/ModuleInline input。Key branch 保存 `{kind: ref, name: actual_library_key}`，inline branch 保存 `{kind: inline, module: Spec input}`。Nested key-or-inline 欄位用普通字串作 key。RecipeTab.use_library 寫實際 key，不建立舊 __ref editing intent。Locked Slot 只有 inline branch，use_library 在送 RPC 前拒絕選 key。

RecipeTab.read_field 每次以一次 `tab.describe_cfg` 讀同一 owner turn 的目前欄位，回傳具名 RecipeFormField，包含 authored input、input_present、ordinary value、key/inline mode、actual key、Form lock、目前 child field names 與 directed errors。它不投影 CfgSchema kind，也不承諾 ref descendants 唯讀。Inline model 的路徑包含 module，例如 `modules.readout.module.freq`。Range count 的路徑是 n，set_sweep 的公開 expts 參數對到這個葉欄位。

實驗 cfg 依 [[0073]] 使用具名裝置欄位，例如 flux_dev、jpa_flux_dev，不再保存 dev role mapping。Form 的 AliasName 在 Open／Reset 解析 direct 名稱。使用 flux 的 cfg 必填 flux_unit，值為 A、V 或 native。Core 與 Notebook 不提供隱含 default，GUI 的初始值為 A。Recipe helper 只比 cached unit，成功後寫 caller 指定的具名裝置欄位與獨立 flux_unit，不換算、不連線或 setup。

Core 以 context.devices[cfg.flux_dev] 等具名欄位取 driver。缺名由實際使用者報錯；RunContext 不提供 setters。操作分工見 [[0067]]，Run 前 snapshot 與保存見 [[0063]]。Measure 不保留舊跨 tab source refresh／Unavailable 恢復路徑；使用時綁範本與缺來源的語義由 [[0073]] 擁有。

Form expression 的 `$alias` 與 `$alias.path` 沿固定 ParameterContext 解析，bare concrete paths 使用獨立 namespace。裸 `$alias` 是名稱字串，不參與數值算術或字串插值。缺列或未綁時 errors 指名 alias，不沿用舊解析值。Expression 不刷新來源。

## Library 與 Inspect

`open_library_entry` 從持久 ModuleSpec／WaveformSpec 建立 LibraryEntryForm。Expression 使用開啟時固定的 env，Apply 將求得的數字寫進 Spec，不把 expression 存成 Library 來源。Name、Saved／Unsaved、Revert 與切換條目前的未存確認由 Inspect 管理。表單 Invalid 或有 local parse errors 時不提交。

New 從 Registry 注入的 TemplateDef 目錄建立 entry input。換範本建新 State 與 Editor，欄位、折疊與捲動回預設，只保留 Name。已綁定的 Slot 保留 child key。生成器負責 defaults 與驗證，caller 不另建 normalization 或 shape catalog。

`context.ml_edit` 使用 Form 路徑陣列與值，一次套用該 entry 的整批 edits。只有有效 Spec 才交既有 Library write owner；錯誤帶欄位路徑，Library 不留下成功前綴。Module writeback 使用相同的 entry form 與 observation，但 draft-local editing 與 selected Apply 的交易邊界依 [[0008]]，不宣稱跨項原子性。`ml_list_roles`／`ml_create_from_role` 保留既有名稱與範本入口。

## 共用 Inspect 與來源

Inspect、Device 與 ARB UI 及其 service 由 session 提供，host 經公開入口注入必填協作者，不靠 host 差異猜能力。Inspect 有 Parameters、Modules、Aliases 三頁。Parameters 使用自己的 YamlValue／FieldSpec，不屬 Form 遷移。Run 或 agent 接管時，Inspect 表單不可改，但仍可選取文字、展開與捲動。

Aliases 由共用 Inspect 的第三頁編輯，沒有獨立 AliasDialog 或 aliases wire 名稱。Apply 將有序 authored rows 整表交 `apply_alias_rows`；有 opened entry 時先原子保存，再安裝 memory 與發布，無 entry 時只改 memory。失敗保留舊表、檔案、版本與 draft。未存修改走 Save／Discard／Cancel，沒有 Revert。

MCP 直接呼叫 owning context，不開 dialog、不切頁或焦點。`context.aliases_set` 回已安裝的 authored rows，不解析；解析另讀 snapshot。`context.set_parameter` 用一次 capture 固定具體 target 與 before，保留 metadata、不寫 ledger。這些 mutation 沿既有 owner 與 holder admission，Run 中可寫，但不改既有 Form 或 Run 的固定 capture。`dialog.screenshot` 只擷取目前已開 dialog 的當前頁，不切 Inspect tab。

## 取捨與相鄰責任

單一表單核心讓各宿主共用 input、驗證與 editor，宿主仍決定 lifetime、來源捕捉與提交時機。來源變更不提示、不自動更新；Measure 要最新值就明確 Reset。Source basis 記錄 provenance，不是 cfg admission 門檻。舊 `@{}` 補全、eval／value_ref wire tags 與 editor.* 事件／RPC 不再提供。`value.list`／`value.read` 與內部 ValueRef／resolve_value_ref 保留，見 [[0037]]。

參數容器與 Library 寫入權由 [[0067]] 定義；保存與 crash durability 歸 [[0063]]；workflow run 與 patches 歸 [[0062]]；wire、guard 與 delivery failure 見 [[0068]]。模組入口見 [Form](../../lib/zcu_tools/gui/form/README.md)、[生成器](../../lib/zcu_tools/gui/form_adapter/README.md)、[session](../../lib/zcu_tools/gui/session/README.md) 與 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)。
