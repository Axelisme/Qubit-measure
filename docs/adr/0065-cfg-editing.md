---
status: accepted
---

# ADR-0065 — Cfg 編輯模型與使用邊界

## 局部取代範圍

本篇關於普通切換補 fresh defaults、數值葉 command 改 parent，以及僅保存 input 並丟棄不相容葉的規則，由 [ADR-0079](0079-form-field-transitions.md#決策狀態與問題) 局部取代。欄位轉換與保存還原以該篇為準，其餘 owner 邊界不變。

## 問題與決策

Qt、remote 與 Run 必須使用同一份 tab cfg。若 widget 持有唯一可編輯的樹，headless caller 就依賴 view 的存在。若 Run 重新解析 live source，caller 看到的值與實際執行值可能不同。

Measure 的 `TabFormResource` 擁有一份 Qt-free `FormState` 與固定來源。`TabFormResources` 管理建立、查找、操作禁令與撤銷。Qt、remote 與 Run 都使用同一資源。Frontend 只讀單層 description、input、preview 與 errors，寫入走 resource 的公開入口。Form 決定 defaults、形狀、驗證與 core cfg 轉換，wire caller 不重建這些規則。

Library、Inspect 與 writeback 也使用 FormState／FormEditor，但各自持有 state 與提交政策，不借用 tab resource。Library 寫入持久 Spec，writeback 的 opaque draft 與 selected Apply 見 [[0008]]。舊 CfgSchema、gui.cfg、widgets.cfg 與 experiment.cfg_editing 已移除。

## 共用機制與領域 owner

2026-10-18 採用的欄位轉換、固定 list 與保存還原設計見 [[0079]]，實作尚待完成。以下相關條款描述目標契約，不代表目前程式已完成遷移。

共用 `gui.form` 擁有 input、單層 description、FormState、preview、result 與錯誤。`widgets/form` 的 FormEditor 依 description 顯示欄位與選項，不解讀 library 或實驗語義。`gui.form_adapter` 從 core cfg 與持久 Spec 生成 Form，集中 FieldMeta 呈現投影、key-or-inline wrappers 與屏蔽欄位政策。Form core 不讀 FieldMeta。

Library entry 表單由 session 擁有，固定 root 的具體 Spec family。初始 input 與提交都保留 child key，不能把既有 pulse 切成 readout；換 family 要新建 entry。Optional None 維持缺席。生成器略過選填的 desc，提交時以 Spec 預設補回，不保留原值；沒有預設的必填欄位仍可編輯。

None、library keys 與各 Inline kind 在同一個合併下拉。Key 只顯示名稱，不展開唯讀 descendants。普通選 Inline 由欄位 converter 依明確的型別配對政策繼承；明確的 to_inline 指令從固定 library 複製內容。無 converter 時只改 selection，子表單結構與 input 不變，不補 fresh defaults，見 [[0079]]。Advanced 群組預設折疊。Expression 只適用於文字欄位，下拉與勾選欄位不求值。

實驗 adapter 宣告 `Form(core_cfg, tree)`。Core annotations 定義可編輯形狀，tree 定義起始值、唯讀屬性、Slot 與 Template。Open 與 Reset 計算 defaults。Memento 保存 selection／branch 與原始 Form input，不保存來源覆寫；還原不執行 converter，identity 不相容時明確報錯，見 [[0079]]。Make_form、範本與固定來源的契約見 [[0073]]。Reset 可在同一資源內更換形狀與初值；成功後舊 editor state 不再承接寫入。

## Input 與宿主提交

FormInput 保存未填、null 與 expression 原文，FormModel 實例只承載求值後的普通值。FormState 擁有 draft／base，不持有 Engine revision。FormEditor 擁有尚未 parse 的文字與 local parse errors；宿主提交前檢查這些錯誤，再取得 Form result，不能把 preview 當成可提交結果。

Host 決定 capture、Apply、Discard 與 input 保存時機。Library／writeback 提交持久 Spec；workflow host 交普通 plan／tunables 給 Engine，只有 Engine 成功 publication 後才接受 input／base，見 [[0062]]。Expression 留在 FormInput，不進 Engine journal。Host 保留 input 的責任不表示它擁有另一套 evaluator、validator 或 codec。

宣告唯讀的欄位使用初值與 `Display(readonly=True)`，由 Form core 對 Qt、MCP 與 RPC 執行同一寫入規則。同值寫入通過，不同值拒絕；父層或批次寫入不能繞過唯讀子欄位。RPC／MCP 拒絕保留 `error.reason="form_input_rejected"` 與 `error.data.form_path`。唯讀拒絕另帶 `error.data.reason="readonly"` 與目前值的 JSON 形式 `error.data.current`；其他 rejection 不帶這兩個欄位。

使用者於 2026-10-19 裁定：「那為什麼不直接都用ReadOnly就好？」「1同樣值不報錯。2我選存檔的版本，因為有些lock值可能是生成的。能不能執行是另外一回事」。還原保留唯讀欄位的存檔值，不從宣告重新計算，執行有效性仍由一般驗證判定，見 [[0079]]。

共用核心不依賴 app、widgets、remote、experiment 或 Qt，editor 不依賴 app、remote、experiment 或 MCP。既有 `.importlinter` contracts `form-core-no-ui-policy` 與 `form-editor-no-host` 投影這兩個方向。使用者於 2026-10-16 以「都採用建議」確認將表單草稿的這些穩定邊界併入本篇及 [[0062]]，方法與欄位細節留在 owner 文件。

## Measure tab 的 input 與使用

Input 保存 ordinary JSON、顯式 `{expr: text}`、complex tags 與 model field names。普通字串保持 literal。Preview 提供 ordinary leaf values，不代表可以 Run。Form errors 包含帶路徑的診斷，合法但尚未完成的 input 保持可編輯。結構或來源錯誤拒絕整批，不保留成功前綴。

`tab.get_cfg`、`tab.edit_cfg` 與 `tab.reset_cfg` 回傳 input、values、errors，不附 schema。Values 是 path/value 葉清單，errors 是 path/message 清單。Tab 以 typed segments 指定 model-field paths，欄位名為 str，list index 為 int。Sweep 的 tagged branch 決定 active 座標與採樣。Range 使用 start／stop，Center 使用 center／span，Expts 使用 n，Step 使用 step。四個分支只保存 active 欄位，切換其中一個維度不重設另一個維度。Inactive paths 沿 form_path 契約拒絕。Wire 不提供 CfgSchema kind、ref 子樹或 Valid/Invalid/Unavailable publication。

Sweep 的模式與 active 原稿存於 FormInput，不由 widget 保存。Sweep row 宣告 command，callable 透過公開 evaluate 求值參與欄位，回該 row 的候選狀態。FormState 檢查 annotation 與 subtree 範圍，再原子安裝；command 不修改 parent 或 siblings。座標切換不使用 n／step，保留其原文、錯誤或缺席。採樣切換使用座標與目前採樣欄位。成功只把參與 expression 改成 direct 數字，之後新輸入的 expression 照常保存。無關欄位錯誤不阻擋換算，失敗保留模式與整組原稿。普通選項切換與 sweep command 共用 [[0079]] 的安裝流程，不另外補 defaults。

Step 的有效點數由共用 Form 驗證層計算。非零 span 的 step 必須非零、同向且不大於整段，點數為 floor(|span|/|step|)+1。商接近整數時以相對容差 1e-9 修正，不對一般不整除的商四捨五入。有效 stop=start+step×(n−1)，保留 start 與 step。Center 以有效端點投影 center／span。零 span 只接受 step=0 的單點。非法或未完成的 draft 保存原稿與診斷，不能縮成另一個可執行掃描。

Preview、result 與 description 都不暗改 input。Values 顯示有效端點，Qt 在 authored 與有效數值不同時明示 Effective 值。例如 stop=1、step=0.3 保留原稿 1，顯示有效 stop=0.9。有效子表單的投影不因無關欄位出錯而消失。Form adapter 使用同一投影與原 step 產生既有 SweepCfg，執行端仍驗證方向、單點與點數一致性。Memento 保存四種 active input 與 expression，Reset 回 fresh Range + Expts。

`tab.run_cfg_command(tab_id, path, name)` 只派發目前 active path 描述的公開指令，沿 resource 的控制身份、busy、lifetime 與 mutation guard。成功回 input／values／errors。掛載欄位的 description 提供公開 command，Qt 檢查相關 local parse errors，不猜模型結構或自行換算。Sweep command 由 sweep row 的右鍵選單呈現，不掛在數值葉；標準文字與 Expression 選單保留。禁止編輯時指令也不可執行。

`tab.describe_cfg(path)` 以 str 欄位名／int index 查目標與直接 children，不產生整棵 JSON Schema。回覆同時包含局部 input_present、input、value 與 Form preview 定向 errors，不 materialize 或呼叫 to_cfg。Lowered cfg errors 用 get_cfg 或 Run admission 檢查。兩個讀取入口另回全表 diagnostics，涵蓋全部原始 list 位置並合併 preview／lowered errors，不受 describe path 或 picker 影響。Form adapter 保留逐位置 seed 來源，校準狀態只依作者的 `with_seed_metadata` 宣告，未知保持 null，不從缺值推論。Unknown、inactive 與 whole-leaf 子路徑回 invalid_params／form_path，data.form_path 保留請求路徑。其他 ValueError 維持 controller_error。Writeback 的 observe_entry 同樣只回 input、values、errors。

使用者 2026-10-15：「好，先照這個規劃開始」。共用描述直接讀宣告與目前 input，options 只列 label、value 與 command name／label，不帶 fields。查 active path 才列 children；單一普通 model wrapper 攤平一層，完整 path 仍保留 wrapper。can_expand 只表示結構可展開，不執行 child hooks 或保證 children 非空。selection_fixed 表示本層 annotation 只有一個非 null 分支，不凍結內部欄位；fixed 表示完整值固定。Core cfg 本身的單值 Literal 仍算固定值。`RecipeFormField.locked` 是 selection_fixed、fixed 與宣告 readonly 三者的 or，不從當次選單數量推定。

FormEditor 首次展開才查詢並建立 descendants，收起保留 widgets 與未提交文字，不偷寫 input。換選項只替換該子樹。折疊錯誤從完整 preview 聚合到可見祖先。Measure、workflows、Inspect、New Entry 與 writeback 共用此機制，保留各宿主的固定 capture、Apply／Cancel、guard 與 lifetime。

Read、preview、result 與 accept 不捕捉來源。通知期間允許讀取，禁止 mutation、accept、Run 及 close。Subscriber 故障記錄 traceback，不回滾已提交 input，也不阻止其他 subscriber 收到通知。

Run 只接受目前有效的 Form result，不收 cfg ref 或 expected revision，也不回退上次有效 cfg。Accepted config 深層隔離執行值與來源 provenance。後續 Edit、全域來源變更與 Reset 都不改已接受 Run。控制身份見 [[0070]]，hardware guard、lease 與 operation lifecycle 見 [[0066]]。

Tab 在來源就緒後建立，建立時 capture 一次，隨即能 headless 編輯。腳本可用 `tabs.new_tab(adapter, input=TabInput(form_json))` 交回 opaque authored input，由 Form owner 解碼及還原，不由腳本重建 form 或來源。`TabInput` 保存 data、expression 原稿、一般 invalid draft 與 explicit selections，不含結果或來源，也不是 framework session 磁碟 schema。不相容路徑由既有 restore 政策丟棄並記 log，未知 adapter、無效 JSON 或來源建立失敗在 tab 發布前報錯。框架沒有 pending 還原；腳本選擇何時建 tab，見 [[0064]]。

Qt attach/detach 只管理 view 與訂閱。Idle close 撤銷舊 handle，重建同名 tab 建立新資源。Active Run 禁止同 tab 的人工 Edit、Reset、replacement 與 close，其他 tab 仍可編輯。Load 只載既有結果及分析來源，保留表單原輸入與固定來源。有效、Invalid 與未完成的 draft 都不被改寫，Load 不要求 current 有效，也不從 native cfg 推回 Form input。Wire 回覆保留結果與分析準備 facts，不提供回填狀態。原始 cfg 保存在 native 資料檔，透過公開 datafile.load 離線讀取為 Container；Results browser 可唯讀檢視保存的 cfg，不回填 Form，見 [[0063]]。

## 固定來源與 Form library key

Open、Refresh 與 Reset 各捕捉一次 point、library、alias 表、cached device infos 與已發布 value sources。Edit、parameter_read、preview、result 與 Run 使用這份 capture，直到下一次 Refresh 或 Reset。全域來源事件不刷新既有 tab。捕捉不輪詢硬體。

公開 `read_context(aliases=None)` 固定 point、library、storage 與全表。非空 aliases 只覆寫這次 capture 的具名列，保留 dependent authored links，不改全域表。省略、None 與 {} 使用全域列。EMPTY 且無覆寫時回 None，保留 context-free defaults，非空覆寫仍拒絕 no_context。

`tab.refresh_sources(aliases=...)` 重新捕捉來源，保留全部 input data 與 explicit selections，包括 expression 原稿與先前初值求得的 direct 值，再以新來源驗證。缺失來源仍是欄位錯誤。`tab.reset_cfg(aliases=...)` 重新捕捉來源並以新來源重算 defaults。兩者的單次覆寫都不存入 alias 表或 memento。`tab.parameter_read` 只讀 resource.env.parameters，不收 aliases。`tab.run_start` 在任何 focus 或 operation 效果前拒絕非空 aliases，None 與 {} 使用畫面上的值與固定 capture。

RecipeSession.open_tab 的公開 aliases/reuse 用法不變。新 tab 與 reuse tab 都先 Reset，再設定參數與 Run。Target 對到 Reset 的 aliases，而非 preview 或 Run 的臨時解析。Recipe 的參數 schema 保持手寫，不從 Form 或 Python signatures 生成。

Form Slot 使用公開 ModuleRef/ModuleInline input。Key branch 保存 `{kind: ref, name: actual_library_key}`，inline branch 保存 `{kind: inline, module: Spec input}`。Nested key-or-inline 欄位用普通字串作 key。RecipeTab.use_library 寫實際 key，不建立舊 __ref editing intent。Module 子欄位唯讀不限制祖先選擇，仍可選 library key。Module 節點本身標唯讀則固定整個子樹，包含 ref／inline 選擇與所有子欄位，不論初值是 ref 或 inline；不另設 selector-only 宣告，也不自動強制 inline。

RecipeTab.read_field 每次以一次 `tab.describe_cfg` 讀同一 owner turn 的目前欄位，回傳具名 RecipeFormField，包含 authored input、input_present、ordinary value、key/inline mode、actual key、locked、via_ref、目前 child field names 與 directed errors。`via_ref` 表示路徑穿過 library 引用，不表示宣告唯讀。它不投影 CfgSchema kind，觀察 ref descendants 也不授權寫入。Inline model 的路徑包含 module，例如 `modules.readout.module.freq`。Range count 的路徑是 n。set_sweep 的完整 start／stop／expts 設定以一次 container edit 建立 Range + Expts，不寫 Center／Step 的 inactive paths，也不要求舊 draft 可換算。部分設定仍只編輯目前 active leaves。

實驗 cfg 依 [[0073]] 使用具名裝置欄位，例如 flux_dev、jpa_flux_dev，不再保存 dev role mapping。Form 在 Open／Reset 從 $qubit.flux_source、$mux.flux_source 或 JPA 的 flux_source／rf_source 讀取可覆寫的 direct 名稱。缺來源時留空，必填欄位報錯。使用 flux 的 cfg 必填 flux_unit，值為 A、V 或 native。Core 與 Notebook 不提供隱含 default，GUI 的初始值為 A。Recipe helper 只比 cached unit，成功後寫 caller 指定的具名裝置欄位與獨立 flux_unit，不換算、不連線或 setup。

Core 以 context.devices[cfg.flux_dev] 等具名欄位取 driver。缺名由實際使用者報錯；RunContext 不提供 setters。操作分工見 [[0067]]，Run 前 snapshot 與保存見 [[0063]]。Measure 不保留舊跨 tab source refresh／Unavailable 恢復路徑；使用時綁範本與缺來源的語義由 [[0073]] 擁有。

Form expression 的 `$alias` 與 `$alias.path` 沿固定 ParameterContext 解析，bare concrete paths 使用獨立 namespace。Run 與 Load capture 相同。裸 `$alias` 是名稱字串，不參與數值算術或字串插值。缺列或未綁時 errors 指名 alias，不沿用舊解析值。Expression 不刷新來源。

## Library 與 Inspect

`open_library_entry` 從持久 ModuleSpec／WaveformSpec 建立 LibraryEntryForm。Expression 使用開啟時固定的 env，Apply 將求得的數字寫進 Spec，不把 expression 存成 Library 來源。Name、Saved／Unsaved、Revert 與切換條目前的未存確認由 Inspect 管理。表單 Invalid 或有 local parse errors 時不提交。

New 從 Registry 注入的 TemplateDef 目錄建立 entry input。換範本建新 State 與 Editor，欄位、折疊與捲動回預設，只保留 Name。已綁定的 Slot 保留 child key。生成器負責 defaults 與驗證，caller 不另建 normalization 或 shape catalog。

`context.ml_edit` 使用 Form 路徑陣列與值，一次套用該 entry 的整批 edits。只有有效 Spec 才交既有 Library write owner；錯誤帶欄位路徑，Library 不留下成功前綴。Module writeback 使用相同的 entry form 與 observation，但 draft-local editing 與 selected Apply 的交易邊界依 [[0008]]，不宣稱跨項原子性。`ml_list_roles`／`ml_create_from_role` 保留既有名稱與範本入口。

## 共用 Inspect 與來源

Inspect、Device 與 ARB UI 及其 service 由 session 提供，host 經公開入口注入必填協作者，不靠 host 差異猜能力。Inspect 有 Parameters、Modules、Aliases 三頁。Parameters 使用自己的 YamlValue／FieldSpec，不屬 Form 遷移。Run 或 agent 接管時，Inspect 表單不可改，但仍可選取文字、展開與捲動。

使用者於 2026-10-23 對 alias 表加 participants 列裁定：「甲」。Alias 是目前工作焦點，parameters 是元件事實。Participants 的有序名稱清單由通用 alias owner 保存與解析，不在 recipe 重做這些規則。名稱轉索引屬於領域表單捕捉邊界，不放進通用 alias owner。

Aliases 由共用 Inspect 的第三頁編輯，沒有獨立 AliasDialog 或 aliases wire 名稱。Value 欄可輸入單一識別名、既有 `$` link 或 YAML flow list，例如 `[Q3, Q1]`。清單元素各自沿用 earlier-row link 規則，順序不變。空清單是未綁，不是已選零個元件。List 讀回使用普通固定 capture 的 `lookup`／`lookup_point`／`value`，不另造字串編碼。單一元件路徑及 template placeholder 拒絕清單，不取首項。Apply 將有序 authored rows 整表交 `apply_alias_rows`。它要求已選 point，否則回 `no_context`；成功時先原子保存該 point 的 `aliases.yaml`，再安裝 memory 與發布。失敗保留舊表、檔案、版本與 draft。Alias draft 固定載入時的 entry／point，沿用 Modules 固定 capture 的慣例。Clean draft 隨切換換表，dirty draft 保留。Apply 發現來源與目前 target 不同時拒絕並提示，兩端檔案都不改。同 target 才能提交；Revert 或重開會丟棄草稿並載入目前 point 的表。關閉時未存修改仍走 Save／Discard／Cancel。

MCP 直接呼叫 owning context，不開 dialog、不切頁或焦點。`context.aliases_set` 回已安裝的 authored rows，不解析；解析另讀 snapshot。`context.set_parameter` 用一次 capture 固定具體 target 與 before，保留 metadata、不寫 ledger。這些 mutation 沿既有 owner 與 holder admission，Run 中可寫，但不改既有 Form 或 Run 的固定 capture。`dialog.screenshot` 只擷取目前已開 dialog 的當前頁，不切 Inspect tab。

## 取捨與相鄰責任

單一表單核心讓各宿主共用 input、驗證與 editor，宿主仍決定 lifetime、來源捕捉與提交時機。來源變更不提示、不自動更新。Measure 要保留目前 input 並取新來源就明確 Refresh，要重算初值就 Reset。

使用者於 2026-10-19 裁定：「Reset/Refresh應該是不同分工，而不是捆綁」「我認為Reset可以包含Refresh，這樣方便」。Source basis 記錄 provenance，不是 cfg admission 門檻。這不取代 [[0079]] 還原時的型別與元素 identity 檢查；library 內容版本不同本身不是 identity mismatch。舊 `@{}` 補全、eval／value_ref wire tags 與 editor.* 事件／RPC 不再提供。`value.list`／`value.read` 與內部 ValueRef／resolve_value_ref 保留，見 [[0037]]。

參數容器與 Library 寫入權由 [[0067]] 定義；保存與 crash durability 歸 [[0063]]；workflow run 與 patches 歸 [[0062]]；wire、guard 與 delivery failure 見 [[0068]]。模組入口見 [Form](../../lib/zcu_tools/gui/form/README.md)、[生成器](../../lib/zcu_tools/gui/form_adapter/README.md)、[session](../../lib/zcu_tools/gui/session/README.md) 與 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)。
