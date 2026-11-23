---
status: accepted
---

# 0065 — Cfg 編輯模型與使用邊界

## 局部取代範圍

已由 [[0079]] 取代，見[〈決策狀態與問題〉](0079-form-field-transitions.md#決策狀態與問題)；[原文](retired/cfg-editing-replacement-scope.md#局部取代範圍)僅供追溯。

## 問題與決策

**規則。** Measure 的 `TabFormResource` 擁有一份不依賴 Qt 的 `FormState` 與固定來源，讓 Qt、remote 與 Run 使用同一份 cfg。`TabFormResources` 管理建立、查找、操作禁令與撤銷。前端只讀單層 description、input、preview 與 errors，修改走 resource 的公開入口。Form 決定 defaults、形狀、驗證與 core cfg 轉換，wire caller 不重建這些規則。

**例子。** 有效的 Expr 初值若在支援 Eval 的欄位，Form 保留原式，初開畫面顯示 Expression 模式與公式。宣告 choices 的文字或數值欄位沒有 Eval 模式，初值仍求值並檢查是否可接受，保存為 direct 值，不交給拒絕 authored Expr 的規則清空。

**邊界。** Library、Inspect 與 writeback 共用 FormState／FormEditor，但各自持有 state 與提交政策，不借用 tab resource。Library 寫持久 Spec，writeback 的 opaque draft 與 selected Apply 見 [[0008]]。Expr 初值的能力分界也適用於分支初值與相容還原補入的 defaults，不新增模式或保存格式。Literal、Enum、bool 與禁用 expression 的 InputPolicy 不新增初值求值能力。求值或欄位驗證失敗時留空；靜態預設和來源故障照既有規則處理。詳見 [Form README](../../lib/zcu_tools/gui/form/README.md)。

**依據。** 2026-11-02，使用者對 Expr 初值答「採用建議」。原始討論未入庫，裁決記於 Git `b67f3d549`，choices 的處理見 `f5da919ba`。

### 表單錯誤的呈現

**規則。** 共用 FormEditor 讓每個可見的無效欄位保留 inline 紅字。底部摘要只顯示第一個沒有可見欄位承接的錯誤及完整路徑，包括折疊、hidden 與 model／group 錯誤。錯誤都已由可見欄位呈現時，摘要隱藏。Measure 的 Run 區顯示無效欄位總數，tooltip 只提示修正標紅欄位。Run 保持停用，修正後清空提示並恢復既有 admission。

**例子。** Start 與 Stop 都無效且展開時，表單只留兩行 inline 錯誤，Run 區顯示 `Config invalid: 2 fields`。收起這兩欄後，底部摘要顯示第一個未呈現的錯誤及路徑。計數按 canonical FormPath 合併 resource model／lowering 與 editor-local 錯誤，包含未提交的 parse errors，不按訊息去重。單複數與 `errors_changed` 的更新方式見 [Tab admission](../../lib/zcu_tools/gui/app/measure/README.md#tab-admission)。

**邊界。** 這只改呈現，不改驗證、提交或 domain admission。MainWindow 的 Ready、Saved 等狀態列訊息不變；無 Form 的歷史 tab 仍依 [[0073]] 顯示 factory 原因。Workflow、Inspect 與 writeback 共用摘要規則，各自保留提交政策。

**依據。** 2026-11-16，使用者指出：「我說的重複是UIUX的layout問題，比如說這張圖，應該只有兩個問題，但4個提示」，並選擇選項 B。沒有另建討論快照，裁決見 Git `8ea32f872`。

## 共用機制與領域 owner

**規則。** `gui.form` 擁有 input、FormState、單層 description、preview、result 與錯誤。`widgets/form` 的 FormEditor 依 description 畫欄位與選項，不解讀 library 或實驗語義。`gui.form_adapter` 從 core cfg 或持久 Spec 生成 Form，處理 FieldMeta 投影、key-or-inline wrapper 與屏蔽欄位；Form core 不讀 FieldMeta。實驗 adapter 用 `Form(core_cfg, tree)`，由 annotations 定義可編輯形狀，tree 定義初值、唯讀屬性、Slot 與 Template。

**例子。** Library entry 的表單由 session 持有，初始 input 與提交都保留 child key。不能把既有 pulse 改成 readout；換 concrete Spec family 要新建條目。Optional None 維持缺席。選填 `desc` 由生成器略過，提交時用 Spec 預設，不保留舊值；無預設的必填欄位仍可編輯。None、library key 與 Inline kinds 共用一個下拉，key 只顯示名稱，不展開唯讀 descendants。普通選 Inline 按 converter 的型別配對繼承，明確的 `to_inline` 才從固定 library 複製。Advanced 預設折疊，Expression 只用於支援它的文字欄位，不用於下拉或勾選。

**邊界。** 欄位轉換、固定 list 與保存還原以 [[0079]] 為準。[Form core](../../lib/zcu_tools/gui/form/README.md) 的 `select` 執行宣告的 converter；沒有 converter 時只改 selection，子表單結構和 input 不變。Memento 保留 selection／branch 和原始 input，不保存來源覆寫。`restore_input` 不跑 converter；型別與固定元素依 [[0079]] 的還原規則處理。[生成器的固定元素宣告](../../lib/zcu_tools/gui/form_adapter/README.md#固定元素宣告)只約束選用它的表單。Open 與 Reset 計算 defaults；Reset 可在同一資源換形狀及初值，成功後不再接受舊 editor state 寫入。`make_form`、範本及固定來源見 [[0073]]。

**依據。** 2026-10-18 採用的欄位轉換、固定 list 與保存還原設計見 [[0079]]。生成器與 widget 的分層見[選項分層討論](../discussions/2026-10-13-form-options-layering.md)，實作細節見[生成器 README](../../lib/zcu_tools/gui/form_adapter/README.md)。

## Input 與宿主提交

**規則。** FormInput 保存未填、null 與 expression 原文，FormModel 實例只承載求值後的普通值。FormState 擁有 draft／base，不持有 Engine revision。Workflow 的 revision 由 Engine 持有並發布，見 [[0062]]。FormEditor 保留尚未 parse 的文字及 local parse errors。宿主決定 capture、Apply、Discard 和何時保存 input；提交前先查 editor 的錯誤，再取 Form result，不以 preview 代替。宿主不另建 evaluator、validator 或 codec。

**例子。** Library 與 writeback 提交持久 Spec。Workflow 宿主提交普通 plan／tunables，並在 Engine 成功發布後才接受 input／base。Expression 留在 FormInput，不進 Engine journal，見 [[0062]]。

**邊界。** 表單用初值與 `Display(readonly=True)` 宣告唯讀。Form core 對 Qt、MCP 與 RPC 採同一規則：同值可寫，不同值拒絕，父層與批次也不能繞過。RPC／MCP 回 `error.reason="form_input_rejected"` 及 `error.data.form_path`；只有唯讀拒絕另附 `error.data.reason="readonly"` 與目前值的 JSON `error.data.current`。還原保留唯讀欄位的存檔值，不重算；能否執行由一般驗證決定，見 [[0079]]。Form core 不依賴 app、widgets、remote、experiment 或 Qt，editor 不依賴 app、remote、experiment 或 MCP。`.importlinter` 的 `form-core-no-ui-policy` 與 `form-editor-no-host` 檢查這兩條邊界。

**依據。** 2026-10-16，使用者以「都採用建議」確認將穩定邊界併入本篇及 [[0062]]。2026-10-19，使用者問「那為什麼不直接都用ReadOnly就好？」並答「1同樣值不報錯。2我選存檔的版本，因為有些lock值可能是生成的。能不能執行是另外一回事」。原始討論未入庫，分別見 Git `854f4198b`、`c499c06d9` 及 [[0079]] 的還原裁決。方法與欄位見 [Form README](../../lib/zcu_tools/gui/form/README.md) 及 [FormEditor README](../../lib/zcu_tools/gui/widgets/form/README.md)。

## Measure tab 的 input 與使用

**規則。** FormInput 保存 model field names、普通 JSON、顯式 `{expr: text}` 與 complex tags；普通字串保持 literal。未完成或一般值無效的草稿仍可編輯。結構或來源錯誤拒絕整批，不留成功前綴。Preview、result 與 description 不改寫 input；preview 可投影有效的子表單，不代表整份 cfg 能 Run。Run 只接受目前有效的 Form result，不收 cfg ref 或 expected revision，不回退上次有效值。接受後的執行值與來源 provenance 深層隔離，後續 Edit、Reset 或全域來源變更不影響該 Run。

**例子。** Sweep 的 Range／Center 與 Expts／Step 各只保存 active 欄位。座標切換保留 n／step 的原文、錯誤或缺席，採樣切換才用目前的採樣值。掛在 sweep row 的 command 只求值參與欄位；無關錯誤不阻擋換算。成功才把參與的 expression 改成 direct 數字，之後新輸入的 expression 照常保存。失敗保留模式與全部原稿。FormState 核對 annotation 和 subtree 後原子安裝，不改 parent 或 siblings。普通切換也走 [[0079]] 的安裝流程，不由核心另補 defaults。Step 的 `start=0`、`stop=1`、`step=0.3` 保留原稿 `stop=1`，Qt 顯示有效 stop 0.9。點數為 `floor(|span|/|step|)+1`，有效 stop 為 `start + step × (n−1)`；零 span 只接受 `step=0` 的單點。Center 用有效端點投影 center／span。Adapter 用同一投影與原 step 產生 SweepCfg，不重算；執行端仍驗證方向、單點與點數一致性。非零 span 的步長、方向與 `1e-9` 相對容差見[生成器的 Sweep 表示](../../lib/zcu_tools/gui/form_adapter/README.md#sweep-表示)。非法或未完成的 draft 不縮成其他可執行掃描。Memento 保留四種 active input 與 expression，Reset 回 fresh Range + Expts。

**邊界。**
- Cfg 讀寫回 input、values、errors，不附 schema。Values 列出 path／value，errors 列出 path／message。Canonical path 以 `str` 欄位名、`int` list index 組成。`tab.describe_cfg` 只讀宣告、目前 input 和一層 children，不 materialize 或呼叫 `to_cfg`。普通單一 model wrapper 可攤平一層，完整 path 仍保留 wrapper。欄位回覆含局部 input_present、input、value 與定向 preview errors；get_cfg 或 Run admission 檢查 lowered errors。Get 與 describe 另回全表 diagnostics，合併 preview／lowered errors，不受 picker 或 describe path 限制。逐位置種子及未知校準狀態見 [Seed diagnostics](../../lib/zcu_tools/gui/form_adapter/README.md#seed-diagnostics)。Unknown、inactive 或 whole-leaf 子路徑回 `invalid_params`／`form_path`，保留請求的 `data.form_path`，其他 ValueError 維持 `controller_error`。Writeback 的 `observe_entry` 也只回 input、values、errors。Agent 的 logical 路徑、GUI 內部 canonical 入口及各方法的回覆見 [[0068]] 與 [Tab cfg 協定](../../lib/zcu_tools/mcp/measure/README.md#tab-cfg-協定)。
- Description 只列 active path 宣告的 command。`tab.run_cfg_command` 經 resource 的控制身份、busy、lifetime 與 mutation guard，成功回 input／values／errors。Qt 檢查參與欄位的 local parse errors，不猜模型、不換算。Sweep 指令放在 row 的右鍵選單，標準文字與 Expression 選單不變。禁編輯時也禁指令。Options 投影、fixed 與延遲展開見 [Form README](../../lib/zcu_tools/gui/form/README.md)。Editor 第一次展開才查 children，收起保留 widgets 與未提交文字，換選項只換該子樹。折疊錯誤聚合到可見祖先。各宿主共用 editor，不共用 capture、Apply／Cancel、guard 或 lifetime。
- Read、preview、result 與 accept 都不捕捉來源。通知期間能讀，不能 mutation、accept、Run 或 close。Subscriber 出錯記 traceback，不回滾 input，也不阻擋其他 subscriber。來源就緒才建立 tab，建立即捕捉並可 headless 編輯。腳本只把 opaque `TabInput` 交 Form owner 還原，不重建 Form 或來源。未知 adapter、無效 JSON 或來源建立失敗在發布前報錯。輸入範圍與丟棄路徑見 [Measure README](../../lib/zcu_tools/gui/app/measure/README.md#composition-root)；框架沒有 pending 還原，啟動責任見 [[0064]]。
- Qt attach／detach 只管理 view 與訂閱。Idle close 撤銷舊 handle，同名重開仍是新資源。Active Run 禁同 tab 的人工 Edit、Reset、replacement 與 close，其他 tab 仍能編輯。控制身份見 [[0070]]，hardware guard、lease 和 operation lifecycle 見 [[0066]]。Load 只載結果與分析來源，不要求現有 cfg 有效，不回填任何 Form input，也不換其固定來源。Wire 回結果與分析準備狀態，不回填表單狀態。保存的 native cfg 可經公開 `datafile.load` 離線讀為 Container，也可在 Results browser 唯讀查看，見 [[0063]]。

**依據。** 2026-10-15，使用者對共用單層描述與延遲展開答「好，先照這個規劃開始」。原始討論未入庫，裁決見 Git `54ea21f0d`。Sweep 換算的背景見[初值與 Sweep 模式討論](../discussions/2026-10-17-measure-form-initial-values-and-sweep-mode.md#sweep-模式)，現行切換範圍依 [[0079]]。

## 固定來源與 Form library key

**規則。** Open、Refresh 與 Reset 各捕捉一次 point、library、alias 表、cached device infos 及已發布的 value sources，不輪詢硬體。Edit、parameter_read、preview、result 與 Run 使用這份 capture，全域事件不刷新 tab。`read_context(aliases=None)` 固定 point、library、storage 與全表。非空 aliases 只覆寫這次 capture 的具名列，保留 dependent authored links，不改全域表。省略、None 和 `{}` 都不覆寫。EMPTY 無覆寫回 None，保留 context-free defaults；非空覆寫拒絕 `no_context`。

**例子。** Refresh 更換 capture 並重驗全部 input 與 explicit selections，expression 隨新來源求值但原文不動，choice 初值或使用者填的 direct 值也不動。缺來源仍是欄位錯誤。Reset 更換 capture 並重算 defaults。兩者的單次 aliases 都不存進 alias 表或 memento。`tab.parameter_read` 只讀 `resource.env.parameters`，不收 aliases。`tab.run_start` 在改焦點或開始 operation 前拒絕非空 aliases；None 或 `{}` 使用畫面與固定來源。Recipe 的 `open_tab` 不論新建或 reuse 都先 Reset，再設定參數及 Run。Target 只覆寫這次 Reset 的 aliases，不在 preview 或 Run 另行解析。Recipe 從作者宣告的 inputs model 取得參數契約，不從 tab Form 產生，見 [Recipe 與 execution](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution)。

**邊界。**
- Form Slot 的 `ModuleRef` 保存 `{kind: ref, name: actual_library_key}`，`ModuleInline` 保存 `{kind: inline, module: Spec input}`。巢狀 key-or-inline 欄位以普通字串作 key。Recipe 選 library 時用實際 key，不建立 `__ref` intent。子欄位唯讀不鎖住祖先的 library 選擇；如果 Module 節點本身唯讀，ref／inline 選擇及整棵子樹都固定。不另設 selector-only 或強制 inline 是 [ADR-0073〈Form 的形狀與初值〉](0073-measure-form-derivation.md#form-的形狀與初值) 所記的 designer 實作選擇。
- Recipe 的 `tab.read` 一次讀同一 owner turn 的欄位。回覆包含原稿、presence、ordinary value、key／inline 模式、實際 key、locked、via_ref、現有 children 和定向錯誤。`locked` 由 selection_fixed、fixed 或宣告唯讀決定，不看當次選項數。`via_ref` 表示穿過 library 引用，不表示宣告唯讀，也不授予寫入。Canonical inline path 含生成的 `module` wrapper，logical path 省略它。入口見 [[0068]] 與 [Tab cfg 協定](../../lib/zcu_tools/mcp/measure/README.md#tab-cfg-協定)。完整的 sweep container edit 可一次建成 Range + Expts，不要求舊草稿可換算；部分更新只寫 active leaves。
- 實驗 cfg 依 [[0073]] 使用 `flux_dev`、`jpa_flux_dev` 等具名裝置欄位，不存 dev role mapping。Open／Reset 從 `$qubit.flux_dev`、`$mux.flux_dev` 或 JPA 的 `flux_dev`／`rf_source` 填可覆寫的 direct 名稱。缺來源留空，必填欄位報錯。使用 flux 時必填 `flux_unit`（A、V 或 native），Core 與 Notebook 無隱含預設，GUI 初值為 A。Recipe helper 只比 cached unit，通過才寫 caller 指定的具名欄位及獨立 `flux_unit`，不換算、連線或 setup。Core 用具名欄位向 `context.devices` 取 driver，缺名由實際取用 driver 的程式報錯；RunContext 無 setters。操作分工見 [[0067]]，Run 前 snapshot 與保存見 [[0063]]。
- `$alias` 與 `$alias.path` 讀固定 DocumentSetCapture，bare concrete paths 使用獨立 namespace。Run 與 Load 使用相同的 capture 契約。裸 `$alias` 是名稱字串，不做數值算術或字串插值。缺列或未綁時指名報錯，不沿用舊解析值，也不刷新來源。Measure 不用舊跨 tab source refresh／Unavailable 恢復路徑；範本綁定與缺來源依 [[0073]]。

**依據。** 固定文件組的推導見 [Global／point 參數型別收斂與文件組](../discussions/2026-11-17-global-point-parameter-types-and-document-set.md)；Refresh 與 Reset 的裁決見[取捨與相鄰責任](#取捨與相鄰責任)。具名裝置欄位與單位一條是 agent 的選擇。使用者 2026-11-18 過目：「我大概看了一遍，基本上是没问题」，並說這類規則「并不算是裁决，而算是建议」；只作參考先例。原始對話未入庫，見 Git `1f4b16aa0`。

## Library 與 Inspect

**規則。** `open_library_entry` 從持久 ModuleSpec／WaveformSpec 建立表單，以開啟時的 env 求值，寫入 Spec 的是數字，不是 expression。Inspect 三頁採隱式提交，不提供 revert 或 undo。Modules 只在 Enter、一般離開欄位、改下拉或勾選時，把 Name 與有效 Spec 一次 replace 交 `apply_library_change`，打字停頓不提交。

**例子。** New 從 Registry 注入的 TemplateDef 目錄建立條目。換範本就換 State 與 Editor，欄位、折疊與捲動回初始狀態，只保留 Name。已綁 Slot 保留 child key。生成器負責 defaults 與驗證，caller 不另造 normalization 或 shape catalog。

**邊界。** Invalid 或有 local parse errors 時標紅、不提交。Owner 拒絕時留錯誤與可編輯輸入，不改檔；revision 衝突用 [[0081]] 的 Keep／Reload。切換條目、point 或關閉時丟棄未提交輸入，不問 Save／Discard／Cancel，不顯示 Apply、Revert 或 Saved／Unsaved。`context.ml_edit` 用 Form 路徑陣列和值套用同一 entry 的整批 edits，錯誤帶欄位路徑，不留成功前綴；只把有效 Spec 交 Library write owner。`ml_list_roles`／`ml_create_from_role` 保留名稱與範本入口。Module writeback 共用 entry form 與觀察方式，不共用 Inspect 的隱式提交；draft-local editing 與 selected Apply 依 [[0008]]，不保證跨項原子性。

**依據。** 2026-10-25，使用者說：「但我比较想是我改完就自动 apply，有隐式 apply。然后 revert 的话，感觉三个 panel 都可以有，只是 UI UX 怎么设计我还不确定。」「因为目前是 write through，所以我觉得好算了，就不做 reverse undo 功能了。」原始討論未入庫，裁決記於 Git `ac82fca14`；面板細節見 [session README](../../lib/zcu_tools/gui/session/README.md#結構)，範本 RPC 見 [MCP Measure README](../../lib/zcu_tools/mcp/measure/README.md#library-template-rpc)。

## 共用 Inspect 與來源

**規則。** Session 提供 Inspect、Device 與 ARB UI 及服務。Host 經公開入口注入協作者，不從 host 差異猜能力。Inspect 有 Parameters、Modules 與 Aliases 三頁；Parameters 沿用 YamlValue／FieldSpec，不屬 Form 遷移。Alias 表保存工作焦點，parameters 保存元件事實。通用 alias owner 保存及解析 `participants` 的有序名稱清單，名稱轉索引由領域表單的 capture 處理，不在 recipe 重做通用規則。

**例子。** Aliases 的 Value 欄可填單一識別名、既有 `$` link 或 YAML flow list，例如 `[Q3, Q1]`。清單保留順序，每個元素照 earlier-row link 規則。空清單表示未綁，讀取沿固定 capture 的 `lookup`／`lookup_point`／`value`，不另編碼。只接受單一元件的路徑或範本 placeholder 遇到清單會拒絕，不取首項。

**邊界。**
- Aliases 由 Inspect 第三頁編輯，不另設 AliasDialog 或 aliases wire 名稱。每次完成儲存格編輯或 Remove，把有效有序的整表交 `apply_alias_rows`，寫 Inspect 所選 Global 或目前 point。Add 的名稱有效前只留在畫面，空 Value 是合法的未綁值。無效列標紅不寫入。未開 entry 回 `no_context`；成功先原子保存 `aliases.yaml`，才安裝記憶體與發布。失敗保留舊表、檔案、版本及畫面輸入。Revision 衝突依 [[0081]]。切換 point 或 entry 丟棄未交輸入並載入當前表，關閉不詢問，也不提供 Apply、Revert 或 undo。
- Run 或 agent 接管時，Inspect 禁止編輯，但仍能選取文字、展開及捲動。MCP 直接呼叫 owning context，不開 dialog、不切頁也不改焦點。`context.aliases_set` 回已安裝的 authored rows，不做解析；解析另讀 snapshot。`context.set_parameter` 一次 capture 固定 target 與 before，保留 metadata，不寫 ledger。這些 mutation 仍走 owner 與 holder admission，Run 中可寫，但不改既有 Form 或 Run 的 capture。`dialog.screenshot` 只擷取已開 dialog 的當前頁。

**依據。** 2026-10-23，使用者對 alias 表加 participants 列答「甲」。原始討論未入庫，裁決記於 Git `09113733b`。Alias 解析及持久化見 [session README](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot) 和 [[0067]]。

## 取捨與相鄰責任

**規則。** 各宿主共用 input、驗證與 editor，但自行決定 lifetime、來源捕捉與提交時機。來源變更不提示，也不自動更新既有 tab。

**例子。** Measure 要保留 input 並讀新來源時用 Refresh；要用新來源重算初值時用 Reset。

**邊界。** Source basis 記 provenance，不是 cfg admission 門檻。還原的型別與固定元素政策見 [[0079]]，不把 library 內容版本不同視為 identity mismatch。`value.list`／`value.read` 與內部 ValueRef／resolve_value_ref 歸 [[0037]]。參數容器和 Library 寫入權見 [[0067]]，保存與 crash durability 見 [[0063]]，workflow Run 與 patches 見 [[0062]]，wire、guard 與 delivery failure 見 [[0068]]。

**依據。** 2026-10-19，使用者裁定：「Reset/Refresh應該是不同分工，而不是捆綁」「我認為Reset可以包含Refresh，這樣方便」。原始討論未入庫，裁決見 Git `a1377ef12`；現行來源交接見 [Measure README](../../lib/zcu_tools/gui/app/measure/README.md#sources-and-workflow)。
