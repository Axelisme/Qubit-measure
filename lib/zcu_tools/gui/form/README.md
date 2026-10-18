# gui/form — Qt-free Pydantic input editing

**Last updated:** 2026-10-18，完整草稿 envelope 與不依賴來源的還原

`FormState.describe(path, env)` 直接從 model 宣告與目前 input 回傳 `FormDescription`，不產生 JSON Schema。Root path 是 `()`，其他位置用 model-field tuple。`target` 是指定欄位的 metadata，root 則為 None；`children` 只列下一層 metadata。每個 child 保留完整 path，不攜帶 descendants。選中選項只有一個非 nullable、沒有 options 的普通 model wrapper 時，children 攤平一層，但 wrapper 的完整 path 仍可查詢。

`FormFieldDescription.can_expand` 表示目前結構可到達 model，不保證展開後 children 非空。普通 model 與選中且帶 fields 的 option 為 true；whole leaf、停用 nullable、no-match 與不帶 fields 的 option 為 false。判斷不執行 child hooks，也不建立 child metadata。固定欄位排除、動態宣告與 wrapper 攤平仍在實際展開時決定 children。

`selection_fixed` 只表示 annotation 在本層允許一個非 null 選項分支，不凍結內部 choices。None 代表啟停，不算另一個分支；nullable 單分支在啟用與停用時都保持 selection_fixed=true。`fixed`／`fixed_value` 表示 leaf 或完整 option value 由宣告固定。固定外層 model 仍可有可切換的內層分支，此時 selection_fixed 為 true、fixed 為 false。兩種固定都不依當次 env 的選項數推定。

Description 的 options 只列 detached value、label 與 command name／label。`selected` 是依 JSON 型別與 value 葉匹配的第一項 index，missing／no-match 為 None，不猜第一項。`options=None` 表示沒有 selector，空 tuple 則保留顯式空清單。Describe 與 `run_command` 共用 active ancestry、最深 local 覆蓋與選項解析，不查 inactive branches 的 hooks。普通缺值 model 可按型別列欄位；停用 nullable model 與 no-match option 只回 target。Unknown、whole-leaf descendant 與 inactive descendant 的 path 立即 raise `FormPathError(ValueError)`，其 `path` 欄位保留完整請求路徑。作者宣告、nonfinite 與其他非 path 失敗維持原例外類別。Describe 不求值、不套 defaults、不寫 input/base，也不抓取來源；宣告 hooks 只讀呼叫端固定 env。有限的 leaf constraints 是顯示提示，驗證仍由 result／preview 負責。

`static_defaults(model)` 接受普通 Pydantic BaseModel 或持久 Spec，產生 detached input JSON。必填欄位與 None 預設保持缺席，讓 before validator 能從其他欄位推導值。Default factory 每次呼叫重新執行。FormModel 初值與宣告者的 converter 可使用這個入口。

FormModel 的實例只承載普通值。類別上的 defaults、choices、options 與 converters 提供輸入契約。FormInput 保存未填、null、expression 與 complex tags；FormState 擁有 draft/base、batch write、branch switch、結果、差異與 compatible restore，不擁有 Engine revision 或提交 policy。

Expression 只用於 int、float、complex 與 str 文字欄位，只有顯式 `{"expr": str}` 求值。Literal、Enum、bool、宣告 choices 的 leaf 與 library key 沒有 expression 能力。Describe 回 `expression=False`，result 拒絕 envelope 且不求值，compatible restore 丟棄它。文字與選項混合的 union 沒有對應控制項，describe、求值與 restore 會 raise 帶欄位路徑的 ValueError。Editor 只看 description 的 expression flag。普通字串包含 `$` 前綴仍為 literal。求值共用 evaluate_scalar_expr，再遵守 Pydantic 約束；missing／unbound alias 指名失敗，不以預設值代替。Input 與 compatible restore 保留 expression 原文。

Defaults hook 可以用 `{"expr": ...}` 宣告 scalar 初始值的來源，例如 `$flux_dev` 或 point path。`default_input` 由 core 用同一個 env 當場求值，input 只存算好的 direct 值；來源之後改變，要重新建立 default input 才反映。Expression 算不出來，或值讓 result 在該欄位報錯時，欄位留空：required 欄位只顯示 `Field required`，有 Pydantic 靜態預設的欄位驗證時用那個預設。來源本身不可用的錯誤照常拋出。使用者或 MCP 寫入的 expression 保留原文。

`Option` 宣告 label、部分 JSON input value、可選 BaseModel fields 與 commands。`options(env)` 是宣告 hook，consumer 讀 description 的 options 有效清單。Core 自動產生 Literal、Enum、choices 與 tagged union 選項。最深 local 宣告優先，包含表示無選項的空清單，非重疊 parent 宣告保留。Bool 與一般 nullable model 照舊用 checkbox 與 Enabled。Fields 只在 describe 該 active path 時列直接 children，不顯示 value 已決定的 tag。比對只看 value 的每個葉，JSON 型別與值相同即符合，第一項匹配優先。無匹配時 consumer 顯示目前值加 `(missing)`，不寫 state。

`FormInput.same_as(other)` 比較 model identity、raw data 與 selections 的完整草稿。實際 branch tags 在 data 中，一併參與比較。它不比較 lowered cfg 或持久 Spec，不求值、不執行 converter。`FormState.is_dirty` 使用同一比較入口判斷 current／base，包含 selection-only 差異。Install、reset 與 logical edit 的 no-op 判斷也共用它。`changes(env)` 仍只回有效 input 的 data／evaluated leaves，不是 dirty predicate。

`InputConverter` 是一個欄位的單一 converter，處理全部目標選項。可在 Annotated metadata 或 `FormModel.converters(env)` 宣告。Local metadata 覆蓋祖先宣告，同一 model 的 hook 再覆蓋自身 metadata。Callable 接收 readonly snapshot、目標 selection 與局部 evaluate context，回 `FormConversionResult(structure, input)`。它與 command 共用原子 installer，Core 不補 defaults。

沒有 converter 時，切換只改 selection，保留原有 branch 和 authored input。`FormInput.selections` 攜帶記憶體選項，`data` 保存實際 child input。`describe.selected` 顯示選項，`describe.branch` 標示實際子表單；兩者可以不同。Partial selector object 在轉換後套用明確提供的 children，tag 葉寫入也走同一個 converter。純讀不執行 converter。預期的缺來源、參與值求值及結構 admission 拒絕使用 `FormConversionError`，保留 cause。任意 callable ValueError 或 provider 故障不轉成正常拒絕。`FormInput.to_json/from_json` 使用完整草稿 envelope，不走 compatible restore。Envelope 保留明確 selections、raw branch、expression、缺席、null 與普通 invalid 值。還原以可信 annotation 的型別 identity 比對，不執行 converter 或宣告 hooks，不補 defaults、不讀 library。具 discriminator 的 branch 與 selection Literal 不符 annotation 時拒絕。未標 tag 的 model-only union 保留原稿，不猜 branch。`write_compatible` 仍是明確的葉投影工具，不是 envelope decoder。

有 env 的一般驗證只對 `FormInput.selections` 明確 overrides 檢查目前 options membership。不在選項中時，result／preview 回該欄位錯誤，保留原稿。沒有 override 的 raw branch 仍使用原有值契約，不把選單當作完整值域。Ref 的存在性與 family 由 adapter 的一般來源／choices 驗證處理，不參與還原 identity。

明確指令用 `OptionCommand(name, label, run)` 或 `InputCommand` 宣告。前者只在目前第一個匹配選項提供指令，後者掛在欄位 metadata 或 active `FormModel.input_commands`。`FormState.run_command(path, name, env)` 只替換掛載欄位與 descendants，不允許葉命令修改 parent。Callable 接收 `FormCommandSnapshot` 和 `FormCommandContext`，回 `FormCommandResult(selection, structure, input)`。Snapshot 的 selection、結構與原始 input 深度唯讀。Context 的 evaluate 使用相對掛載欄位的 path，只求值已填寫的參與欄位，不補缺值。Core 檢查 annotation 允許的結構與作用範圍，再一次安裝與通知。普通值錯誤保存成 invalid draft。

未知或不適用指令、來源與結構失敗不寫入，不建立第二份 state。`defaults(env)` 只依型別與固定來源產生新 input，沒有 previous 參數或隱含 ref 複製。

每個 intrinsic tag 都有一個普通 Option。多個 tag 指向同一 model 時，各選項共用 branch fields，保留 tag 的 JSON 型別與宣告順序。Core 從 annotation 讀取 tag，不從動態 choices 或 consumer 標記推導身分。Model 欄位可達的每個 branch 都是一個具體 model，同一 tag key 的具名 union 展開成各成員。Choices 仍限制當次可接受的 leaf 值，不改目前 input 的 branch。

`FormModel.default_input` 與 `FormInput` 在入口拒絕未標 tag 的 `str | Model` 欄位。ValueError 指出欄位路徑與不支援的形狀。檢查走過 nullable、alias 與所有 model branches，包含未啟用的分支；它不讀 captured sources。Key-or-inline input 必須使用明確的 tagged ref／inline union，不能靠 JSON 字串或物件型別選分支。持久 Spec 的 `str | Spec` 不改，生成器在 input 邊界投影成 tagged wrappers。

Sweep 的四種數值 model、座標／採樣換算與 row command 宣告由 `gui.form_adapter.sweep` 擁有。Core 不辨識 sweep 型別或其 tag。

Module／library 的 tagged input、literal key 與 family 政策由 `gui.form_adapter` 擁有。Core 只讀普通 annotation、`Option` 與 `InputPolicy`，不以領域 subclass 判定行為。`InputPolicy(expression=False)` 放在 field 的 Annotated metadata，關閉 describe、decode/result、defaults evaluation 與 compatible restore 的 expression 能力。True 不會替 Literal、enum、bool、choices 或混合 union 開啟原本不支援的能力。Write 仍保存普通 invalid draft，由 result 回報。

Editor 與 MCP 讀同一份 description options，照 label 列選項，一次寫入 value，首次展開才 describe 選中項目的 children。它們不推論 union、ref 或 library 的呈現。普通 nullable model 仍用 Enabled，bool 仍用 checkbox。明確複製只經 `to_inline` 指令，使用 host 的 guarded command callback；一般選 Inline 由 `form_adapter` 的集中 converter 決定繼承或新建 input。

Scalar、整個 sequence 與整個 typed mapping 都是 leaf。普通 nested model 只編輯子欄位，一般可留空 model 區段仍用 Enabled 控制 object／null 啟停。Choices 使用 model-field leaf paths，驗證與 description 共用同一份當次選項。Parent 與 child 宣告同一 leaf 時，最深 nested FormModel 的 local choices 優先，包含空選項。非重疊的 parent 選項保持有效，重用同型 nested model 的欄位各自隔離。

Compatible restore 以原文欄位集合與型別選擇未標 tag 的 model-only union。恰好一個 member 相容才保留 subtree，合法 expression envelope 依欄位能力參與比對，不求值。零個或多個 member 相容就丟棄，不猜第一個 branch。Tagged branch 的 selector 與 defaults 契約不變。還原後 captured source 檢查失敗的 leaf（例如 library 沒有的 ref 名稱，或沒有 captured library 時的任何 ref）保留目前 input，目前沒有就留空，路徑和其他 dropped paths 一起回傳。不是由還原 leaf 造成的 source 失敗仍然 raise；一般 write、default_input 與 result 的 source 語意不變。

`FormModel.check_sources(data, env)` 檢查本層 raw input 真正消費的來源。Hook 不遞迴、不補 defaults、不驗證普通值，也不改 capture。Core 沿 active children 與被消費的靜態 defaults 呼叫同一 hook，供 defaults、write、preview/result 與 compatible restore 共用。`field_error` 的相對路徑由 core 加上 consuming ancestry。Compatible restore 只回退造成來源失敗的 restored leaf；其他來源故障保留原例外。

`FormState.prepare_write(batch, env)` 回 detached `FormInput`，完成與 write 相同的結構及來源 admission，不安裝、不通知。`install_input(candidate)` 只接受同 model，保留 base，內容改變才通知一次。Host 先 guard，再 prepare，再 install。候選只限同一 owner turn、current input 與 env 使用，不能跨 await、Reset 或 guard 邊界保存，也不是授權或 revision token。普通 `write` 使用同一流程。

FormEnv 借用 host 捕捉的 ParameterContext、library 與 detached `BaseDeviceInfo` snapshots（含 custom driver info）。共用工具只讀裝置名稱；需要特定 driver 欄位的表單自己用 `isinstance` 收窄。沒有 parameter capture 時仍可編輯 ordinary input 與純數學 expression；參數引用失敗會落在消費欄位的 error path。工具不寫來源、不 refresh 或 poll。領域來源 hook 可以讀固定 Spec；to_cfg 的 materialization 與 range 展開歸具體表單。

FormState 的 subscriber 在 input 安裝後同步執行。個別 subscriber 故障記錄 traceback，不回滾 input，也不阻止後續 subscriber 或 host publication。Host 的 mutation guard 涵蓋安裝、editor subscriber 與 resource publication，同一通知期間可讀不可重入 mutation。

FormState.preview 提供隔離的 ordinary leaf values 與完整 errors，供 view 在 model-invalid draft 中顯示其他欄位的新求值結果。失敗的 expression 或含失敗子值的 whole container 不提供值。成功時包含 Pydantic defaults 與 coercions，不把它們寫回 input。Preview values 不代表可提交；host 仍檢查 errors 與 view 的 local parse errors。

WriteRejected 是 host 寫入入口拒絕一批輸入時 raise 的具名例外，message 由 host 寫給使用者看。Host 在寫入任何內容前 raise，FormState 維持原狀；model 會拒絕的值仍寫進 state，由 result 回報。Core 自己不 raise 它。Editor 收到時保留該欄輸入並顯示 message，其他例外照常拋出。

Display 只承載欄位呈現資訊。順序沿欄位宣告，段落沿 nested model；collapsed 只用於子 model。`group` 是同層畫面群組名稱，空字串在建構時報錯。`decimals` 是 float 顯示精度，非 float 宣告在 describe 報錯，不改 input。`hidden=True` 不畫 controls，但值、預設與驗證保留。Nested model 沒有 label 時，title 是欄位名，底線換空白且只有首字母大寫。普通 model 的單值結構 tag 標 hidden；branch 身分從 annotation 判斷，包含其子類別。Union 位置的 tag 仍是 selector。`Display(selector=False)` 只隱藏 dropdown，仍保留 selector metadata 與 active branch。Editor、host 與 MCP 共用 description，但不反向成為 core dependency。

跨模組契約見 ADR-0065、ADR-0062 與 ADR-0008。Measure、Library／Inspect、writeback 與 workflows 共用本核心，各宿主持有 state 並決定 capture、Apply 與保存。Core cfg 推導與持久 Spec 投影由上層 `gui.form_adapter` 組合，不放進本 package。Complex codec 直接使用 `core.complex_codec`，不經舊 cfg owner。
