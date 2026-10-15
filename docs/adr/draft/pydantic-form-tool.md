---
status: draft
---

# Pydantic 表單、輸入與宿主提交

**Last updated:** 2026-10-15，共用 Form 與持久 Spec 編輯

## 脈絡與範圍

Workflow plan 與 tunables 需要 sequence、nullable、tagged union、typed mapping、expression 和 library module reference。共用表單用 Pydantic 表示普通值，用獨立的 input JSON 保存未填與 expression。Measure、Library、Inspect、writeback 與 workflows 使用同一核心和 FormEditor。Measure 推導見 [[0073]]，各宿主提交邊界見 [[0065]]、[[0008]]。

保存與來源責任沿 [[0063]]，表單與固定來源沿 [[0065]]，GUI 與 remote 的共同 owner 入口沿 [[0067]]。本文描述新表單與 Engine 的分工，不承諾宿主整合或硬體驗收已完成。

## 位置與 owner

| Module | 擁有的行為 | 不擁有的行為 |
| --- | --- | --- |
| `zcu_tools.gui.form` | Input、FormState、FormEnv、FormErrors、Display、求值、驗證、input schema、保存與相容還原 | Qt、Engine revision、提交 policy、檔案位置、儀器與來源 publication |
| `zcu_tools.gui.widgets.form` | 按型別與 Display 產生 editor，修改 FormState，顯示值與錯誤 | 領域 defaults、to_cfg、Engine 控制與提交時機 |
| `zcu_lab.workflows.{coherence_fluxdep,mist_fluxdep,overnight}.form` 與 `common` 表單 helpers | 具體表單、defaults、choices、跨欄位 validators、明寫的 to_cfg | 通用 parser、schema walker、Engine lifecycle |
| `zcu_tools.experiment.workflows` | 普通 tunables 值的 revision、step capture、journal 與原子發布 | Expression、表單、Qt、library lookup |
| Workflow host | 捕捉 FormEnv，Start、Apply、Discard、revision 衝突、輸入保存與 RPC 入口 | 重做表單、Engine 或 storage owner 的工作 |

表單是 GUI 框架的 Qt-free 資料工具，不放入 cfg_model 或 utils。具體表單帶呈現資訊，屬於使用者 workflow 的 GUI 附件。Native workflow definitions 不反向 import GUI 附件。

組合根注入 workflow 與表單宣告。Framework 不 import zcu_lab 猜 registry，也不在 import 時註冊。MCP 經 GUI RPC 使用同一宿主入口，不持 Engine、Figure 或硬體連線。

## 輸入與普通值

FormInput 配對表單型別與可不完整的 JSON tree。缺 key 表示未填，null 表示明確的 null。只有文字欄位接受 `{"expr": "..."}`：以文字框輸入的 int、float、complex 與 str value leaves。以下拉或勾選呈現的欄位沒有 expression：`Literal`、Enum、宣告 choices 的 leaf、bool，以及 literal key（`KeyRef.name`，包括 `ModuleRef.name`）。這些值是挑出來的，不是算出來的。同時含文字成員與選項成員的 union（例如 `float | Literal["auto"]`、`bool | int`）沒有一種控制項能同時打字與挑選，core 的 input schema 與求值遇到時 raise ValueError，訊息帶欄位路徑。判斷在 core，只有一條規則：這些欄位的 input schema 沒有 Expr 分支，result 在該欄位報錯且不求值，compatible restore 丟棄它們的 envelope；editor 只依 schema 有沒有 Expr 分支決定給不給右鍵 Expression，不自己推論。Workflows 的 wire 版本涵蓋這個 expression 邊界與目前的 options／tagged input 契約。只有顯式 envelope 求值，普通字串包含 `$` 開頭的內容仍是 literal。Preview 與 result 使用共用 evaluator，再遵守 Pydantic 約束。Input schema 與 compatible restore 保留 expression 原文，editor 使用同一 schema 與 preview，不自行求值。Editor 不在欄位旁放模式下拉：文字欄位的右鍵選單在標準項目之後列出 Expression 切換；選項欄位的下拉只列選項，沒有右鍵選單；Expression 模式的輸入框以淺色底色區分。Complex 直接值使用 `{"__complex__": [re, im]}`。Model 實例只含求值後的普通值。

FormState 持有隔離的 input 與 base，不持 Engine revision。

- `write` 在候選 input 套用整批修改後一次安裝。Model-invalid 中間態留在 draft，result 回報錯誤，不以 defaults 蓋掉。
- Sequence 與 typed mapping 是 whole leaves，包含空 mapping。改名、刪 key 與清空都替換整個 mapping。普通 nested model 只寫 child leaves。
- Absent 非 nullable 子 model 的首次 child write 建立靜態 defaults，再寫明送內容。沒有預設的 required key 保持未填。
- Nullable 子 model 可寫 null 停用，或寫 object 啟用。停用後須先啟用才能編輯 child。
- `result(env)` 求值、檢查 choices，再交 Pydantic 驗證。成功回傳 model，失敗回傳 FormErrors。
- `preview(env)` 提供可求值 leaves 與 errors，供 invalid draft 的顯示使用。Preview 不代表可提交，也不回寫 input。
- `changes(env)` 比較 base 與 input，回傳 leaf path、input 和普通 value。Expression 改變而值相同仍算 input 變更。移除的分支欄位沒有 tombstone。
- `reset(base)` 使用宿主提供的新 base。宿主決定何時 Discard 或 Apply。

Write paths 是 model-field tuples，不進入 sequence index 或 mapping key。Mapping key 可含點。Host 將已確認的 leaf path 轉成 Engine dot path。FormErrors 保留 Pydantic 的 error path 與 sequence index；model-wide error 使用空 tuple。GUI 在表單頂端顯示 model-wide error，RPC 保留路徑與訊息。

Editor 預設直接呼叫 `FormState.write`。Host 需要在寫入前檢查或替換輸入時，傳入 write callable：editor 把每批 parsed write 交給它，自己不寫 state，照舊讀取並訂閱同一個 FormState。Host 以 `gui.form.WriteRejected(message)` 拒絕時，該欄保留使用者輸入並顯示 host 的訊息；其他例外照常拋出。這個接縫是一個 callable 參數，不是 Protocol、registry 或 writer 類別。

## Defaults、分支與 module reference

`defaults(env)` 是唯一 defaults hook，只依型別與固定來源，不接收 previous。`default_input(env)` 是 final 組裝入口，依序呼叫 `defaults(env)`、檢查來源、求值 expression defaults 並建立 FormInput。Defaults 可以包含 complex tags，也可以用 `{"expr": ...}` 宣告 scalar 初始值的來源，例如 `$flux_dev` 或 point path。Core 用同一個 env 當場求值，input 只存算好的 direct 值，畫面在 Direct 模式顯示這個值；畫面上的初始值就是會跑的值。來源之後改變，要重新建立 default input（Reset、Prepare）才反映。Expression 算不出來（缺 path、alias 未綁定、值不是數字），或算出的值在 result 的該欄位報錯時，欄位留空：required 欄位只顯示 `Field required`，有 Pydantic 靜態預設的欄位驗證時用那個預設。不另設初始值錯誤機制。Captured source 本身不可用的錯誤照常拋出。使用者或 MCP 寫入的 expression 保留原文。Required keys 沒有預設時保持未填。

Branch switch 先呼叫新分支的 fresh defaults，expression defaults 照 `default_input` 的規則算成值。它不繼承舊分支的同名欄位，明送內容最後覆蓋。Discriminator 使用新分支的普通 typed tag。它是結構 selector，決定子 model 的形狀與 branch defaults，不是延後求值的 value leaf。Core schema 不提供 expression envelope，editor 不提供 expression mode，字串與 numeric selector 遵守同一規則。沒有 branch factory registry。

只有一個合法值的 `Literal` 沒有可選或可算的內容。Editor 把它畫成唯讀文字，沒有下拉與 expression 模式；core schema 照常輸出 `const`，沒有 Expr 分支。Model 不經 union selector 到達時，例如單一 model 欄位或 nullable model，它身為 tagged union branch 的 tag 只剩一個值，是固定的結構 tag。Schema 把這個 tag 標成 `hidden`，editor 不畫它，值照常留在 input 並驗證。Branch 身分來自本表單、ModuleInput 可達的 tagged union，也涵蓋 branch 的子類別，例如單獨使用的 `LinRange.kind`，以及 inline readout 的 `ro_cfg.type`。在 union 位置的 tag 照舊是 selector。

核心在 `default_input`、`input_schema` 與 FormInput 入口全樹拒絕未標 tag 的 `str | Model`，包含 nullable、alias、containers 與 inactive branches。錯誤指名欄位並要求 tagged ref／inline。持久 Spec 仍可保存 child key 字串，生成器在 input 邊界加 wrapper，轉回持久 Spec 時去掉 wrapper。核心不豁免 Spec，也不依賴生成器。

Compatible restore 的 model-only union 仍只保留恰好一個相容 member，ambiguous 或 unmatched subtree 丟棄。Tagged restore 改 branch 時呼叫同一個 fresh defaults。舊 ref 複製 hook 已退休，複製只由明確選項指令提供。

T1TunablesForm 在 captured point 有 `general.ext.guess_us` leaf 時，以這個 path 的 expression 宣告 lifetime 初始值，開表單時 core 把它算成值。這個 leaf 是 null 或不是正數時算不出合法值，guess_us 留空；guess_us 沒有靜態預設，所以 result 只報它的 `Field required`，不默默落回 20 µs。缺 leaf 或沒有 capture 時，defaults 明確給 20 µs 的普通值。

`choices(env)` 提供 model-field leaf 的合法值。最深 nested FormModel 的 local choices 優先，包含空選項。Input schema 與 result 共用這些選項。Display 只描述標籤、說明、單位與呈現，順序沿欄位宣告，段落沿 nested model，畫面群組沿 Display 的 `group`。

Display 的 `decimals` 是 float 欄位顯示的小數位數。Editor 四捨五入並去掉尾端的 0，不改 input；非 float 欄位宣告它時 input schema 報錯。`hidden` 讓 editor 不畫欄位：值與預設留在 input，照常驗證，錯誤落在最近的已畫祖先列或表單頂端。Input schema 列出 hidden 欄位並標 `hidden`。MCP 的 `workflow.form` schema 照樣列出它們與 hidden 標記，不過濾，所以 schema 與 model 一致；agent 把 hidden 欄位讀成固定值，不需要填。Schema 的 `decimals` 只出現在宣告它的欄位，`hidden` 只出現在宣告的欄位與固定的結構 tag，其他表單的 schema 不多出這兩個 key。巢狀 model 欄位沒有 Display label 時，段落標題是欄位名：底線換成空白，只有第一個字母大寫，例如 `ro_cfg` 是 `Ro cfg`，`ModuleInline.module` 是 `Module`。Pydantic 帶進來的類別名不當標題。

Display 的 `group` 把欄位歸到同一個 model 層級內的具名群組，例如 `group="Advanced"`；空字串在建構 Display 時報錯。群組只影響呈現，不改驗證、defaults、choices、寫入路徑或 FormValue。Input schema 只在宣告群組的 property 節點標 `x-form-group`，值是群組名稱；未宣告的欄位與表單的 schema 不變，wire 版本不變。Editor 在每個 model 層級先畫未分組欄位，再依群組第一次出現的順序畫群組區塊，區塊內照欄位宣告順序。Nested model 欄位本身可以是群組成員。群組預設折疊；所有成員都 hidden 時不畫區塊。使用者在同一個 editor 裡展開或折疊的群組與 model 以 path 記住，換分支、在下拉切換 key 與 Inline 等重建後保持原狀；沒動過的 model 照 `Display(collapsed=...)`。折疊中的群組有欄位錯誤時，標題顯示有錯誤，不自動展開，避免使用者看不到錯誤而卡在 Start。MCP 的 `workflow.form` 說明告訴 agent：`x-form-group` 只影響畫面分組，組內欄位一樣要填，必填與否看 schema。

Host 不允許修改 input 的期間呼叫 editor 的 `set_editing_enabled(enabled=False)`，例如 workflows 的 run 從 Start 到 terminal delivery。表單變成唯讀但可讀：文字欄位可選取與複製；下拉與勾選框顯示目前值，不回應點擊、滾輪與鍵盤；群組與 model 照常展開、折疊與捲動；右鍵選單只留複製類項目，Expression 停用。控制項不停用，所以不變灰。外部 state 更新照常顯示，`enabled=True` 恢復全部編輯。Host 不再停用整個面板，因為停用會讓使用者看不到折疊起來的值。

Module ref 與 inline 使用 tagged 子 model。Ref 保存 literal library key，inline 保存生成的支援型別。普通分支切換只用 fresh defaults；`to_inline` 指令才從固定 library 複製 Spec，複製後不跟隨 library 改動。具體 to_cfg 使用公開 factory 與固定 validation context。缺 key、錯 family、materialization 與 asset 錯誤不降級為 default。

掃描輸入只使用 LinRange。具體 to_cfg 才展開 setpoints。Delay grid 由 workflow 計算。固定值在具體 to_cfg 或 runtime 明寫，不新增通用 sweep 連動、inheritance、`$` capture、refresh、live linking 或 Unavailable publication。

## 選項清單與分層

2026-10-13 使用者裁決。推導見 [討論紀錄](../../discussions/2026-10-13-form-options-layering.md)。

### 分工

Form 生成器（`gui.form_adapter`）把實驗語義翻成顯示偏好。使用者：「這些資訊應該是由Form生成器負責將帶有實驗語義轉變成有顯示偏好的格式」。表單核心（`gui.form`）只提供通用機制，不知道 module 或 library。Widget 與 MCP 照 input schema 的清單畫、照清單寫，不解讀 union 結構。使用者：「FormInput/FormModel與Widget間是簡單的關係……在不同widget/app中使用，不一定是為了Module/Library」。

### 選項

- 每個挑選欄位在 input schema 有一份有效選項清單 `x-form-options`。每項含 `label`、`value`，可編輯物件值可含 `fields`：內嵌的子表單 schema，不含已由選項決定的 tag。
- `FormModel.options(env)` 是宣告 hook，預設為空。核心把宣告和自動選項合併成有效清單。自動選項涵蓋 Literal、Enum、`choices(env)` 與 tagged union 的成員。
- 生成器為 module 與波形欄位宣告以下選項：可留空時的 `None`、kind 符合的 library keys、每個 inline kind 一項 `Inline: <kind>`。只有一個 kind 時顯示 `Inline`，沒有欄位的 kind（例如 `reset/none`）不列。Key 只顯示名稱。
- 比對規則只有一條：目前 input 在某個選項 `value` 的每個葉都具有相同 JSON 型別與值，就選中它，依清單順序取第一個。沒有選項符合時顯示目前值加 `(missing)`，不改 state。
- 可留空的 key-or-inline 欄位把 `None` 放進下拉，不畫 Enabled 勾選框。一般可留空的 model 區塊仍用 Enabled 勾選框。

### 換分支

在 union 位置寫入 partial object 就是選分支。新分支用預設值，也就是範本或型別預設，不繼承舊分支的同名欄位，defaults 也不看切換前的值。寫 tag 葉換分支也遵守同一規則。使用者在比較時選「改 kind 時，舊 kind 填過的值不帶過去」。形狀與選項只看型別和捕捉的來源，初值只看型別、範本與來源，因此不需要帶狀態的 factory hook。

### 選項指令

選項可以帶 `commands: [{name, label}]`。Widget 在右鍵選單列出目前選中選項的指令，按下就經 host callback 呼叫 `FormState.run_command(path, name, env)`，結果以一次寫入提交。生成器在每個 key 選項上宣告 `to_inline`（`Convert to inline`）並提供實作：用捕捉的 library 把該 key 的內容複製成 inline，複製後和 library 斷開。呼叫者可以從目前選中的選項知道有哪些指令可用。`run_command` 遇到目前選項沒有帶的指令會報錯，不會靜默略過。Recipe 的 `use_inline` RPC 先讀公開 S2 input tag，不求值或驗證 draft：是 key 才呼叫 `to_inline`，已經是 inline 就直接寫。`use_inline` 包裝這個判斷，對已是 inline 的欄位維持不變。使用者：「recipe需要一種方法知道目前欄位是ref還是inline，進而決定要不要先轉inline」。使用者提議：「對ref下拉按鈕右鍵有轉成inline的選項」；後續追問：「widget層需要有ref/inline的認知嗎？」，答案是不需要。

### 一種 key-or-inline 表示法

生成器把未標 tag 的 `str | Model`（`waveform`、`pulse_cfg`、`pulse1_cfg`、`pulse2_cfg`、`raise_waveform` 等）換成有 tag 的 ref／inline 形狀，`to_cfg` 換回持久 Spec。核心不再支援未標 tag 的 key-or-inline，手寫這種表單時在入口報錯。Ref 選中時下拉只顯示名稱，不顯示被引用 entry 的內容。使用者：「如果實作困難ref就不用支援只讀顯示」、「只顯示選單名就可以」。

### Consumer 與版本邊界

Widget 與 MCP 只讀選項的 value、fields 與 commands，不從 union 猜 branch 或 library。MCP 的 `RecipeFormField.fields` 只列選中項目的可編輯子欄位，不含 tag；ref 沒有 fields。Mode 讀公開 input kind，inline 的顯式 path 仍是 `<field>.module.<leaf>`，tag 葉不提供 `read_field`。Measure 的 editor 成對傳入 resource guarded write／run_command；workflows 不另注入 callback。

Measure wire 102、workflows wire 4 分別涵蓋 options 與 tagged input。Measure memento 3、workflows memento 2 沿各自既有 caretaker／codec 拒絕舊版分頁，沒有 migration 或雙讀。

## 來源與既有工具

FormEnv 借用 host 捕捉的 ParameterContext、LibrarySnapshot，以及裝置名稱到 detached `BaseDeviceInfo` snapshot 的 mapping（含 custom driver 的 info）。共用工具只讀裝置名稱，需要特定 driver 欄位的表單自己以 `isinstance` 收窄。ParameterContext owner 在 capture 邊界隔離 point 與 roles，表單不再次複製，也不轉成 flat MetaDict。純數學 expression 與普通 input 不需要 parameter capture。來源缺失的錯誤落在消費欄位。

Host 決定 capture 與重算時機。每次 result 和 changes 使用傳入的 env。工具不 load YAML、不 refresh、不 poll device；frozen dataclass 不承諾凍結 driver 或 waveform asset bytes。

Expression 重用 `gui.session.expression.evaluate_scalar_expr` 的型別、白名單與限制，不建立另一個 parser。Complex 直接重用 `core.complex_codec.encode_complex` 與 `decode_complex`。Editor 的共用 value-widget helpers 由 `gui.widgets.form.value_widgets` 擁有，並重用 `gui.widgets.spinbox`。舊 cfg widgets 與 codec owner 已移除。

## Engine 只記普通值

Engine.start 保存 validated plan 與 tunables。Update journal 保存 leaf path、舊值、新值、actor、時間與 revision。Journal append 成功後才發布新值。Stale revision、model error 與 append failure 不發布候選值。當前 step 保留自己的 capture；後續 step 或 resume 取最新 revision。

Engine 不保存 expression 原文，也不求值。Input JSON 的 expression 留在 FormInput，輸入保存由 host 負責。這避免 Engine 同時維護值與另一份 input 狀態，代價是 Engine journal 不能重建當時的 expression。

## 依賴與驗證

`.importlinter` 的 `form-core-no-ui-policy` 禁止 core 依賴 app、widgets、remote、session services/UI/adapters、experiment 與 Qt。`form-editor-no-host` 禁止 editor 依賴 app、remote、experiment 與 MCP。C18 保護 Engine 的 GUI independence，C16 禁止 framework import zcu_lab，C13 限制 MCP 的 remote 路徑。

Core 到 gui.session.expression、core.complex_codec、resources.context、cfg_model 與 device 型別的依賴沿 C8/C9 的 frontend 方向。C11 保護 cfg_model 不反向依賴 GUI。具體 GUI 附件可以消費 framework 與 native workflow models。現有 C15 只保護 `zcu_lab.v2.**.core`，不涵蓋 `zcu_lab.workflows`。Native workflow definitions 不依賴 GUI 的規則目前以 source review 確認，尚無專屬 contract。

公開 FormState、input schema、保存還原、Engine start/update/execute 與 journal 驗證資料契約。Qt editor 測試觀察 editor 到 FormState 的行為。具體表單測試觀察 defaults、result 與 native conversion。文件、設定、依賴宣告與 Qt-free 性質由直接 source review 及 import contracts 確認，不以 tests 另啟程序。

## 取捨

各宿主共用 Form input、evaluator、complex codec 與 FormEditor，不保留 CfgSchema 的第二套編輯模型。領域 defaults 和 to_cfg 留在生成器或具體表單。Core 不承擔提交 policy，host 決定 Apply 時機。
