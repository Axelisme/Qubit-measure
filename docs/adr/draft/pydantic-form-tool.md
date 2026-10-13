---
status: draft
---

# Pydantic 表單、輸入與宿主提交

**Last updated:** 2026-10-13，引用或 inline 合併成一個不可編輯的下拉，`ModuleInput` 也帶 `x-form-key-or-inline`；defaults 的 expression 在 `default_input` 與分支 defaults 中算成值；editor 以右鍵選單切換 Direct／Expression；Display 的 group、`x-form-group` 標記與 editor 的可折疊群組區塊；editor 的 host write callable 與 WriteRejected；Display 的 decimals／hidden、單值 Literal、固定的結構 tag 與段落標題；MCP 照列 hidden 欄位；str-or-model 欄位的 `x-form-key-or-inline` 標記；FormEnv 的 devices 是 detached BaseDeviceInfo snapshots

## 脈絡與範圍

Workflow plan 與 tunables 需要 sequence、nullable、tagged union、typed mapping、expression 和 library module reference。新表單用 Pydantic 表示普通值，用獨立的 input JSON 保存未填與 expression。它不擴充 CfgSchema，也不遷移 Measure。

保存與來源責任沿 [[0063]]，既有 cfg 機制沿 [[0065]]，GUI 與 remote 的共同 owner 入口沿 [[0067]]。本文描述新表單與 Engine 的分工，不承諾宿主整合或硬體驗收已完成。

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

FormInput 配對表單型別與可不完整的 JSON tree。缺 key 表示未填，null 表示明確的 null。Numeric、complex 與所有字串 value leaves 接受 `{"expr": "..."}`，包含 `Literal[str]` 與 str Enum。只有顯式 envelope 求值，普通字串包含 `$` 開頭的內容仍是 literal。Preview 與 result 使用共用 evaluator，再遵守 choices 與 Pydantic 約束。Input schema 與 compatible restore 保留 expression 原文，editor 使用同一 schema 與 preview，不自行求值。Editor 不在欄位旁放模式下拉：輸入元件的右鍵選單在文字欄位的標準項目之後列出 Expression 切換，選項欄位的下拉只列選項，Expression 模式的輸入框以淺色底色區分。Complex 直接值使用 `{"__complex__": [re, im]}`。Model 實例只含求值後的普通值。

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

`defaults(env, previous)` 是唯一 defaults hook。`default_input(env)` 是 final 組裝入口，依序呼叫 `defaults(env, {})`、檢查來源、求值 expression defaults 並建立 FormInput。Defaults 可以包含 complex tags，也可以用 `{"expr": ...}` 宣告 scalar 初始值的來源，例如 `$flux_dev` 或 point path。Core 用同一個 env 當場求值，input 只存算好的 direct 值，畫面在 Direct 模式顯示這個值；畫面上的初始值就是會跑的值。來源之後改變，要重新建立 default input（Reset、Prepare）才反映。Expression 算不出來（缺 path、alias 未綁定、值不是數字），或算出的值在 result 的該欄位報錯時，欄位留空：required 欄位只顯示 `Field required`，有 Pydantic 靜態預設的欄位驗證時用那個預設。不另設初始值錯誤機制。Captured source 本身不可用的錯誤照常拋出。使用者或 MCP 寫入的 expression 保留原文。Required keys 沒有預設時保持未填。

Branch switch 先呼叫新分支的 defaults，previous 是舊分支的隔離 input，expression defaults 照 `default_input` 的規則算成值。接著沿用同名欄位（使用者寫的 expression 保留原文），明送內容最後覆蓋。Discriminator 使用新分支的普通 typed tag。它是結構 selector，決定子 model 的形狀與 branch defaults，不是延後求值的 value leaf。Core schema 不提供 expression envelope，editor 不提供 expression mode，字串與 numeric selector 遵守同一規則。沒有 branch factory registry。

只有一個合法值的 `Literal` 沒有可選或可算的內容。Editor 把它畫成唯讀文字，沒有下拉與 expression 模式；core schema 不因此改變。Model 不經 union selector 到達時，例如單一 model 欄位、nullable model 或 `str | Model` 的單一 model，它身為 tagged union branch 的 tag 只剩一個值，是固定的結構 tag。Schema 把這個 tag 標成 `hidden`，editor 不畫它，值照常留在 input 並驗證。Branch 身分來自本表單、ModuleInput 與 RangeInput 可達的 tagged union，也涵蓋 branch 的子類別，例如單獨使用的 `LinRange.kind`，以及 inline readout 的 `ro_cfg.type`。在 union 位置的 tag 照舊是 selector。

未標籤的 `str | Model` union 以 JSON 型別（字串／物件）當結構 selector：字串是 library key，物件是 inline model，inline 物件再依自己的 tag 選 branch。Key 不接受 expression envelope，因為 envelope 是物件。Core 在這類欄位的 schema 節點標 `x-form-key-or-inline: true`，`ModuleInput` 欄位也標，editor 只依這個標記決定要不要合併下拉，不自己從 schema 形狀判斷，所以規則只在 core。Editor 對這類欄位只畫一個不可編輯的下拉：第一項 Inline，其後是 schema 的 `x-form-library-keys`。`str | Model` 的清單是 Spec 型別相符的 keys；`ModuleInput` 的清單在 ref branch 的 `name`，是全部 library module keys，因為 inline 的 `ModuleSpec` 接受所有 module 型別；Form 用 `choices()` 為該 name 宣告選項時取交集。選 key 寫入引用（`ModuleInput` 一批寫 `kind` 與 `name`），選 Inline 寫入 inline 預設值，不轉換另一邊的值。目前值是清單沒有的 key 時，下拉顯示 `<key> (missing)` 並選中它，不改寫 state，錯誤照常顯示。GUI 下拉不可打字，wire 照舊可寫任意 key；清單只是 GUI 可選的項目，不限制輸入。

T1TunablesForm 只在 previous 為空、captured point 有 `general.ext.guess_us` leaf 時，以這個 path 的 expression 宣告 lifetime 初始值，開表單時 core 把它算成值。這個 leaf 是 null 或不是正數時算不出合法值，guess_us 留空；guess_us 沒有靜態預設，所以 result 只報它的 `Field required`，不默默落回 20 µs。缺 leaf 或沒有 capture 時，defaults 明確給 20 µs 的普通值。

`choices(env)` 提供 model-field leaf 的合法值。最深 nested FormModel 的 local choices 優先，包含空選項。Input schema 與 result 共用這些選項。Display 只描述標籤、說明、單位與呈現，順序沿欄位宣告，段落沿 nested model，畫面群組沿 Display 的 `group`。

Display 的 `decimals` 是 float 欄位顯示的小數位數。Editor 四捨五入並去掉尾端的 0，不改 input；非 float 欄位宣告它時 input schema 報錯。`hidden` 讓 editor 不畫欄位：值與預設留在 input，照常驗證，錯誤落在最近的已畫祖先列或表單頂端。Input schema 列出 hidden 欄位並標 `hidden`。MCP 的 `workflow.form` schema 照樣列出它們與 hidden 標記，不過濾，所以 schema 與 model 一致；agent 把 hidden 欄位讀成固定值，不需要填。Schema 的 `decimals` 只出現在宣告它的欄位，`hidden` 只出現在宣告的欄位與固定的結構 tag，其他表單的 schema 不多出這兩個 key。巢狀 model 欄位沒有 Display label 時，段落標題是欄位名：底線換成空白，只有第一個字母大寫，例如 `ro_cfg` 是 `Ro cfg`，`ModuleInline.module` 是 `Module`。Pydantic 帶進來的類別名不當標題。

Display 的 `group` 把欄位歸到同一個 model 層級內的具名群組，例如 `group="Advanced"`；空字串在建構 Display 時報錯。群組只影響呈現，不改驗證、defaults、choices、寫入路徑或 FormValue。Input schema 只在宣告群組的 property 節點標 `x-form-group`，值是群組名稱；未宣告的欄位與表單的 schema 不變，wire 版本不變。Editor 在每個 model 層級先畫未分組欄位，再依群組第一次出現的順序畫群組區塊，區塊內照欄位宣告順序。Nested model 欄位本身可以是群組成員。群組預設折疊；所有成員都 hidden 時不畫區塊。使用者在同一個 editor 裡展開或折疊的群組與 model 以 path 記住，換分支、在下拉切換 key 與 Inline 等重建後保持原狀；沒動過的 model 照 `Display(collapsed=...)`。折疊中的群組有欄位錯誤時，標題顯示有錯誤，不自動展開，避免使用者看不到錯誤而卡在 Start。MCP 的 `workflow.form` 說明告訴 agent：`x-form-group` 只影響畫面分組，組內欄位一樣要填，必填與否看 schema。

Module ref 與 inline 使用 tagged 子 model。Ref 保存 library key，inline 保存 ModuleSpec。Ref 切 inline 從固定 library 讀取 Spec，切換後不跟隨 library 改動。Defaults 與 choices 使用 LibrarySnapshot 的公開讀取；具體 to_cfg 使用公開 factory 與固定 validation context。缺 key、錯 family、materialization 與 asset 錯誤不降級為 default。

LinRange 與 ExplicitList 保存兩種 flux 輸入。具體 to_cfg 才展開 setpoints。Delay grid 由 workflow 計算。固定值在具體 to_cfg 或 runtime 明寫，不新增通用 sweep 連動、inheritance、`$` capture、refresh、live linking 或 Unavailable publication。

## 來源與既有工具

FormEnv 借用 host 捕捉的 ParameterContext、LibrarySnapshot，以及裝置名稱到 detached `BaseDeviceInfo` snapshot 的 mapping（含 custom driver 的 info）。共用工具只讀裝置名稱，需要特定 driver 欄位的表單自己以 `isinstance` 收窄。ParameterContext owner 在 capture 邊界隔離 point 與 roles，表單不再次複製，也不轉成 flat MetaDict。純數學 expression 與普通 input 不需要 parameter capture。來源缺失的錯誤落在消費欄位。

Host 決定 capture 與重算時機。每次 result 和 changes 使用傳入的 env。工具不 load YAML、不 refresh、不 poll device；frozen dataclass 不承諾凍結 driver 或 waveform asset bytes。

Expression 重用 `gui.session.expression.evaluate_scalar_expr` 的型別、白名單與限制，不建立另一個 parser。Complex 重用 `gui.cfg.codec.encode_complex` 與 `decode_complex`。Editor 重用 `gui.widgets.cfg.fields.common` 的公開 value-widget 入口及 `gui.widgets.spinbox`，不為借用版面而建立 ScalarSpec 或 CfgDraft。

## Engine 只記普通值

Engine.start 保存 validated plan 與 tunables。Update journal 保存 leaf path、舊值、新值、actor、時間與 revision。Journal append 成功後才發布新值。Stale revision、model error 與 append failure 不發布候選值。當前 step 保留自己的 capture；後續 step 或 resume 取最新 revision。

Engine 不保存 expression 原文，也不求值。Input JSON 的 expression 留在 FormInput，輸入保存由 host 負責。這避免 Engine 同時維護值與另一份 input 狀態，代價是 Engine journal 不能重建當時的 expression。

## 依賴與驗證

`.importlinter` 的 `form-core-no-ui-policy` 禁止 core 依賴 app、widgets、remote、session services/UI/adapters、experiment 與 Qt。`form-editor-no-host` 禁止 editor 依賴 app、remote、experiment 與 MCP。C18 保護 Engine 的 GUI independence，C16 禁止 framework import zcu_lab，C13 限制 MCP 的 remote 路徑。

Core 到 gui.session.expression、gui.cfg.codec、resources.context、cfg_model 與 device 型別的依賴沿 C8/C9 的 frontend 方向。C11 保護 cfg_model 不反向依賴 GUI。具體 GUI 附件可以消費 framework 與 native workflow models。現有 C15 只保護 `zcu_lab.v2.**.core`，不涵蓋 `zcu_lab.workflows`。Native workflow definitions 不依賴 GUI 的規則目前以 source review 確認，尚無專屬 contract。

公開 FormState、input schema、保存還原、Engine start/update/execute 與 journal 驗證資料契約。Qt editor 測試觀察 editor 到 FormState 的行為。具體表單測試觀察 defaults、result 與 native conversion。文件、設定、依賴宣告與 Qt-free 性質由直接 source review 及 import contracts 確認，不以 tests 另啟程序。

## 取捨

新表單和 CfgSchema 暫時並存，增加兩種 input 模型，但共用 evaluator、complex codec 與 value widgets。領域 defaults 和 to_cfg 留在具體表單。Core 不承擔提交 policy，host 可以共用資料機制而自行決定 Apply 時機。
