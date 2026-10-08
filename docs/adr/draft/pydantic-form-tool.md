---
status: draft
---

# Pydantic 表單、輸入與宿主提交

## 脈絡與範圍

Workflow plan 與 tunables 需要 sequence、nullable、tagged union、typed mapping、expression 和 library module reference。新表單用 Pydantic 表示普通值，用獨立的 input JSON 保存未填與 expression。它不擴充 CfgSchema，也不遷移 Measure。

保存與來源責任沿 [[0063]]，既有 cfg 機制沿 [[0065]]，GUI 與 remote 的共同 owner 入口沿 [[0067]]。本文描述新表單與 Engine 的分工，不承諾宿主整合或硬體驗收已完成。

## 位置與 owner

| Module | 擁有的行為 | 不擁有的行為 |
| --- | --- | --- |
| `zcu_tools.gui.form` | Input、FormState、FormEnv、FormErrors、Display、求值、驗證、input schema、保存與相容還原 | Qt、Engine revision、提交 policy、檔案位置、儀器與來源 publication |
| `zcu_tools.gui.widgets.form` | 按型別與 Display 產生 editor，修改 FormState，顯示值與錯誤 | 領域 defaults、to_cfg、Engine 控制與提交時機 |
| `zcu_lab.workflows.gui` | 具體表單、defaults、choices、跨欄位 validators、明寫的 to_cfg | 通用 parser、schema walker、Engine lifecycle |
| `zcu_tools.experiment.workflows` | 普通 tunables 值的 revision、step capture、journal 與原子發布 | Expression、表單、Qt、library lookup |
| Workflow host | 捕捉 FormEnv，Start、Apply、Discard、revision 衝突、輸入保存與 RPC 入口 | 重做表單、Engine 或 storage owner 的工作 |

表單是 GUI 框架的 Qt-free 資料工具，不放入 cfg_model 或 utils。具體表單帶呈現資訊，屬於使用者 workflow 的 GUI 附件。Native workflow definitions 不反向 import GUI 附件。

組合根注入 workflow 與表單宣告。Framework 不 import zcu_lab 猜 registry，也不在 import 時註冊。MCP 經 GUI RPC 使用同一宿主入口，不持 Engine、Figure 或硬體連線。

## 輸入與普通值

FormInput 配對表單型別與可不完整的 JSON tree。缺 key 表示未填，null 表示明確的 null。Numeric 與 complex 欄位接受 `{"expr": "..."}`。普通字串仍是普通字串，不支援字串求值。Complex 直接值使用 `{"__complex__": [re, im]}`。Model 實例只含求值後的普通值。

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

## Defaults、分支與 module reference

`defaults(env, previous)` 是唯一 defaults hook。`default_input(env)` 是 final 組裝入口，依序呼叫 `defaults(env, {})`、檢查來源並建立 FormInput。Defaults 可以包含 expression 與 complex tags。Required keys 沒有預設時保持未填。

Branch switch 先呼叫新分支的 defaults，previous 是舊分支的隔離 input。接著沿用同名欄位，明送內容最後覆蓋。Discriminator 使用新分支的值。沒有 branch factory registry。

T1TunablesForm 只在 previous 為空時預填 lifetime expression。Captured point 的 `general.ext.guess_us` 存在時保留引用，即使它是 null 或無效值。缺 leaf 或沒有 capture 時使用暫定普通值。無效引用在 guess_us 回報錯誤，不降級為 fallback。

`choices(env)` 提供 model-field leaf 的合法值。最深 nested FormModel 的 local choices 優先，包含空選項。Input schema 與 result 共用這些選項。Display 只描述標籤、說明、單位與呈現，順序與分組沿欄位宣告及 nested model。

Module ref 與 inline 使用 tagged 子 model。Ref 保存 library key，inline 保存 ModuleSpec。Ref 切 inline 從固定 library 讀取 Spec，切換後不跟隨 library 改動。Defaults 與 choices 使用 LibrarySnapshot 的公開讀取；具體 to_cfg 使用公開 factory 與固定 validation context。缺 key、錯 family、materialization 與 asset 錯誤不降級為 default。

LinRange 與 ExplicitList 保存兩種 flux 輸入。具體 to_cfg 才展開 setpoints。Delay grid 由 workflow 計算。固定值在具體 to_cfg 或 runtime 明寫，不新增通用 sweep 連動、inheritance、`$` capture、refresh、live linking 或 Unavailable publication。

## 來源與既有工具

FormEnv 借用 host 捕捉的 ParameterContext、LibrarySnapshot 與 native DeviceInfo mapping。ParameterContext owner 在 capture 邊界隔離 point 與 roles，表單不再次複製，也不轉成 flat MetaDict。純數學 expression 與普通 input 不需要 parameter capture。來源缺失的錯誤落在消費欄位。

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
