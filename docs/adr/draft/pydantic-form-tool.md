---
status: draft
---

# Pydantic 表單、輸入原文與宿主提交

## 脈絡與狀態

P2 的 plan 與 knobs 需要 sequence、nullable、tagged union、typed mapping、expression 和 library module ref。舊 CfgSchema 的 Spec／Value tree 不直接表示這些輸入。擴充舊工具會把本次 P2 與 Measure 遷移綁在一起。

使用者已依 storage-redesign D195／D197 定稿新工具設計。本稿把定稿的「設計總覽」轉為 package 位置、依賴與 Engine 擴充安排，不重設計互動。Sr-advisor 已按 D198 代裁核准 F0。F2 的 Engine 原文機制已 collect 並通過完整 range review。F1 正在準備公開 declarations、真實 T1TunablesForm caller 與正式 tests；這不表示 core 內部實作或 GUI host 已完成。

定稿來源是 task-local `storage-redesign/spec/form-tool-design.md`。該檔持有 U1–U7、A1–A4、F0–F5 與完整 CfgSchema 能力對應。工具適用於新表單，本次不碰 CfgSchema，不遷移 Measure。舊表單退休由 backlog BL-20261007T155403537614Z-cfgschema-pydantic 另行安排。

保存與來源責任沿 [[0063]]，舊 cfg 機制沿 [[0065]]，GUI／remote 共用 owner 入口沿 [[0067]]。本稿只補新表單與 Engine 原文擴充的分工。

## 位置與 owner

| Module | 位置 | 擁有的行為 | 不擁有的行為 |
| --- | --- | --- | --- |
| Qt-free 表單工具 | `zcu_tools.gui.form` | Input、FormState、FormEnv、FormErrors、Display、子 model、求值與驗證、input schema、輸入保存／相容還原 | Engine、owner revision、Apply、檔案位置、Qt、儀器或 live context publication |
| Qt editor | `zcu_tools.gui.widgets.form` | 按表單類別與 Display 產生元件，編輯 FormState，呈現結果與欄位錯誤 | Defaults 的領域規則、to_cfg、Engine、MCP、提交時機 |
| T1 對外表單 | `zcu_lab.workflows.gui` | T1PlanForm、T1TunablesForm、default_input、choices、跨欄位 validators、明寫的 to_cfg | 通用 parser、schema walker、editor、Engine 控制 |
| Workflow 執行核心 | `zcu_tools.experiment.workflows` | Tunables 值與 opaque 原文的 revision、capture、journal 與原子發布 | Expression 求值、表單、Qt、library lookup |
| Workflow app | `zcu_tools.gui.app.workflows` | 建立 FormEnv，Start、Apply／Discard、revision 衝突、上次輸入保存、將 RPC 導入同一入口 | 重做表單／Engine／storage owner 的工作 |

`gui.form` 是 GUI 框架的 Qt-free 資料與編輯工具，符合 module-boundaries 的 `gui` 定位，不放入 cfg_model 或 utils。T1 表單帶呈現資訊，屬於具體 workflow 的 GUI 附件，不放入 framework，也不從 runtime workflow 定義反向 import。

GUI／外部組合根顯式注入 workflow 與表單宣告。Framework 不 import zcu_lab 來猜 registry，不在 import 時註冊。MCP 只透過 GUI RPC 使用 app 的公開入口，不持 Engine、Figure 或硬體連線。

## 輸入、值與編輯狀態

Input 是可不完整的 JSON tree。缺 key 表示未填，null 表示明確的 null。Scalar 可直接填值。Numeric／complex 欄位也可填 `{"expr": "..."}`，沿用數值 evaluator，不支援字串求值。Complex 直接值使用 `{"__complex__": [re, im]}`。Discriminator 決定 tagged-union 分支。表單 model 的實例只含求值後的普通值。

FormInput[F] 把輸入與表單型別 F 配對。其內容以具名 JSON 型別表示，不把 BaseModel、driver、array 或 callable 塞入持久輸入。FormState[F] 持有隔離的 input 與 base，不持 Engine revision。Editor 與 host 的 MCP 請求使用同一套資料行為，但 MCP 不寫某個畫面的私有草稿。

- `write` 將一批輸入寫入候選後一次安裝。Model-invalid 中間態仍留在 draft，result 回報錯誤，不以預設值掩蓋它。Typed mapping 與 sequence 都是完整 leaf，刪 key、改名與清空都寫入整個 mapping。普通 nested model 只寫 child leaves，不接受整個替換。Absent 非 nullable 子 model 的首次 child write 先建立靜態 defaults，再寫明送欄位；沒有 defaults 的 required key 仍未填。Nullable 子 model 可寫 null 停用，或寫 object 啟用；明確 null 後必須先啟用才能編輯子欄位。
- `result(env)` 解碼 scalar、求值 expression、檢查 choices，最後呼叫表單 model 驗證。成功回傳合法 model，失敗回傳 FormErrors。
- `changes(env)` 回傳相對 base 的改動葉節點與其原文、求值後值。List／tuple／typed mapping 是一個完整葉節點，包含空 mapping，不產生 index 或 mapping-key patch。
- `reset(base)` 以 host 提供的新 base 重設草稿。Discard 與成功 Apply 都由 host 決定何時呼叫，不是工具 policy。
- 變更通知使用普通 callback，不使用 Qt Signal，也不另設 owner Protocol、auto_apply 選項或發布服務。

Write paths 是 model-field tuple，不進入 sequence index 或 mapping key。Mapping 內的 key 可含點，不影響整個欄位的寫入路徑。Host 把已確認的 model-field leaf path 轉成 Engine dot path；mapping leaf 使用該欄位路徑。Engine 仍不接受 sequence index 或整個 nested model replacement。FormErrors 以欄位路徑 tuple 對應訊息，list 驗證錯誤保留 Pydantic 的 index 位置。沒有經 `field_error` 指定欄位的 model-wide error 使用空 tuple `()`。GUI 將這類錯誤顯示在表單頂端，MCP 原樣回傳路徑與訊息，不把錯誤塞到某個欄位。

表單類別使用 Annotated 的 Display 提供標籤、說明、單位與輸入提示。順序依欄位宣告，分組依 nested 子 model；Display 不另設 order／group。collapsed 只用於子 model 欄位。動態合法值由 `choices(env)` 提供，驗證與 input schema 共用結果。Choices 路徑只指向 model-field leaf，可進入 nullable／tagged-union 子 model，不進入 container key 或 index。同型 nested model 在不同欄位的選項彼此隔離。Numeric／complex leaf 的 input schema 允許 well-formed arithmetic Expr，result 在求值後檢查 membership。`field_error(path, message)` 讓 model_validator 指向具體欄位。Pydantic 擁有型別與 Field 範圍，工具不維護第二套 model 合法性規則。

## Defaults、分支、range 與 module ref

頂層表單類別以 `default_input(env)` 明寫預設值，可包含 expression。部分 MCP 輸入與還原都先建立 defaults，再寫入一批，沒有另一套合併來源政策。

`FormModel.input_schema(env)` 是不需建立 FormState 的 classmethod，描述輸入而非僅描述合法 model，包含 expression／complex tag、Display 說明與當次 choices。FormState 的同名入口只委派。保存使用輸入 JSON。`write_compatible` 只沿用仍存在且型別相容的葉節點，回報丟棄的路徑；保留的跨欄位錯誤仍可見，不以 defaults 蓋掉錯誤。保存位置與時機歸 host，P2 只保存一份上次輸入。

Batch write 先套用 discriminator 改動，再寫其餘 leaf，明送內容優先。切換分支時，先呼叫新分支的 `defaults(env, previous)`。沒有此方法時使用靜態預設，仍缺的欄位保持未填。接著沿用舊分支同名欄位。Discriminator 使用新分支的值，不從舊分支帶回。沒有 branch-factory registry 或 Display 路徑對照表。

Module ref 與 inline 是以 kind 區分的通用子 model。Ref wire 是 `{kind: ref, name: actual_key}`，inline wire 是 `{kind: inline, module: {...}}`，module 保留對應 Spec，不將它的欄位攤平到 kind 層。Ref 切 inline 時，分支 factory 從 fixed library 取得隔離的 Spec，以它為起點，再沿用同名輸入。切換後內容脫鉤，不跟隨後續 library 改動。缺 key、錯 family、materialization 或 asset 錯誤由 library／cfg_model owner 明確失敗，不在表單降級為 missing 或 default。

Ref 的查詢與 materialization 分工保持 ADR-0063。表單及其 defaults 使用 LibrarySnapshot 的公開讀取；to_cfg 使用公開 factory 與固定 validation context。工具不重新組裝 library 文件或解析 cfg_model 的內部格式。

P2 fluxes 使用 LinRange(start, stop, n) 或 ExplicitList(values)。輸入保存 range，不在編輯時偷偷換成陣列。T1 to_cfg 才展開成 native device setpoints。T1 delay grid 仍由 workflow 計算，不放進表單，也不加入通用 sweep 欄位連動。

固定值不出現在表單，在 to_cfg 明寫。條件顯示只包含 tagged union 與 nullable 區段，不新增一般條件語言、inheritance 或 `$` capture。

## 每個新抽象的現存理由

| 抽象 | 現在需要它的行為或使用者 |
| --- | --- |
| FormModel | T1TunablesForm 與 nested CalibrationTunablesForm 共用 typed default_input／choices／defaults hooks。它集中 Pydantic 靜態預設與缺 key 語意，避免 getattr、cast、Protocol 或 registry；一般 nested BaseModel 不需改寫。D200 已審閱此現存理由。 |
| FormInput[F] | GUI 草稿、MCP 部分輸入與上次輸入保存都需要區分未填、null、expression 原文與普通值。它集中 JSON 編碼及表單型別配對，不增加另一份 Spec tree。 |
| FormState[F] | Editor 與 GUI host／MCP 共用 write、result、changes、reset。它隱藏分支切換、batch draft 安裝及 base 差異，不讓每個 caller 重做。 |
| FormEnv | Defaults、choices、expression 與驗證都使用同一 captured source。具名 frozen dataclass 說明 md、library、devices 的用途。只有 workflow host 建立它，不另造 Protocol。 |
| FormErrors／field_error | GUI 欄位標記與 MCP 結構化錯誤都要保留同一位置。Helper 集中 model-validator error 到欄位的轉譯。 |
| Display | Editor 與 MCP input schema都消費標籤、單位與說明，避免 app 與 catalog 各寫一份。它不選 owner 或提交時機。 |
| Expression input | Input／保存／MCP 必須識別 expression，而普通字串仍是普通字串。具名 token 對應既定 expr JSON，不引入字串前綴或自訂語法。 |
| Module ref／inline | T1 calibration 的 module 輸入需要 ref 與 override；目前的 library 已提供 pulses、readout、reset 的共同 Spec 入口。子 model 集中 discriminator 與切換起點，不建立另一個 library registry。 |
| LinRange／ExplicitList | P2 fluxes 同時要保留使用者的 range 輸入與 agent 的 explicit list。它們承載不同輸入結構，展開仍歸 T1 to_cfg。 |
| Form editor | 多份 plan／knobs 表單共用型別呈現與錯誤標記。它只編輯 FormState，不包一層新的 app Controller。 |

不新增 factory class、反射式 lowering registry、owner adapter framework 或通用 read-only wrapper。後續實作若需要此表以外的新公開抽象，先列現存理由並回到 F0／contract 審閱。

## FormEnv、evaluator 與 complex codec 重用

FormEnv 的 md 是必要的 expression/defaults 唯讀輸入，library 是固定的 LibrarySnapshot 或 None，devices 是 captured native DeviceInfo 的只讀 mapping 或空 mapping。只有 md 是 knobs caller 必需；F4 才接 4a accepted library／devices source。使用 ModuleRef 而 library 缺席時，default_input／write／result 以指名消費欄位路徑的錯誤 Fast Fail，不當未填或預設；需要 devices 而缺席時同理。Host 建立 env，工具不 load YAML、refresh 或 poll device。Frozen dataclass 不等於凍結 driver，也不承諾凍結 waveform asset bytes。

F1 按定稿先使用 MetaDict 與既有 LibrarySnapshot 讀取介面。MetaDict.snapshot 的公開契約是複製 current memory 且不做 I/O；它不是新的 immutable context 型別。工具只讀 host 提供的副本，不把 snapshot 擴解為型別上的不可變保證。

Workflow host 在 run 開始時捕捉來源，之後對同一份來源求值。Measure 日後可以換 env，工具的 result(env)／changes(env) 每次使用傳入的 env，何時重算與發布仍歸 host。本次不接 Measure。

Expression 直接重用 `gui.session.expression.evaluate_scalar_expr`。它保留 int／float／complex，擁有函式、算子白名單與 AST／數值限制。Real-only 欄位沿用既有 evaluator 入口或由表單型別拒絕 complex。不建立第二個 parser、eval、函式表、整數 coercion 或保留名稱判斷。

Complex 直接重用 `gui.cfg.codec.encode_complex`／`decode_complex`，維持既定 JSON tag。新表單不使用舊 Spec／Value draft，也不搬複製 codec。舊工具退役時才由 owning task 移動 codec owner，格式不變。

Qt editor 的 primitive 使用既有 `gui.widgets.cfg.fields.common` 的公開 value-widget 入口與 `gui.widgets.spinbox`。不得把 ScalarSpec／CfgDraft 組出來，只為使用其內部版面。若現有 primitive 入口不足以提供型別或 parse 中間態，F3 在既有 owner 擴充公開介面，再讓新舊 caller 共用，不在 workflows app 複製元件或 parser。

F4／F5 的 FormEnv 欄位以 4a 已 accepted 的 context 型別為準。當前 ParameterContext read packet不代表 ContextService publication／entry binding／compile／writeback已完成。它也不是 MetaDict，不能轉成 flat MetaDict 或 wrapper 來略過接縫。4a 持有正式 context capture 與來源讀取契約，本方在 F4 前核對其 accepted SHA／tree；必要 evaluator callee 擴充由其 owner 處理並另行接受。若不能按定稿接線，提出 ruling-request，不自行改 expression／context 設計。

## Engine 原文擴充候選

Engine 仍只驗證與使用普通 tunables model。它把 expression 原文當 opaque text，不 import gui.form，不求值、不查 md，不重建表單輸入。

F2 使用以下具名欄位擴充現有 interface，不建立 tunables wrapper：

- TunableChange 增加 `original: str | None = None`。文字是該 leaf 的 expression 原文，不含 expr JSON envelope；None 表示本次沒有原文，清除該 path 的舊原文。
- TunablesSnapshot 增加 `originals: Mapping[str, str]`。Key 是既有合法 tunable dot path，只列有原文的 leaf；values 仍是完整的求值後 tunables JSON。兩者都是 detached view。
- Engine.start 增加 keyword `tunables_originals: Mapping[str, str] | None = None`，用於 revision zero。未傳表示沒有原文。非法 path 或非文字在 startup 入口失敗，不取代先前 run。
- Startup journal保存初始普通值與原文。Tunables update journal 對每個修改保留舊／新普通值、舊／新原文、actor、時間與前後revision。

原文改變而值相同仍是一次非空修改，保留其revision與audit。Journal成功後才同時發布新值與原文。Stale revision、model error或journal error不能只更新其中一份。Journal failure沿既有Engine failure路徑傳播，不回滾先前已發布的量測檔。

目前 step 使用既有的完整值 capture；原文是證據，不參與 workflow 計算。後續 step 或 resume仍取最新revision，七參數workflow def模板與pause/stop提交語意不變。沒有原文的現有caller繼續提供普通值，不新增舊artifact讀取alias或fallback。

## 可執行依賴投影

現有 C8／C9 涵蓋 GUI frontend 到 resources、cfg_model、device 與 experiment 的合法方向。C11維持cfg_model不依賴GUI，C16禁止framework import zcu_lab。C18維持Engine不依賴GUI／Qt。C13維持MCP只走remote／wire contract。

F1 declaration seed 已建立實際 package，並在 .importlinter 加入下列第一項收緊 contract，不增加 ignore_imports。Core 內部實作完成後仍需重跑 gates 與完整 range review。第二項保留為 F3 候選，editor package 尚未建立。Contract 使用固定名稱，避免與平行上游的數字 ID 衝突。

```ini
[importlinter:contract:form-core-no-ui-policy]
name = Pydantic form core is Qt-free and has no host policy
type = forbidden
source_modules =
    zcu_tools.gui.form
forbidden_modules =
    zcu_tools.gui.app
    zcu_tools.gui.widgets
    zcu_tools.gui.remote
    zcu_tools.gui.session.services
    zcu_tools.gui.session.ui
    zcu_tools.gui.session.adapters
    zcu_tools.experiment
    qtpy
    PyQt5
    PyQt6
    PySide2
    PySide6

[importlinter:contract:form-editor-no-host]
name = Pydantic form editor does not depend on app or execution policy
type = forbidden
source_modules =
    zcu_tools.gui.widgets.form
forbidden_modules =
    zcu_tools.gui.app
    zcu_tools.gui.remote
    zcu_tools.experiment
    zcu_tools.mcp
```

第二項由F3在editor package建立時加入。F4同步保護既有runtime workflow definitions不import `zcu_tools.gui`或`zcu_lab.workflows.gui`，與既有C15 user-core政策一致，不將GUI附件放回core。

新增依賴邊的判讀如下。

| 邊 | Contract與理由 |
| --- | --- |
| gui.form → gui.session.expression、gui.cfg.codec | Qt-free leaf重用；form-core contract限制不能轉成services／UI／app dependency。C9保持top-level無循環。 |
| gui.form → resources.context、device型別 | C8/C9合法frontend到capability，C10/C11/C12保護callee方向。只讀captured資料，不控制硬體。 |
| gui.form → cfg_model | Module ref 切 inline 時取得的 Spec 屬於 cfg_model。C8／C9 涵蓋此 frontend 到共享核心的方向，C11 保護 cfg_model 不反向依賴 GUI。 |
| gui.widgets.form → gui.form、既有primitive owner | form-editor contract隔離host，form-core contract防止反向editor import。 |
| zcu_lab.workflows.gui → gui.form、experiment／cfg_model／native workflow models | 外部GUI附件消費framework與native型別；C16禁止反向，F4新增runtime-core禁GUI contract。C8不直接涵蓋外部附件，理由是D75的具體實驗frontend位置。 |
| workflows app → gui.form／widgets.form、Engine／shared session | C8/C9 frontend組合方向，C18禁止Engine反向GUI。具體workflow與表單由外部組合根注入，不豁免C16。 |

新增contract與每一條實際import仍在各切片執行lint-imports。Imports PASS不是省略edge清單的理由。

## 驗證、切片與未解除邊界

F1∥F2在F0核准後各用獨立lane。F1只寫form core／tests與指定contracts，F2只寫Engine／tests。F3∥F4在F1介面接受後各用獨立lane。Primitive擴充歸F3，T1 defaults／to_cfg歸F4，不讓兩個writer共寫。共享ADR index與import contract修改由Orchestrator安排先後。

F1從公開FormState、schema與保存還原驗證U1–U3、U7、A1–A4資料行為。F2從Engine start/update/tunables/execute及journal驗證原文與值原子發布。F3從Qt editor→FormState驗證編輯行為。F4從defaults→result→to_cfg驗證native成品。F5以GUI／MCP共用host入口完成U1–U7與A1–A4，另外完成裝置上下限、refresh與CP-B。

不直接測私有函式、腳本、文件或設定字面內容，不在tests另啟程序驗證Qt-free。Qt-free由import contracts及無QApplication的公開seam選集驗證，測試平行化交給pytest CLI。

F4／F5仍受4a已接受context型別與D196必要P2 caller blockers限制。149項繼承回歸由workflows tracker逐項持有，沒有waive。Main前全樹Pyright與detectors必須對當時main零回歸，且由使用者核准。不push、不操作硬體或真資料。

## 取捨

新工具和CfgSchema暫時並存，代價是兩種input模型。它們共用現有evaluator、complex格式與primitive owner，沒有兩套求值／編碼規則。Defaults與to_cfg由adapter明寫，讓外部參數與native cfg的關係能在同一函式中閱讀。表單工具不承擔提交policy，省去為顯式Apply與日後隱式Apply設計owner協定。

本次只實作定稿列出的能力。CfgSchema的inheritance、`$` capture、Measure refresh與Unavailable publication只保留對應與未實作界線，不加未使用的extension point。
