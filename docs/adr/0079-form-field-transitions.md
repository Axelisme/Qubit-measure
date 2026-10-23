---
status: accepted
date: 2026-10-18
---

# ADR-0079 — Form 欄位轉換、固定元素與逐元素操作

## 決策狀態與問題

本篇記錄已採用的設計，實作尚未完成。它取代 [ADR-0065](0065-cfg-editing.md#局部取代範圍)、[ADR-0073](0073-measure-form-derivation.md#局部取代範圍) 中普通切換補 fresh defaults、數值葉 command 改 parent、僅保存 input 並丟棄不相容葉的規則，並擴充 [[0068]] 的 indexed paths 與 recipe 契約。既有 guard、capture、Run／Load 和 writeback 所有權不變。

同一份 Form 必須讓 Qt、remote 與 recipe 執行同一操作。若 Core 認得每種 module、reference 或 sweep，新增領域型別就會改動通用核心。若 command 能改掛載欄位之外的資料，單看欄位宣告就無法判斷副作用。固定多元素設定還需要區分元素位置、顯示名稱與此次參與順序。

## 決策

### 中立 Core 與單一安裝流程

`gui.form` 只擁有欄位、子表單結構、原始 input、description、驗證、安裝與通知。它不認 Ref、Module、Waveform 或 Sweep。宣告者的 callable 負責來源解析、複製、繼承與數值換算。

一個欄位至多宣告一個 converter，處理這個欄位所有選項切換。Caller 提供目標 selection。Converter 回新的子表單結構與 input，不回另一個 selection。沒有 converter 時，Core 只改 selection，保留子表單結構與原始 input。Core 不補 fresh defaults，也不自動搬移同名欄位。Selection 與目前子表單 branch 是各自需要保存的狀態，不能只靠 selection 猜回 branch。

Command 掛在欄位上，只能替換這個欄位及 descendants。它可以回包含 selection 的整個欄位狀態，不能修改 parent 或 siblings。Sweep 模式 command 因此掛在 sweep 條目，Qt 由該條目的右鍵選單呈現，不再掛在 start、stop 或 n。MCP／recipe 呼叫相同的公開操作。

兩種 callable 共用 Core 的原子安裝。Callable 不直接寫 state、不操作 UI、不自行通知。Core 在安裝前檢查修改範圍及型別註釋允許的結構，再一次提交 selection、結構與 input。通知讀者不會看到切到新選項但仍等待安裝的中間狀態。

使用者於 2026-10-18 裁定：「我認為沒給converter就與new = old一樣」，並確認是選項更新、子表單不變。作用範圍裁定為「command應該只能改自己以及子欄位」。

### 值寫入與選項切換

`FormState.write(batch, env)` 一律覆寫每個 typed path 上的完整 raw 子樹，清除該 path 及其下的 selection overrides，由新值決定 branch。它不合併、不補 defaults、不執行 converter。單欄修改使用完整欄位 path。Model result 可以依預設補值，但不回填 raw。

`select(path, selection, env)` 是明確的選項切換，只有這條路徑執行 converter。`prepare_write` 與 `prepare_selection` 分別提供未安裝的候選，host 必須在同一 guard 與 current input 下使用。寫入仍遵守 annotation、來源與 owner admission、initial-active path 解析及批次原子性。Generated selectors 保留專用操作。

使用者於 2026-10-19 問「為什麼不是總是覆寫path位置的值？」，並對統一覆寫、converter 僅用於選項切換的方案回答「同意」。因此不另提供 replacement operation 或 operation 欄位。

### 宣告唯讀與子樹

`Display(readonly=True)` 是 Form core 執行的寫入限制。`FieldMeta(readonly=True)` 由生成器投影成相同宣告。同值寫入通過，不同值拒絕，Qt、MCP 與 RPC 共用此規則。完整父層替換與批次寫入不能繞過唯讀子欄位。

唯讀只作用於被宣告欄位的子樹，不固定祖先選擇。Module 子欄位唯讀不移除 library key 選項。Module 節點本身標唯讀則固定整個 module 子樹，包含 ref／inline 選擇與所有子欄位，不論初值是 ref 或 inline。不另設 selector-only 宣告，也不自動強制 inline。固定欄位的初值可逐 list 位置不同，不以生成單值 Literal 實作唯讀。範本的非 lock 初值與 input projection 沿 [[0073]] 保留。

使用者於 2026-10-19 裁定：「那為什麼不直接都用ReadOnly就好？」「我會選B，lock應該被認知為只對子樹作用。」固定值的生成與 cfg-only 欄位分工見 [[0073]]，remote 拒絕格式見 [[0065]]。

### 原始草稿、求值與失敗

Callable 接收具名唯讀 old snapshot，包含 selection、子表單結構與原始 input。它不要求整份 Form 有效。具名 context 提供 `evaluate(path)`，只求值 callable 所需欄位。宣告者在宣告時綁定來源讀取能力，宿主管理其生命週期，Core 不接管 library 或 group。

輸出使用具名型別。結構受可信型別註釋與宣告限制，不是可執行的第二套 schema。例：`list[ToneCfg]` 的元素結構仍必須是 ToneCfg。值驗證與結構檢查分開。

| 結果 | Core 行為 |
| --- | --- |
| 無法產生結果，例如必要 expression 求值失敗 | 拒絕操作，selection、結構與 input 全部不變 |
| 結構合法，但普通值驗證失敗 | 安裝結果，保留 invalid draft 與診斷 |
| 結構違約或修改範圍外資料 | 拒絕安裝，全部不變 |

Sweep 只填 start／stop、尚未填 n 時仍可切座標，因為該換算不需要 n。未參與換算的原稿及 expression 保留。Preview、describe 與 aggregate read 不執行 converter。

### 共用 inline 政策

`gui.form_adapter` 共用一條同名同型別繼承規則，不維護來源／目標配對表，各實驗不重寫。新型別的每個欄位若在舊型別有同名且同型別的欄位，就沿用原始輸入；否則用目標初值。Waveform、Pulse、PulseReadout、MuxReadout 與各 Reset 都用這條規則，巢狀欄位逐層比較。Core 不加入 domain 規則，也不補 defaults；converter 提供完整目標 input。

型別指生成後的可編輯 Form 型別，沿 adapter 既有去除 QickParam 的規則。各次生成的同一持久 model 不因動態 class identity 不同而變成不同型別。欄位名必須完全相同，不做 rename。PulseSpec.ch 與 PulseReadoutSpec.gen_ch 不同名，不帶過；freq、gain、waveform 等實際同名同型別欄位則沿用。

沿用的是原始 expression、缺席與 invalid 值，不依 raw 值是否有效決定繼承，不求值或自動修正。超出目標限制時，由一般驗證顯示錯誤。同型別 ref→inline 與 inline→inline 自然保留全部可編輯欄位。Ref 讀固定 Spec，inline 不重新讀 library。

Gauss length=0.8、sigma=0.2 切 Drag 時保留 length／sigma，delta／alpha 用初值。切 FlatTop 時保留 length，raise_waveform 用初值，不把舊 waveform 包進新的巢狀欄位。切 Arb 時全用初值。FlatTop 的 length 與 raise_waveform 共同決定形狀，因此沿用 length 可能不符合使用者對平台長度的預期。表單顯示原值，由使用者修改，不加配對例外。

使用者於2026-10-18裁決：「我認為Module間切換也可以採用類似的行為，這樣才一直，我甚至認為就單純的比對欄位名與形態來決定是否繼承值就好」。確認規則與例子後答「可以」。此規則取代先前配對政策。顧問同日釐清比較可編輯 Form 型別，並確認 ch／gen_ch 不搬移。

明確的 Convert to inline 保留，表示從固定來源複製內容。選項切換依目標型別及共用繼承規則產生子表單。兩個入口可以產生相同結果，不為了區別入口而改值。

### 保存與還原

FormInput 的 JSON 固定為 `{data, selections}`。data 保存 raw input，selections 是 `[{path, value}]`，path 保留字串與整數。沒有 override 時仍寫空 list。真實表單欄位在 data 內，不與兩個外層名稱衝突。不保存版本、型別 identity、domain metadata、callable 或動態 class。Host 只保留必要的表單容器，不再包一層版本。舊 raw-only 或 versioned 格式不雙讀。

啟動讀檔、source reload 與 RAM switching 共用 `FormState.restore_input`。Model 按現行同名欄位遞迴，缺值用 defaults，多出欄位列 dropped。List 按位置遞迴，長度照存檔。Union／Literal 的 selection 仍為 annotation 接受的選項就保留，否則用 defaults 並列 dropped。沒有 discriminator 的 union 不猜 branch，raw 照搬。葉欄位保留 expression、缺席與 invalid raw，交一般驗證處理。唯讀欄位沿同一形狀規則保留存檔值，不從宣告重算，也不因初值不同而丟棄。形狀不符的內容丟棄並列 dropped，host 將清單顯示給使用者。還原與 Reset 不走普通 mutation 的 owner input-change 檢查。

還原不執行 converter 或 options hook，也不讀 library。一般驗證檢查明確 selection override 的當前 hook membership。沒有 override 的 raw branch 不因選單未列出它而失效。Ref 不存在或 family 不符時，adapter 的一般驗證在欄位顯示錯誤，保留草稿並擋 Run。已 inline 的原稿不要求原 ref 仍存在，也不從改過的 library 重新複製。

JSON 損壞或不符固定 FormInput 格式時，共用 caretaker 顯示錯誤並以 defaults 開啟，不備份。之後依原有時機正常保存。Provider 與其他 restore 故障仍傳播，read I/O 故障沿既有 error outcome，不轉譯成 JSON 格式錯誤。見 [[0063]]。

使用者於 2026-10-19 確認唯讀還原政策：「1同樣值不報錯。2我選存檔的版本，因為有些lock值可能是生成的。能不能執行是另外一回事」。還原保留值，能否執行由後續驗證判定。

使用者於 2026-10-18 要求簡化保存還原：「還有版本？太複雜了。我需要更簡單的機制」。看過依名與形狀的遞迴規則後回覆「好，可以」，並選擇固定 data／selections 兩欄的 JSON 方案。

### 固定具名元素與參與清單

Core 對普通 `list[Model]` 提供中性描述、picker 與 indexed editing，不辨識固定元素或 participants。`Display(element_label=...)` 提供元素 label，`Display(readonly=True)` 由 Core 強制唯讀。具名元素的 identity 與 participants 政策集中在 `gui.form_adapter`，宣告 owner 列出 list path、名稱欄位及 participants path。新建與 Reset 從 capture 建表，重開及還原沿存檔的長度、位置與名稱，不與新 capture 比 identity。

`FormModel.check_input_change(before, after)` 在普通 mutation 的完整候選安裝前執行宣告檢查，兩棵輸入都是複本。改名、增刪或重排以具名、帶原 typed path 的拒絕原子失敗，input、base、selections 與通知都不變。Widget 與 RPC 顯示可恢復錯誤。還原及 Reset 不經此檢查。Participants 非法值仍存成 invalid draft，由一般驗證報錯並擋 Run。

使用者於 2026-10-18 看過傳複本與宣告檢查方案後裁定：「可以」。這個檢查集中普通 mutation 的 admission，不把固定 identity 政策放回 Core。

`FixedElements(list_path)` 不宣告 name 或 participants，使用固定槽位模式。
它只拒絕目前 inline 清單的增刪，不限制各槽位的欄位值，也不把可編輯 label 當 identity。
整列表交換值與逐槽位修改無法區分，因此本模式不保證 resonator 身份或禁止值的重排。
它不產生 ElementIdentity，也不新增持久欄位。

宣告路徑可穿過 nested ref／inline module，定位 active inline list。
合法 ref／inline 切換仍可進行。切到 inline 時以新清單建立槽位數邊界，不跨分支保存隱藏身份。
還原與 Reset 照既有規則建立新邊界。具名 list 與 participants 的原有規則不變。
Lab 從 mux group 決定 resonator 清單與初值，Form adapter 依原 Spec 生成 typed model-list input。
Core、widget 與 host 不辨識 mux group。

使用者於 2026-10-20 同意從 mux group 產生固定清單並逐欄編輯，原話為「可以」。
顧問同日採 typed model-list seeds 與無 participants 的固定槽位模式，保留合法 ref／inline 切換。

需要名稱定位的表單在建立時拒絕重複元素名，錯誤指名重複名稱與原始 indices。宣告 owner 只檢查名稱的唯一性，不解讀 resonator，不改 [[0072]] 的 group schema。元素名取法歸實驗定義者；若實驗需要重複 resonator，可以宣告 R1#0、R1#1 等不同名稱，framework 不自行加後綴。這項限制由顧問於 2026-10-18 確認，避免 participants 與 recipe overrides 無法唯一定位。

Workflow 的 Apply 以宣告的 Form.to_cfg() 轉成完整 native tunables model，再由 Engine 依 workflow 型別驗證、整體替換並增加 revision。List 與固定 list 隨完整 model 提交，不攤平 preview leaves，也不建立 native index path grammar。每次 Apply 的 journal 記 actor、revision_before／after 與完整新 tunables JSON。Append 成功後才發布值與 revision，宿主才接受 input／base。使用者於 2026-10-18 確認完整 model 直接套用、不比對差異，並在日誌寫法中確認「甲乙都可以」。採完整快照，不維護逐葉差異或舊格式相容層。當前 step capture 與 iteration record.json 格式不變，見 [[0062]]。

Mux 的 lab cfg 使用完整有序 `tones: list[ToneCfg]`。每個元素的 `name` 由 capture 產生且不可編輯。Drive 類實驗以 qubit 名稱識別，onetone 以 resonator 名稱識別。Readout 身分欄位全 tone 必填。Drive 與逐 tone sweep 是 nullable，只有 participants 必填並檢查 drive 與 channel。Open／Reset 先選定 participants，再替其 drive／sweep 填初值，其餘填 None。欄位保持一般 nullable 編輯，非 None 值照常驗證。清單不裁剪、不重新編號。

`participants: list[int]` 必須非空、無重複，且每個值都是完整 tones 的合法原始索引。順序就是執行與資料 column 順序。例如完整元素為 Q1、Q2、Q3，participants 為 `[2, 0]` 時先量 Q3，再量 Q1。只選 Q2 時仍是 `[1]`。名稱只供 label、錯誤與回覆顯示，recipe 與 runtime 不再以名稱查位置。Program／raw 接口不改，framework 不解讀 mux group。

使用者於 2026-10-23 裁定：「然後participants用list of index好了，不要用list of str」。對非 participant 可留空的甲案，使用者答「甲」。原始索引讓子集與執行順序共用同一座標，不再需要名稱查表。未量測的 drive／sweep 不應阻擋 Run，但所有 readout tones 都參與讀出，所以其身分與通道仍須完整。

一般編輯 participants 不重設 tone 值。新加入的 tone 若仍為 None，cfg validator 在原始 tone 路徑提示「填入 …」。移出的 tone 保留原值。掃描、drive 與 channel 檢查只遍歷 participants，readout 身分仍全 tone 必填。初值政策由 lab 擁有，Core 不辨識參與名單。

使用者於 2026-10-23 裁定：「這機制有點太複雜了，我的想法是預設participant以外的tone都初始化填None」。此為 Open／Reset 初值規則，不在編輯時改值或停用欄位。

使用者於 2026-10-18 裁定：「那設定也不用逐個toggle了，直接[Q1,Q2]這樣就好。」同日授權「可以，之後就交給advisor，我就不直接對話」。顧問據此採具名 list，未採使用者仍不確定的 mapping 傾向。理由是 list 的位置直接保存原始 tone index，不需依賴 mapping 順序或另一張對照表。保存還原及下節契約同樣由顧問代裁，待使用者覆核。

Group 固定到 Reset 是討論中的傾向，尚不把它提升為新增的獨立政策。既有 Form host 仍遵守 [[0065]] 的固定 capture；本設計不引入自動刷新。

### Remote 與 recipe

Model-field path 使用 typed segments：欄位名為 str，list index 為 int，例如 `["tones", 1, "drive", "gain"]`。Describe 同時提供元素位置與 name。Agent 先用名稱辨識位置，再使用 typed path，不增加名稱 path grammar。

一次讀全部元素由 owner 在單一 readonly 呼叫產生。MCP 不逐元素發出 N 次讀取再拼成假快照，不持有平行的名稱排序表。完整讀取保留每個元素的 authored input、可用值與定向錯誤，不因一個元素無效而隱藏其他元素。

Recipe 輸入包含以完整原始索引表示的 participants，以及與 MCP 相同的 typed path overrides。例如 `[{"path": ["tones", 0, "drive", "gain"], "value": 0.3}]` 直接定位完整原始 list 的第 0 個元素。Overrides 不以元素 name 為 key，owner 不做 name→位置轉換。Owner 在 Run 前沿相同 Form 操作套用 overrides，不在 MCP／recipe 重做 converter，也不另建排序表。

每個葉欄位保留 explicit／default／derived 來源。Actual、estimates 與 writeback candidates 以完整原始 list 的位置標示，並附元素 name 供閱讀，不依 participants 順序重新編號。這些內容取當次 Run／Load 的事實，不用後來的 Reset 或當前 alias 改寫已完成回報。顯示位置與元素名不授權寫回，目的地與 Apply 仍歸既有 writeback owner。

使用者於 2026-10-18 指出 recipe 名稱定位與核心 index 不一致，裁定：「既然核心是用index，那就用適合核心的方案，不要加一堆難維護的轉換邏輯」。因此 overrides 與 MCP 共用 typed paths；2026-10-23 的索引裁定進一步讓 participants 也直接表示完整原始位置與此次參與順序。

### 模組分工與可執行依賴規則

- `gui.form` 擁有中立狀態及安裝，沿用 `form-core-no-ui-policy`。
- `gui.form_adapter` 擁有生成、領域 converter 與共用繼承宣告，沿用 `form-adapter-no-host`。實驗的初值政策仍歸實驗，見 [[0073]]。
- `gui.widgets.form` 消費 description、呈現 picker／選單，沿用 `form-editor-no-host`。
- Host 擁有 capture、guard、生命週期與提交，remote 投影 owner 的結果。MCP 只依賴公開 remote 契約，沿用 C13 `mcp-remote-only`。
- Lab 擁有 mux cfg、來源關係、Open／Reset 的 participants 索引選定與 tone 初值，以及實驗 recipe 接線。C15 保護實驗 core 不依賴 GUI，C16 禁止 framework 反向依賴 lab。分層與無循環沿用 C8、C9。

本篇不新增跨 package 依賴方向，不放寬 ignore_imports。共用機制供 measure、workflows、Inspect、New Entry 與 writeback 使用，各宿主的 Apply／Cancel、Run admission 及關閉規則不變。

## 代價與不採用的方案

保留 selection 與 branch 是為了實現「沒有 converter 就保留子表單」，保存格式因此不能只記選項。若可信宣告不足以還原某個合法草稿，必須用具體反例重新裁決，不偷偷序列化執行時 schema。

所有元素統一驗證會使未參與元素的錯誤擋住 Run。使用者需要修正或 Reset；不另設 inactive 特例。固定 list 不提供任意容器編輯器，command 不提供參數表單，picker 不增加 reactive label 系統。

Framework 交付通用能力與既有 hosts 遷移。Mux 實驗 cfg、lab recipe 及接線另行交付。FTF、硬體操作、任意動態 schema、legacy 雙讀與 program／raw 格式變更均不在本設計內。
