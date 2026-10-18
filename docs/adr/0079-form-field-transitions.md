---
status: accepted
date: 2026-10-18
---

# ADR-0079 — Form 欄位轉換、固定元素與逐元素操作

## 決策狀態與問題

本篇記錄已採用的設計，實作尚未完成。它取代 [[0065]]、[[0073]] 中普通切換補 fresh defaults、數值葉 command 改 parent、僅保存 input 並丟棄不相容葉的規則，並擴充 [[0068]] 的 indexed paths 與 recipe 契約。既有 guard、capture、Run／Load 和 writeback 所有權不變。

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

保存內容包括每個欄位的 selection／branch 與原始 input，含 expression 和未完成草稿。還原從可信型別註釋與宣告重建結構，再填回原稿。不執行 converter，不序列化 callable 或動態 class，不另造 schema 語言。

還原依現行 annotation 與存檔的名稱、形狀遞迴；list 長度與位置沿存檔，不比固定 identity。不相符欄位列入 dropped。使用者於 2026-10-18 要求「還有版本？太複雜了。我需要更簡單的機制」，看過遞迴規則後確認「好，可以」。Ref→inline 後保存的草稿，還原時保留當時 inline branch 與原稿，不從已修改的 library 重新複製。

### 固定具名元素與參與清單

Core 對普通 `list[Model]` 提供中性描述、picker 與 indexed editing，不辨識固定元素或 participants。`Display` 只提供元素 label 與 readonly 呈現提示。固定政策集中在 `gui.form_adapter`，宣告 owner 列出 list path、名稱欄位及 participants path。新建與 Reset 從 capture 建表，重開及還原沿存檔的長度、位置與名稱，不與新 capture 比 identity。

`FormModel.check_input_change(before, after)` 在普通 mutation 的完整候選安裝前執行宣告檢查，兩棵輸入都是複本。改名、增刪或重排以具名、帶原 typed path 的拒絕原子失敗，input、base、selections 與通知都不變。Widget 與 RPC 顯示可恢復錯誤。還原及 Reset 不經此檢查。Participants 非法值仍存成 invalid draft，由一般驗證報錯並擋 Run。

使用者於 2026-10-18 看過傳複本與宣告檢查方案後裁定：「可以」。這個檢查集中普通 mutation 的 admission，不把固定 identity 政策放回 Core。

需要名稱定位的表單在建立時拒絕重複元素名，錯誤指名重複名稱與原始 indices。宣告 owner 只檢查名稱的唯一性，不解讀 resonator，不改 [[0072]] 的 group schema。元素名取法歸實驗定義者；若實驗需要重複 resonator，可以宣告 R1#0、R1#1 等不同名稱，framework 不自行加後綴。這項限制由顧問於 2026-10-18 確認，避免 participants 與 recipe overrides 無法唯一定位。

Workflow 的 Apply 以宣告的 Form.to_cfg() 轉成完整 native tunables model，再由 Engine 依 workflow 型別驗證、整體替換並增加 revision。List 與固定 list 隨完整 model 提交，不攤平 preview leaves，也不建立 native index path grammar。每次 Apply 的 journal 記 actor、revision_before／after 與完整新 tunables JSON。Append 成功後才發布值與 revision，宿主才接受 input／base。使用者於 2026-10-18 確認完整 model 直接套用、不比對差異，並在日誌寫法中確認「甲乙都可以」。採完整快照，不維護逐葉差異或舊格式相容層。當前 step capture 與 iteration record.json 格式不變，見 [[0062]]。

Mux 的 lab cfg 使用完整有序 `tones: list[ToneCfg]`。每個元素的 `name` 由 capture 產生且不可編輯。Drive 類實驗以 qubit 名稱識別，onetone 以 resonator 名稱識別。每個元素都有必填 drive，有掃描的實驗另有 sweep。未參與元素保留設定，適用同一驗證規則，無效時也擋 Run。

`participants: list[str]` 必須非空、無重複且每個名稱都存在。順序就是執行與資料 column 順序。Lab runtime 邊界只轉換一次名稱到原始位置。例如完整元素為 Q1、Q2、Q3，participants 為 Q3、Q1，既有 program 接收的 active_tones 為 2、0。只選 Q2 時仍是 1，不重新編號。Program／raw 接口不改，framework 不解讀 mux group。

使用者於 2026-10-18 裁定：「那設定也不用逐個toggle了，直接[Q1,Q2]這樣就好。」同日授權「可以，之後就交給advisor，我就不直接對話」。顧問據此採具名 list，未採使用者仍不確定的 mapping 傾向。理由是 list 的位置直接保存原始 tone index，不需依賴 mapping 順序或另一張對照表。保存還原及下節契約同樣由顧問代裁，待使用者覆核。

Group 固定到 Reset 是討論中的傾向，尚不把它提升為新增的獨立政策。既有 Form host 仍遵守 [[0065]] 的固定 capture；本設計不引入自動刷新。

### Remote 與 recipe

Model-field path 使用 typed segments：欄位名為 str，list index 為 int，例如 `["tones", 1, "drive", "gain"]`。Describe 同時提供元素位置與 name。Agent 先用名稱辨識位置，再使用 typed path，不增加名稱 path grammar。

一次讀全部元素由 owner 在單一 readonly 呼叫產生。MCP 不逐元素發出 N 次讀取再拼成假快照，不持有平行的名稱排序表。完整讀取保留每個元素的 authored input、可用值與定向錯誤，不因一個元素無效而隱藏其他元素。

Recipe 輸入包含 participants，以及以元素 name 為 key 的逐元素 overrides。Owner 在 Run 前解析 name 到位置，沿相同 Form 操作套用，不在 MCP／recipe 重做 converter。每個葉欄位保留 explicit／default／derived 來源。Actual、estimates 與 writeback candidates 逐元素標示 name，取當次 Run／Load 的事實，不用後來的 Reset 或當前 alias 改寫已完成回報。顯示元素名不授權寫回，目的地與 Apply 仍歸既有 writeback owner。

### 模組分工與可執行依賴規則

- `gui.form` 擁有中立狀態及安裝，沿用 `form-core-no-ui-policy`。
- `gui.form_adapter` 擁有生成、領域 converter 與共用繼承宣告，沿用 `form-adapter-no-host`。實驗的初值政策仍歸實驗，見 [[0073]]。
- `gui.widgets.form` 消費 description、呈現 picker／選單，沿用 `form-editor-no-host`。
- Host 擁有 capture、guard、生命週期與提交，remote 投影 owner 的結果。MCP 只依賴公開 remote 契約，沿用 C13 `mcp-remote-only`。
- Lab 擁有 mux cfg、來源關係、名稱到原始 index 的 runtime 轉換，以及實驗 recipe 接線。C15 保護實驗 core 不依賴 GUI，C16 禁止 framework 反向依賴 lab。分層與無循環沿用 C8、C9。

本篇不新增跨 package 依賴方向，不放寬 ignore_imports。共用機制供 measure、workflows、Inspect、New Entry 與 writeback 使用，各宿主的 Apply／Cancel、Run admission 及關閉規則不變。

## 代價與不採用的方案

保留 selection 與 branch 是為了實現「沒有 converter 就保留子表單」，保存格式因此不能只記選項。若可信宣告不足以還原某個合法草稿，必須用具體反例重新裁決，不偷偷序列化執行時 schema。

所有元素統一驗證會使未參與元素的錯誤擋住 Run。使用者需要修正或 Reset；不另設 inactive 特例。固定 list 不提供任意容器編輯器，command 不提供參數表單，picker 不增加 reactive label 系統。

Framework 交付通用能力與既有 hosts 遷移。Mux 實驗 cfg、lab recipe 及接線另行交付。FTF、硬體操作、任意動態 schema、legacy 雙讀與 program／raw 格式變更均不在本設計內。
