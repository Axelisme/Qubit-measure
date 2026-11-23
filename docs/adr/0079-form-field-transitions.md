---
status: accepted
date: 2026-10-18
---

# 0079 — Form 欄位轉換、固定元素與逐元素操作

## 決策狀態與問題

欄位切換、command 作用範圍與草稿還原以本篇為準，局部取代 [ADR-0065](0065-cfg-editing.md#局部取代範圍) 與 [ADR-0073](0073-measure-form-derivation.md#局部取代範圍)。本篇也擴充 [[0068]] 的 indexed paths 與 recipe 契約。既有 guard、capture、Run／Load 與 writeback 所有權不變。

[Form core](../../lib/zcu_tools/gui/form/README.md) 提供選項轉換與草稿還原。[Measure 的 reload](../../lib/zcu_tools/gui/app/measure/README.md#sources-and-workflow)使用共用 codec 與還原入口。[固定元素宣告](../../lib/zcu_tools/gui/form_adapter/README.md#固定元素宣告)也用於 Mux Amp Rabi 的 tones。

Qt、remote 與 recipe 要能在同一份 Form 執行同一操作。領域型別擴充不應改動通用 Core，command 的掛載位置應能說明它會改哪些資料。固定多元素設定需要區分元素位置、顯示名稱與此次執行順序。

## 決策

### 中立 Core 與單一安裝流程

**規則。** `gui.form` 擁有欄位、子表單結構、原始 input、description、驗證、安裝與通知。每個欄位最多有一個 converter，處理該欄位所有選項。Caller 提供目標 selection，converter 只回子表單結構與 input。Command 可回包含 selection 的完整欄位狀態，但只能替換掛載欄位及其子欄位。Core 檢查作用範圍與型別註釋允許的結構，再一次安裝 selection、結構與 input 並通知。

**例子。** Sweep 模式 command 掛在 sweep 條目，Qt 在該條目的右鍵選單呈現。MCP 與 recipe 呼叫同一公開操作，不讓 start、stop 或 n 的 command 改父層。

**邊界。** 沒有 converter 時，Core 只更新 selection，子表單 branch 與原始 input 不變。保存時要分開記 selection 與 branch。Core 不補 fresh defaults，不搬同名欄位，也不辨識 Ref、Module、Waveform 或 Sweep。來源解析、複製、繼承與換算由宣告者的 callable 負責。Callable 不直接寫 state、操作 UI 或通知。介面見 [Form Core README](../../lib/zcu_tools/gui/form/README.md)。

**依據。** 2026-10-18，使用者裁定：「我認為沒給converter就與new = old一樣」，並確認只改選項，子表單不變。對作用範圍裁定：「command應該只能改自己以及子欄位」。原始討論未入庫，見 Git `3baf6eb0a`。

### 值寫入與選項切換

**規則。** 值寫入覆寫指定 typed path 的完整 raw 子樹，清除該路徑與後代的 selection overrides，以新值決定 branch。寫入不合併、不補 defaults。只有明確選項切換才執行 converter。

**例子。** 只改一欄就寫該欄的完整路徑。若改寫整個 model，result 可以按預設補值，但不回填 raw。`write`、`select` 及未安裝候選的 `prepare_write`、`prepare_selection` 見 [Form Core README](../../lib/zcu_tools/gui/form/README.md)。

**邊界。** Host 在同一 guard 與 current input 下使用候選。寫入遵守 annotation、來源與 owner admission，按操作開始時的 active path 解析並原子安裝整批。Generated selectors 沿用[專用操作](../../lib/zcu_tools/gui/form_adapter/README.md#logical-操作)。不另設 replacement operation 或 operation 欄位。

**依據。** 2026-10-19，使用者問「為什麼不是總是覆寫path位置的值？」，並對統一覆寫、converter 僅用於選項切換答「同意」。原始討論未入庫，見 Git `227f6dab8`。

### 宣告唯讀與子樹

**規則。** Core 執行 `Display(readonly=True)`，生成器將 `FieldMeta(readonly=True)` 投影成同一限制。Qt、MCP 與 RPC 都接受同值寫入，拒絕異值寫入。整個父層替換和批次寫入也不能繞過唯讀子欄位。

**例子。** Module 子欄位標唯讀時，library key 仍可選。Module 節點本身標唯讀時，ref／inline 選擇與所有子欄位都固定，不論初值是哪一支。

**邊界。** 唯讀只限制被宣告的子樹，不固定祖先。不另設 selector-only 宣告或強制 inline 是 [ADR-0073〈Form 的形狀與初值〉](0073-measure-form-derivation.md#form-的形狀與初值) 所記的 designer 實作選擇。唯讀不靠生成單值 Literal 實作。List 各位置可以有不同的唯讀初值。範本的非 lock 初值、input projection 與 cfg-only 分工沿用 [[0073]]，remote 拒絕格式見 [[0065]]。

**依據。** 2026-10-19，使用者問「那為什麼不直接都用ReadOnly就好？」並裁定：「我會選B，lock應該被認知為只對子樹作用。」原始討論未入庫，見 Git `c499c06d9`。

### 原始草稿、求值與失敗

**規則。** Callable 接收具名唯讀 snapshot，內含 selection、子表單結構與原始 input，並以具名 context 的 `evaluate(path)` 只求值所需欄位。輸出用具名型別，結構依可信型別註釋與宣告檢查，不另建可執行 schema。值驗證獨立於結構檢查。

**例子。** Sweep command 只換座標時，有 start／stop 就能換算，不要求 n 已填。未參與換算的原稿與 expression 照留。`list[ToneCfg]` 的結果仍須使用 ToneCfg 的元素結構。介面及錯誤見 [Form Core README](../../lib/zcu_tools/gui/form/README.md) 與 [Sweep 表示](../../lib/zcu_tools/gui/form_adapter/README.md#sweep-表示)。

**邊界。** 不要求整份 Form 有效。必要 expression 求值失敗等無法產生結果的情況，或結果違反結構及作用範圍時，Core 拒絕操作，selection、結構與 input 都不變。結構合法但普通值驗證失敗時，照樣安裝並保留 invalid draft 和診斷。Preview、describe 與 aggregate read 不執行 converter。宣告者綁定來源讀取能力，host 管理其生命週期，Core 不接管 library 或 group。

**依據。** 原始討論未入庫。局部求值與區分結構拒絕、值錯誤的設計見 Git `3baf6eb0a`。

### 共用 inline 政策

**規則。** `gui.form_adapter` 為 Waveform、Pulse、PulseReadout、MuxReadout 與各 Reset 共用同一繼承規則。巢狀欄位逐層比較，只有同名同可編輯 Form 型別才沿用 raw，其餘用目標初值。不維護來源／目標配對表，各實驗不重寫規則。Converter 提供完整目標 input。一般生成 tagged-model union 也共用這條規則：None 切到分支用目標初值，同分支重選保留 raw，異分支按規則繼承，不保留離開分支的草稿。

**例子。** Gauss 的 `length=0.8`、`sigma=0.2` 切到 Drag 時沿用兩值，delta／alpha 用初值。切 FlatTop 時沿用 length，raise_waveform 用初值，不把舊 waveform 包進去。沿用 length 可能不符合使用者對平台長度的預期，表單保留原值供使用者改，不設配對例外。切到 Arb 則全用初值。PulseSpec.ch 與 PulseReadoutSpec.gen_ch 不同名，不繼承；freq、gain、waveform 等同名同型別欄位則沿用。

**邊界。** 型別比較沿用 adapter 去除 QickParam 的規則。同一持久 model 不因生成的 class identity 不同而變成異型。繼承保留 expression、缺席與 invalid raw，不求值、不修正，超出目標限制由一般驗證報錯。同型別 ref→inline 與 inline→inline 保留全部可編輯欄位；ref 讀固定 Spec，inline 不重讀 library。Convert to inline 明確複製固定來源，普通選項切換按目標型別繼承；兩個入口可以得到相同結果。[Form adapter README](../../lib/zcu_tools/gui/form_adapter/README.md) 記錄投影與複製介面。

作者的 metadata 或 model hook converter 優先於一般 union 政策。Sweep 的作者 converter 在必要欄位都是有效直接數值時，重用 command 換算 Range↔Center 與 Expts↔Step；同時跨兩個維度就兩者都換算。任一必要值是 expression、缺席或 invalid，或換算失敗時，整次退回同名同型別的 raw 繼承，不安裝部分換算。普通 select 不改明確 command 的契約。詳細例子見 [Sweep 表示](../../lib/zcu_tools/gui/form_adapter/README.md#sweep-表示)。一般 union 政策不改 module ref／inline，也不改 Core 無 converter 時只切 selection 的規則。

**依據。**
- 2026-10-18，使用者裁決：「我認為Module間切換也可以採用類似的行為，這樣才一直，我甚至認為就單純的比對欄位名與形態來決定是否繼承值就好」。看過規則與例子後答「可以」。顧問同日釐清比較可編輯 Form 型別，ch／gen_ch 不搬移。原始討論未入庫，見 Git `482d1c79a`。
- 2026-10-29，將規則套用到一般生成 tagged-model union 是 designer 依夜間授權代裁，使用者未覆核。原始討論未入庫，見 Git `4d4a84c44`。
- 2026-10-29，使用者對「sweep 切換是否帶值換算」答「2B」，對「n↔step 是否也換算」答「2同意」。原始討論未入庫，見 Git `586f76a22`。

### 保存與還原

**規則。** `FormInput` 只保存 raw input 與明確的 selection overrides，不另存版本、型別 identity、domain metadata、callable 或動態 class。啟動讀檔、source reload 與 RAM switching 共用遞迴還原。按現行型別及形狀還原，保留合法 selection、原始葉值與唯讀欄位的存檔值。Host 顯示不再符合結構的 dropped 清單。

**例子。** 一份沒有 override 的草稿寫成 `{ "data": {…}, "selections": [] }`，真實欄位都放在 `data`，不與外層名稱衝突。有 override 時，每筆 `selections` 項目包含 `path` 和 `value`，路徑保留字串與整數。新增 model 欄位用 defaults；存檔 list 保留長度與位置。`FormState.restore_input`、同名與形狀遞迴、dropped 條件見 [Form Core README](../../lib/zcu_tools/gui/form/README.md)。

**邊界。** 還原不執行 converter、options hook 或 library 讀取。還原與 Reset 不經普通 mutation 的 owner input-change 檢查。一般驗證才檢查明確 override 的 hook membership 與 ref 的存在性及 family，保留錯誤草稿並擋 Run。沒有 override 的 raw branch 不因未列於選單而失效；已 inline 的原稿不要求原 ref 存在，也不從 library 重複複製。Host 不另包版本，舊 raw-only 或 versioned 格式不雙讀。JSON 損壞或格式不符由 caretaker 回報、以 defaults 開啟、不備份，之後照既有時機保存。Provider、其他 restore 故障仍傳播，read I/O 故障不當作 JSON 格式錯誤。失敗分工見 [[0063]]。

**依據。**
- 2026-10-18，使用者要求「還有版本？太複雜了。我需要更簡單的機制」，看過依名與形狀的遞迴規則後答「好，可以」，選擇固定 data／selections。原始討論未入庫，見 Git `ea0d14922`。
- 2026-10-19，使用者確認：「1同樣值不報錯。2我選存檔的版本，因為有些lock值可能是生成的。能不能執行是另外一回事」。能否執行由後續驗證判定。原始討論未入庫，見 Git `c499c06d9`。

### 固定具名元素與參與清單

**規則。** Core 提供一般 `list[Model]` 的描述、picker 與 indexed editing。`gui.form_adapter` 擁有具名元素的 identity 與 participants 政策，宣告 owner 指定 list path、名稱欄位與 participants path。新建與 Reset 從 capture 建表，重開和還原保留存檔的長度、位置與名稱，不與新 capture 比 identity。普通 mutation 不得改名、增刪或重排具名元素；非法 participants 仍可存為 invalid draft，由一般驗證擋 Run。

Mux 的 lab cfg 保留完整有序 `tones: list[ToneCfg]`。Capture 產生不可編輯的名稱，drive 類用 qubit 名稱，onetone 用 resonator 名稱。`participants: list[int]` 必須非空、無重複，使用完整清單的合法原始索引，其順序就是執行與資料欄的順序。名稱只供 label、錯誤與回覆，不供 recipe 或 runtime 查位置。所有 tones 都參與讀出，所以 readout 身分與通道必須完整。Drive 與逐 tone sweep 可為 None，只有 participants 必填。Open／Reset 先決定 participants，再為其 drive／sweep 填初值，其餘填 None。[Mux 參與名單解析](../../zcu_lab/v2/support/measure/README.md) 記錄 lab 如何把同次固定 capture 的 `$participants` 轉為索引，以及全部選取、未綁與外群名稱的處理。Edit、preview 與 Run 沿用這次 capture。修改全域 aliases 後須 Reset 或重新 Open，不自動刷新。MuxGroup 不增加欄位。

**例子。** 完整清單是 Q1、Q2、Q3 時，`[2, 0]` 先量 Q3 再量 Q1，只選 Q2 仍寫 `[1]`。一般編輯 participants 不重設 tone 值。新加入的 tone 若仍為 None，cfg validator 在原始 tone 路徑提示「填入 …」，移出的 tone 留原值。對需要名稱定位的表單，同一 resonator 可用 `R1#0`、`R1#1` 兩個不同名稱。重名時，宣告 owner 在建表報出重複名稱與原始 indices；實驗作者決定名稱，framework 不自動加後綴，也不改 [[0072]] 的 group schema。

**邊界。** `FormModel.check_input_change(before, after)` 以兩棵複本檢查普通 mutation 的完整候選。具名拒絕保留原 typed path，input、base、selections 與通知都不變；widget 和 RPC 顯示可恢復錯誤。還原與 Reset 不走此檢查。Picker label 與 Core 唯讀是各自的宣告，介面見 [固定元素宣告](../../lib/zcu_tools/gui/form_adapter/README.md#固定元素宣告)。掃描、drive 與相應 channel 檢查只遍歷 participants，非 None 的欄位值仍依一般規則驗證，不因未參與而停用。Framework 不解讀 mux group，也不改 program／raw 介面。

不宣告 name 或 participants 時，`FixedElements(list_path)` 只固定當前 inline 清單長度，不限制槽值、不以可編輯 label 建 identity，也不新增持久欄位或 `ElementIdentity`。交換槽值和逐槽位改值無法區分，不保證 resonator 身分。宣告可穿過巢狀 ref／inline module 定位 active inline list。合法切到 inline 時，由新清單設定長度邊界，不跨分支保存隱藏身分。還原與 Reset 依既有規則建立新邊界。Lab 從 mux group 決定 resonator 清單及初值，adapter 依原 Spec 生成 [typed model-list input](../../lib/zcu_tools/gui/form_adapter/README.md#native-model-lists)。

Workflow Apply 將 list 隨完整 `Form.to_cfg()` native model 交給 Engine，不攤平 preview leaves、不增訂 native index path。Engine 驗證、整體替換並增加 revision，journal append 成功後才發布，host 才接受 input／base。完整快照與不變的 step capture、iteration record.json 契約見 [[0062]]；不維護逐葉差異或舊格式相容層。

**依據。**
- 2026-10-18，使用者對傳複本與宣告檢查答「可以」，見 Git `227f6dab8`。
- 2026-10-18，使用者採參與清單而非逐項切換：「那設定也不用逐個toggle了，直接[Q1,Q2]這樣就好。」同日授權「可以，之後就交給advisor，我就不直接對話」。原話中的參與名稱後由下列索引裁決取代。具名元素採有序 list，以位置保存原始索引，不靠 mapping 順序或另建對照表；容器選擇是 designer 代裁，使用者未覆核，見 Git `3baf6eb0a`。
- 2026-10-18，designer 代裁名稱唯一性，使用者未覆核，見 Git `7e50be800`。
- 2026-10-18，使用者確認整個 workflow model 直接套用，不比對差異，並對日誌寫法答「甲乙都可以」。原裁決未附入庫討論，見 Git `361901e67`。
- 2026-10-20，使用者同意從 mux group 產生固定清單並逐欄編輯，原話是「可以」。Typed model-list seeds 與不帶 participants 的固定槽位模式是 designer 同日代裁，使用者未覆核；合法 ref／inline 切換不變，見 Git `2277294b9`。
- 2026-10-23，使用者裁定：「然後participants用list of index好了，不要用list of str」。對非 participant 可留空答「甲」。原始討論未入庫，見 Git `032915152`。
- 2026-10-23，使用者裁定 Open／Reset 初值：「這機制有點太複雜了，我的想法是預設participant以外的tone都初始化填None」。原始討論未入庫，見 Git `1c9c9c85a`。固定來源沿 [[0065]] 的 capture，接線見 Git `a7a1f309d`。

### Remote 與 recipe

**規則。** Model-field path 使用欄位名字串與 list 整數索引。Describe 提供位置和名稱，Agent 可按名稱找到位置，不另設名稱路徑。Owner 在單次唯讀操作回傳所有元素的 authored input、可用值與定向錯誤。MCP 不逐項讀取再拼接，也不維護另一份名稱排序表。

**例子。** Recipe override `[{"path": "tones.0.drive.gain", "value": 0.3}]` 由 MCP 編成 RPC 的 typed path `["tones", 0, "drive", "gain"]`，直接定位完整清單的第 0 項，與 participants 順序無關。一次讀全部元素也不會因一項無效而隱藏其他項。呼叫與回覆見 [Tab cfg 協定](../../lib/zcu_tools/mcp/measure/README.md#tab-cfg-協定)。

**邊界。** Recipe 接收完整原始索引的 participants，先套實驗值，再依序套 overrides，最後以 check_ready 查完整 Form／cfg errors。Overrides 不去重，不組原子 batch。Set 不執行 converter；select 與 command 用既有 Form owner 規則。省略的 Form 輸入用 Python KEEP 保留，明填 null 清空或停用 nullable 欄位。每個葉欄位保留 explicit／default／derived 來源。Actual、estimates 和 writeback candidates 用當次 Run／Load 的原始位置並附名稱，不被後來的 Reset 或目前 alias 改寫。不因顯示名稱與位置而授權寫回，目的地和 Apply 仍歸既有 writeback owner。

**依據。** 2026-10-18，使用者裁定：「既然核心是用index，那就用適合核心的方案，不要加一堆難維護的轉換邏輯」。同日依授權補定的 remote／recipe 細節屬 designer 代裁，使用者未覆核當時的整套細節。原始討論未入庫，見 Git `3baf6eb0a`、`be4b9039a`。2026-10-23 的 participants 索引裁決見[上一節](#固定具名元素與參與清單)。後續 logical 路徑與 overrides 的裁決範圍見 [logical cfg 路徑與 overrides 討論](../discussions/2026-11-20-logical-cfg-paths-and-overrides.md)。

### 模組分工與可執行依賴規則

**規則。** `gui.form` 擁有中立狀態與安裝，`gui.form_adapter` 擁有生成、領域 converter 與共用繼承宣告，`gui.widgets.form` 讀 description 呈現 picker 和選單。Host 擁有 capture、guard、生命週期與提交，remote 投影 owner 的結果，MCP 只依賴公開 remote 契約。Lab 擁有 mux cfg、來源關係、Open／Reset 的 participants 與 tone 初值和實驗 recipe 接線。實驗初值政策見 [[0073]]。

**例子。** Measure、workflows、Inspect、New Entry 與 writeback 共用編輯機制。各宿主仍以自己的 Apply／Cancel、Run admission 與關閉規則管理編輯結果。

**邊界。** 不新增跨 package 依賴方向，不放寬 `ignore_imports`。沿用 `form-core-no-ui-policy`、`form-adapter-no-host`、`form-editor-no-host` 與 C13 `mcp-remote-only`。C15 保護實驗 core 不依賴 GUI，C16 禁止 framework 反向依賴 lab。分層與無循環沿用 C8、C9。

**依據。** 既有模組邊界見 [[0072]]，各依賴規則見 [`.importlinter`](../../.importlinter)。本篇採用的分工見 Git `3baf6eb0a`，未附獨立的入庫討論。

## 代價與不採用的方案

沒有 converter 時仍要保留原有的子表單，所以不能只存 selection。若可信宣告不足以還原合法草稿，拿具體反例重新裁決，不序列化執行時 schema。

一般驗證若阻擋 Run，使用者需修正草稿或 Reset。固定 list 不提供任意容器編輯器，command 不提供參數表單，picker 不另設 reactive label 系統。

Framework 交付通用能力與既有 hosts 遷移。Mux 實驗 cfg、lab recipe 及接線另行交付。FTF、硬體操作、任意動態 schema、legacy 雙讀與 program／raw 格式變更不在本設計內。
