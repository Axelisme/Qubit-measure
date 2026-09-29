# Cfg 資源公開契約與同步發布

狀態：設計草稿，尚未實作。本文細化 measure 優先的 cfg slice，不宣稱四個 app 已遷移。現況依 [ADR-0065](../0065-cfg-editing.md)、[ADR-0067](../0067-gui-application.md)、[ADR-0068](../0068-remote-transport.md)。[既有 cfg draft](cfg-editing-boundaries.md) 的 library conversion、writeback Apply 與其他 app 義務仍保留。本文的 measure slice 以同步 Valid／Invalid／Unavailable 取代該 draft 的 pending 描述；expression capture 也以始終保存 expression 取代舊 resolve-once direct value 目標。

## 問題

Controller 的轉送入口讓 caller 必須知道 editor identity、binding grammar 與 app service 的關係。GUI 與 remote 若各自判斷哪些欄位可寫，就會維護兩套驗證。逐筆修改 live cfg 也使 batch 失敗留下成功前綴，caller 無法以一份版本明確表達修改與執行依據。

目標是讓 caller 取得一個綁定資源的 editing handle。Cfg owner 隱藏 path 驗證、候選準備、解析與 publication；Controller 不再逐項代理這些命令。

## 領域詞彙

- Cfg resource 是有 identity、revision 與固定 definition 的可編輯資源。Tab owner 建立及撤銷資源；cfg subtab 只是它的一個 view。
- Cfg observation 是同一 revision 的 input、解析結果、狀態、diagnostics 與 source basis。它不是 live mutable tree。
- Source basis 是解析使用的本地來源版本集合，不是硬體即時讀取憑證。
- Accepted config 是指定 Valid revision 的固定 values 與 provenance。它不是可回讀來源的 editor，也不代表已取得 hardware lease。
- Capture 是寫入時固定 expression 中被 $ 標記的引用。未標記引用仍依原 expression 規則更新。

## 決策

### 一個 owner，兩種使用能力

`gui.cfg` 擁有 input、path、edit、observation 及錯誤型別。App resource owner 持有 lifetime，並提供來源與操作禁令。Qt 和 remote 共用 editing handle 的 observe、watch、edit、reset、refresh。Run owner 另外取得 accept 能力，不讓 frontend 先凍結值再提交給 Run。

Watch 一致交付初始 observation 與後續 publication。所有回傳值與 caller 傳入內容均隔離 mutable alias。View detach 不撤銷資源；close 後舊 identity 失效，重建同名 tab 也不能讓舊請求命中新資源。

### 可寫節點取代 caller grammar

Edit 接收有序的 path／value 清單。Path 使用名稱序列，root 為空序列。Leaf、sweep、reference 或 definition 明確支援的 aggregate 都可作寫入對象，不要求 caller 選 SetInput／SetSweep，也不提供 agent_edit 分流。

一般資料欄位不得以 __ 開頭。Editing codec 使用 __complex、__text、__expr；reference 的 __ref 表示 relink 或停用。普通 ref 仍是資料名稱。型別由 definition 或領域既有 discriminator 決定，不增加 __shape registry。Codec 只解碼意圖，target definition 才決定是否允許該模式。

Whole subtree 是局部輸入覆寫，省略欄位保留。領域原本允許的型別切換要求完整新輸入，edit 不根據省略欄位暗中混入舊值或補 defaults。Custom 切換在提交前建立完整候選的繼承政策見下一節。Relink 加 children 先換來源再套內容，不依 dictionary 順序。__ref=null 加 children 拒絕。修改 linked 內容解除沿修改路徑的相應來源連結，其他 siblings 不受影響。

### Custom reference 的 best-effort 繼承

切換至 Custom 時，共用 cfg 邏輯先依新 definition 建立完整候選，並 best-effort 繼承相容的既有輸入，再一次提交、解析、驗證與發布。這保留切換 waveform 後不用重填 length 的操作方式，不讓 Qt 擁有第二套繼承規則。

同名、同型別欄位可繼承。不相容或新增的欄位使用新型別初始值；新型別沒有的欄位捨棄。style 等固定欄位一律使用新 definition。不猜欄位名稱、不做單位或自動型別轉換，也不新增每型別歷史 cache。例如 Gauss → Arb → Gauss 不保證找回原 length，因為 Arb 沒有此欄位。

Expression 保留輸入式，由新候選正常解析，不直接信任舊解析結果。繼承後未通過值驗證時發布 Invalid，不偷偷替換成另一個值。同型別由 Library 改成 Custom 保留目前內容，只解除本層 linkage；nested reference 與 expression 保留自己的依賴。候選與舊值隔離 mutable alias，nested reference 的 linkage／override 資訊不能在複製時遺失。

建立完整候選與接收 edit 是兩個責任。普通 edit 仍拒絕不完整的跨型別輸入，MCP 不因 GUI 的切換便利性而獲得隱含繼承或補值。準備或提交的非預期故障仍保留舊 publication，best-effort 不是吞掉任意例外的許可。不為這項功能建立通用 migration framework。

### 同步候選與一次發布

Cfg resolver 只消費本地已發布來源，不做外部 I/O、不輪詢硬體、不處理 GUI events。Edit 在未發布候選上依序處理全部意圖，全部可接受才發布一次。成功同值命令也推進 revision。沒有成功前綴，沒有 Pending 或背景解析 completion。

Cfg owner 以獨立 input tree 為權威。先修改輸入，再解析修改完成後仍存在的內容；不先重建會自動求值的舊 field tree。完整 override 不依賴被取代的舊 source key，舊 expression 即將被取代時不求值。Partial reference write 或 range step 計算若需要現有內容，才依同一固定 source basis 取得必要資料。Binding 可以承擔最終解析，但不能藉 constructor／setter 同時控制 input mutation。

合法但未完成的輸入可發布 Invalid。必要來源未就緒或故障為 Unavailable。Edit／Reset 的非預期準備故障保留原 observation；source refresh 失敗則發布 Unavailable，不能繼續把舊值當 Valid。

同來源影響多個 cfg 時，先完成來源與所有受影響 cfg 的一致發布，再通知。個別 cfg 故障不撤銷新來源或其他 cfg。通知期間允許 read，拒絕 mutation、Run start 與 close，不隱含排隊。Subscriber 故障送診斷，不回滾已提交命令。

### 共用 expression 引擎

`gui.session.expression` 以 simpleeval 執行數學運算。App 不另外建立求值器。共用封裝限制可用函式、算子與數值大小，並把變數查詢接到來源 owner。支援 int、float、`1+1j` 等 complex literal 與算術；real-only caller 明確拒絕 complex，不丟棄虛部。

開放 `sin cos tan sqrt exp log log10 abs` 與 `pi e`。除 `abs` 外使用 math 的實數函式，不自動切換 cmath。`**` 為冪次，`^` 只沿用 simpleeval 原生 XOR，不改寫優先序或原輸入文字。函式、常數與被使用的來源名稱衝突時拒絕，不悄悄覆蓋。任意屬性、索引、方法呼叫、assignment 與多 statement 都不開放。

### Expression 逐引用固定

`{"__expr":"$device.flux_device.value + offset"}` 在寫入時只讀已發布的 device cache，固定被標記引用，保留 offset 的動態依賴。若固定值為 0.001，保存 expression 的形式為 `(0.001) + offset`，不是 DirectValue。

Capture 必須在當次完成。含 capture 的 expression 若無法解析完整結構或取得 typed literal，整批拒絕，不保存待未來解析的 $。無 capture 的未完成 expression 可作 Invalid 輸入保存。替換使用語法結構與 typed literal，每個替換位置保留括號，避免負值冪次或 complex 改變優先序。沒有 $ 的 expression 不因此重新排版。

不新增 __once 或 capture 持久型別。括號不是 provenance 標記，精確浮點 spelling 也不是產品保證。Editing wire codec 與既有保存格式分工，不藉此全面改寫 memento。

### 明確版本與錯誤

Remote 完整讀取回 `{cfg_id, revision}`。Id 為不透明字串，revision 為十進位字串。Edit 與 Run 必須帶這份依據，不隱含預讀、refresh、換版或 retry。只有 cfg guard 改用 explicit reference，auth 與非 cfg 資源 guards 不變。

非法 path、codec、mode 或 readonly 寫入屬 invalid input。Stale、撤銷、busy、通知重入與 capture 來源不可用屬 failed precondition。合法 incomplete input 是成功 publication，不是拒絕命令。非預期故障保留 cause／traceback，不降級為輸入問題。

Run 在同一有序接受點檢查版本與 app 條件、取得固定 values、登記 operation 及同 tab 編輯禁令。後續硬體啟動失敗由 operation 回報。Timeout 或 reply encoding failure 不表示命令未提交。

## 第一條接線與遷移範圍

Measure 的 `tab.get_cfg`／`tab.edit_cfg` 經 composition 注入的 tab cfg lookup 取得 resource-bound handle，直接 observe／edit，再投影完整 observation。不經 Controller cfg setter 或 editor_id 中轉。Lookup 只對應 identity，不複製 editing commands。

現有 `h_tab_set_cfg`、`Controller.cfg_editor_set_fields`、`CfgEditorService.set_fields` 是遷移位置。最小 seed 包含真實 read/edit caller 與 cfg owner 的公開契約測試，不能只放未使用的宣告。所有 tab writer 最終必須進同一 authority，不以鏡像雙寫過渡。接線尚未完整前，integration 中間狀態不可部署。

Library editor／writeback 不因使用同一舊 service 就改成 tab 契約。MCP 其餘工具、完整 Run／lifetime 及 Qt presentation 分後續票驗證。其他 app 的需求不因 measure 先行而刪除。

## 取捨

同步解析省去排程、取消與晚到結果處理，代價是來源取得必須在 cfg resolver 外完成。未來若需要昂貴解析，必須另作架構決策，不能悄悄加入 Pending。

Input-first 將輸入意圖與解析副作用分開，代價是 node writer 不能直接重用會求值的 field setter。Scalar parsing 與 range 算法抽成共用純規則，避免另建一份 grammar。不建立 dependency graph；只有來源依據一致且有實際效能需求時才另行考慮結果快取。

單一 cfg authority 讓 UI 和 remote 得到相同結果，代價是不能讓舊 mutable binding 與新 owner 同時寫入。遷移必須交代所有 writer，而不是只替換 remote handler。

Explicit revision 讓跨連線 caller 可以明確指定依據，代價是 caller 必須處理 Stale 並自行決定是否重送。這不提供 exactly-once，也不取消其他資源的 guard。

## 轉正條件

透過公開 cfg interface 驗證 batch 原子性、版本、Invalid／Unavailable、watch 順序、通知隔離、alias 隔離及固定 accept。Custom 切換另驗證 waveform 共同欄位繼承、新型別固定值、不相容欄位的初始值、expression 保留與重新解析、nested reference 狀態，以及失敗不改舊 publication。透過 source publication 驗證多 cfg 一致性。透過真實 request/reply 驗證讀寫同一 authority、版本衝突與錯誤映射。

Run、Qt 與 MCP 完成各自接線後，在共同 tree 驗證同一資源及固定執行資料。Owner、型別重複、私有存取、resolver 外部依賴與 method 宣告用直接 review。重構涉及的型別／lint 負面指標作 best-effort 清理，剩餘問題列明原因；不豁免必要契約或關閉規則。

完成上述實作與驗證後才更新現行 ADR。本 draft 不作為已實作證據。
