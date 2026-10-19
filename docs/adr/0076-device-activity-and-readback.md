---
status: accepted
---

# ADR-0076 — Device 活動快照與 run 後讀回

**Last updated:** 2026-10-19，未連線 device 改名與 FakeDevice address 身分

關聯 [[0062]] 的 acquisition capture、[[0067]] 的 Session／State 分工，以及 [[0066]] 的 operation 生命週期。保存權威見 [[0063]]，分層見 [[0072]]。

## 問題與決策

Experiment 直接透過公開 driver 操作儀器，不經 Device dialog 的 Apply 路徑。只觀察 Apply 無法辨認 experiment、MCP 或 workflow 實際操作哪台儀器。GUI 需要各台儀器目前是否在操作或剛結束，不需要逐次活動的事件歷史。Driver 記錄活動狀態，DeviceManager 提供具名快照，GUI 定時讀取。活動快照不攜帶數值，run 後另做背景讀回。

使用者於 2026-10-18 接受活動狀態快照方案：「Device的建議採用」。此方案取代 sink 注入與活動事件投遞。同日使用者限制範圍：「請參考我的設計理念與分工明確的脈絡，我沒提到的請不要過度工程，沒想到的請不要猜」。本設計不加入額外的活動事件、queue 或硬體控制機制。

## Driver 狀態與 manager 快照

Device 層的 `device_operation` 是共同的活動記錄入口。Driver 在其中只為活動記錄更新兩個欄位，是否在操作中，以及最後一次操作結束的 monotonic 時間。活動記錄在持鎖期間只寫這兩個屬性，不呼叫外部 callback。Device README 的持鎖禁令不加例外。

`DeviceManager.activity_snapshot()` 回傳每個已登記名稱對應的活動狀態，每台一筆固定大小資料。沒有 consumer 時，活動資料也不增長。回傳資料以具名型別描述欄位與時間語意，不提供修改 driver 狀態的介面。快照不讀硬體，不攜帶寫入值，也不包含操作名稱、結果或事件歷史。不注入或取消注入 sink，不提供活動事件流或 queue。

既有 driver 操作互斥、巢狀操作與 busy fail-fast 不變。活動資訊只供觀察，不接管硬體控制、取消、lease 或 operation settlement。這些責任仍依 [[0066]] 由各 owner 處理。

## GUI 觀察與標記

GUI 在主 thread 定時讀取活動快照，初始約每 200 ms。Device dialog 擁有「運作中」標記的呈現與時序。初始做法在操作中或最後結束後約 1 秒保持標記，合併後更新。連續快速操作保持穩定標記，停手後熄滅。後續 UI/UX 打磨由 GUI 負責，不改變 device 的狀態來源。

Dialog Apply、experiment、MCP 與 workflow 經由公開 driver 操作時，都使用相同的活動狀態。GUI 不解析 cfg 預查裝置名稱，也不把全部已連線儀器一律標為操作中。標記不是 driver busy admission 或 operation lease，呼叫者不能用它決定能否操作硬體。

## 單名稱 registry

使用者於 2026-10-18 裁定：「我認為devicemanager不要支援同儀器異名比較好」。DeviceManager 不支援同一 driver 同時登記多個名字。`register_device(name, device)` 遇到已用其他名字登記的 driver 時，在入口 raise，不提交第二筆登記。只為 alias 存在的 close cleanup、批次 identity 分組及專用測試一併移除。不加入 alias 相容、重新綁名或 fan-out。

不同 driver 的登記與同名 overwrite 契約不變。與 alias 無關的並行 close claims、失敗保留與重試、批次部分成功、成功清理、同名 replacement 保護及鎖外硬體 I/O 仍保留。

顧問於 2026-10-18 細化 register 的拒絕條件。Register 使用既有 close claims，遇到正在 close 的 driver 時在入口 raise，即使原名字已被 drop 或 overwrite。這不新增 registry、鎖或追蹤結構，也不恢復 alias 掃描。`register_device` 的 docstring 寫明此契約。這是顧問對並行 close 邊界的細化，不是使用者原話。

### 未連線 device 改名

使用者於 2026-10-19 裁定：「1可以，2不用」。此裁決採用只允許未連線 device 改名的方案，並明確不自動更新 alias。DeviceDialog 提供改名入口。DeviceService 只接受沒有進行中操作的 `MEMORY_ONLY` row，連線中必須先斷線。同名請求也先檢查這些前置條件，符合後不做任何修改。

名稱沿用建立／連線時接受的範圍，拒絕空字串、全空白和既有名稱衝突，不另加字元語法限制，也不自動刪除前後空白。Owner 同次提交新舊鍵後發布集合變更，live value sources 與後續 preferences 保存使用新名。`MEMORY_ONLY` 沒有已登記 driver，不需要 DeviceManager 換鍵或新的 MCP rename RPC。改名不更新 `aliases.yaml`、已開 tab 或歷史 snapshot、samples、ledger。使用者自行調整 alias，已開 tab 保留固定輸入。

## FakeDevice 的 address 身分

2026-10-19，使用者裁定：「FakeDevice採用以address為分辨的單例模式」，並指定「存在FakeDevice的類容器裡」。只有顯式 address 進入 process 共享的 class 容器。同 address 取回同一個 driver，保留模擬綁定、單位與設定。省略 address 的 anonymous instance 沒有可比較的身分，每次獨立建立，不進容器。真實 driver 不受影響。

使用者說明上限「可以很大」，再修正「可能也不丟了，創建時報錯」。容器最多保留 1024 個 addressed instances，僅防止無限增長。滿額時拒絕新 address，既有 address 仍可取回，不淘汰。測試使用公開 isolation scope，結束時清空測試建立的容器並還原原 scope。

單名稱 registry 規則維持不變。同 address 的 driver 已連線時，第二個名稱的連線明確失敗。先斷開舊名再用新名連線，則取回同一實例。FTF source 以 source name 作 address，數值 owner 保留原 source 身分，不隨 GUI registry 改名。SimulationEnvironment 只保存可建模的 address metadata；composition 注入其 lookup，未知 type 或 address 明確失敗，不建立無作用的 FakeDevice。

## 活動資訊與數值讀回

活動狀態不把寫入目標推定為儀器實際值，也不在每次 setter 或 sweep 的每一步新增硬體讀取。Run 結束後，host 在背景讀取一次已連線儀器，由 State owner 將讀回結果寫入 cached DeviceInfo。Device dialog 透過 State 的觀察投影顯示讀回值。

Run 前 acquisition snapshot 與 run 後 cache 讀回的用途不同。Run 前 capture 在 experiment callback 前固定硬體證據，讀取失敗仍依 [[0062]]、[[0063]] 拒絕執行，不以 cache 補值。Run 後讀回只更新 GUI 觀察值，不改寫已保存的 provenance 或 `RunMetadata.device_snapshot`，也不把 cache 變成下一次 Run 的控制命令。

[[0062]] 的 host capture 不更新 GUI State，適用範圍是 run 前 acquisition capture，不禁止 run 後的獨立背景讀回更新 cache。[[0067]] 的 State 提交仍由單一 owner loop 序列化。`DeviceService` 仍控制 driver 與 in-flight 工作，觀察需求不把 driver、worker 或 lease 的控制權交給 State。

## 未採用的做法

不採用 sink 注入、取消注入或持鎖投遞。Sink 是否非阻塞只能靠呼叫者遵守紀律，driver 沒有必要為活動記錄在持鎖期間呼叫外部程式碼。

不建立 bounded queue。GUI 不需要事件歷史，queue 會多出取出與合併工作。若未來需要失敗紀錄或操作歷史，必須另行決定有上限的事件機制，本設計不預留。

不在每個 driver 維護 operation 到 Info 欄位的映射。那會把 GUI 的數值投影規則放進儀器操作介面，還需要處理設定目標與實際讀回的差異。

Experiment 繼續直接使用 driver，不改 experiment 層，也不在 RunContext 包 proxy。Proxy 只涵蓋 run 路徑，無法觀察所有呼叫者的操作。

## 分層與依賴契約

依 [[0072]]，`device` 是能力模組，`experiment` 是實驗框架，GUI host 是前端。Experiment 使用 device，GUI 透過公開 query 觀察活動。Device 不 import GUI 或 experiment，也不理解 flux／RF 角色。

這些方向由既有 `.importlinter` contracts 涵蓋，不新增依賴例外。

| Contract ID | `.importlinter` name | 本設計的適用範圍 |
| --- | --- | --- |
| C8，`layers` | C8: frontends over the experiment framework over capability modules | 前端依賴 experiment／device，experiment 依賴 device；device 不反向依賴前端或 experiment。 |
| C9，`top-level-acyclic` | C9: top-level packages form no import cycle | GUI 查詢與 device 活動狀態不形成頂層 package import 循環。 |
| C1，`experiment-no-gui` | C1: the experiment domain does not depend on the GUI | experiment 不為活動顯示依賴 GUI。 |
| C2，`session-no-app` | C2: shared session does not depend on apps | 共用 session 不反向依賴各 app 的 run policy 或呈現。 |

本篇記錄設計契約，不代表活動快照、單名稱 registry、GUI 呈現或 run 後讀回的程式碼已通過驗證。
