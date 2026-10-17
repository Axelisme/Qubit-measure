---
status: accepted
---

# ADR-0076 — Device 活動通知與 run 後讀回

**Last updated:** 2026-10-18

關聯 [[0062]] 的 acquisition capture、[[0067]] 的 Session／State 分工，以及 [[0066]] 的 operation 生命週期。保存權威見 [[0063]]，分層見 [[0072]]。

## 問題與決策

Experiment 直接透過公開 driver 操作儀器，不經 Device dialog 的 Apply 路徑。只觀察 Apply 無法辨認 experiment、MCP 或 workflow 實際操作哪台儀器。活動通知由 device 的共同操作入口發出，host 負責投遞，Device dialog 負責呈現。活動通知不攜帶數值，run 後另做背景讀回。

使用者於 2026-10-18 提出：「由環境注入提供儀器給實驗操作，然後基本儀器類加入事件hook接口讓host注入」。使用者接受事件只帶活動資訊、通知失敗不中斷量測，以及標記由 GUI UI/UX 負責，回覆為「1a認同,2可以，這是gui的uiux負責，3可以 4可以」。

同日使用者限制範圍：「請參考我的設計理念與分工明確的脈絡，我沒提到的請不要過度工程，沒想到的請不要猜」。本設計不加入未裁定的活動追蹤或硬體控制機制。

## 事件來源與失敗隔離

Device 層的 `device_operation` decorator 是活動事件的單一來源。受它保護的 driver 操作不因呼叫者是 dialog Apply、experiment、MCP 或 workflow 而另建通知路徑。Device 層定義 sink 介面，不 import GUI，也不理解 flux／RF 角色。

活動事件只帶 operation 與 started／finished／failed，不帶寫入值。Device 層攔下並記錄 sink 例外，不讓通知失敗中斷量測，也不改變原本儀器操作的成功或失敗。

活動通知保留 driver 既有的操作互斥、巢狀操作及 registry close 契約。事件只供觀察，不接管硬體控制、取消、lease 或 operation settlement。這些責任仍依 [[0066]] 由各 owner 處理。

## Sink 注入與拆除

環境是建立並持有 `DeviceManager` 的一方。GUI 的環境是 session，notebook／腳本的環境是呼叫者自身。環境在建立 `DeviceManager` 時提供可省略的 sink。未提供 sink 時，既有 notebook／腳本行為不變。

Registry 在 `register_device(name, device)` 時綁定裝置名稱，將 sink 安裝到 driver，讓 host 能辨認事件所屬的儀器。Registry 在 `drop_device`／`close_device` 時拆除 sink，生命週期仍遵守既有 registry 契約。Host 不逐台重做註冊與拆除配對，各 driver 建構子也不承擔 host sink 注入。

使用者於 2026-10-18 確認環境定義與由 registry 注入的方案：「你的理解沒錯， 而目前需求看起來A即可」。

## 持鎖投遞與 GUI 呈現

持鎖期間可以同步呼叫 sink，但 sink 只能把事件排進 owner thread 的佇列後立刻返回。Sink 不執行 GUI 工作，也不等待任何 thread。GUI host 將事件交回主執行緒後才更新呈現。

這是持鎖規則中僅限非阻塞投遞的例外，不允許在鎖內同步執行或等待 GUI 回呼。Sink 介面的 docstring 與 device README 明定此契約。顧問於 2026-10-18 裁定這個鎖邊界，避免為了把通知移到鎖外而改變 decorator 時序。

Device dialog 擁有「運作中」標記的呈現與時序。初始做法在最後一次 finished 後保留約 1 秒，合併後更新。後續 UI/UX 打磨由 GUI 負責，不改變事件來源。標記不是 driver busy admission 或 operation lease，呼叫者不能用它決定能否操作硬體。

## 活動資訊與數值讀回

活動事件不把寫入目標推定為儀器實際值，也不在每次 setter 或 sweep 的每一步新增硬體讀取。Run 結束後，host 在背景讀取一次已連線儀器，由 State owner 將讀回結果寫入 cached DeviceInfo。Device dialog 透過 State 的觀察投影顯示讀回值。

Run 前 acquisition snapshot 與 run 後 cache 讀回的用途不同。Run 前 capture 在 experiment callback 前固定硬體證據，讀取失敗仍依 [[0062]]、[[0063]] 拒絕執行，不以 cache 補值。Run 後讀回只更新 GUI 觀察值，不改寫已保存的 provenance 或 `RunMetadata.device_snapshot`，也不把 cache 變成下一次 Run 的控制命令。

[[0062]] 的 host capture 不更新 GUI State，適用範圍是 run 前 acquisition capture，不禁止 run 後的獨立背景讀回更新 cache。[[0067]] 的 State 提交仍由單一 owner loop 序列化。`DeviceService` 仍控制 driver 與 in-flight 工作，觀察需求不把 driver、worker 或 lease 的控制權交給 State。

## 未採用的做法

不在每個 driver 維護 operation 到 Info 欄位的映射。那會把 GUI 的數值投影規則放進儀器操作介面，還需要處理設定目標與實際讀回的差異。

Experiment 繼續直接使用 driver，不改 experiment 層，也不在 RunContext 包 proxy。Proxy 只涵蓋 run 路徑，無法作為所有呼叫者的單一事件來源。

不解析 cfg 預查裝置名稱，也不把全部已連線儀器一律標成運作中。標記反映實際 driver 操作，不推測實驗可能使用哪些儀器。

不另加 alias fan-out、新 registry 或通用 telemetry 機制。Sink 安裝與拆除由現有 registry 負責，host 不再維護第二套逐台配對。

## 分層與依賴契約

依 [[0072]]，`device` 是能力模組，`experiment` 是實驗框架，GUI host 是前端。Experiment 使用 device，GUI host 使用較低層的公開介面。Device 定義 sink，GUI host 提供實作，不因 runtime callback 反向 import GUI。

這些方向由既有 `.importlinter` contracts 涵蓋，不新增依賴例外。

| Contract ID | 本設計的適用範圍 |
| --- | --- |
| C8，`layers` | 前端依賴 experiment／device，experiment 依賴 device；device 不反向依賴前端或 experiment。 |
| C9，`top-level-acyclic` | host 投遞與 device sink 不形成頂層 package import 循環。 |
| C1，`experiment-no-gui` | experiment 不為活動顯示依賴 GUI。 |
| C2，`session-no-app` | 共用 session 不反向依賴各 app 的 run policy 或呈現。 |

本篇記錄設計契約，不代表活動通知、GUI 呈現或 run 後讀回的程式碼已通過驗證。
