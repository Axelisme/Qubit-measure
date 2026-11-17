---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../code-quality.md
  - ../../lib/zcu_tools/simulate/evolution/README.md
basis:
  repo_commit: 9c169e6d9af99aa62685383c38282b9678bc4160
---

# Engine capture 的逐輪選路事件

## 結論

這次選擇由呼叫端持有 `simulate.evolution.CaptureBuffer()`，保存一次或多次 `Engine.evolve` 的輸入與實際選路。
每次呼叫使用新的隨機 `capture_id`，每個 batch 各留一筆 `FrozenRecord`，含空 batch。
Engine 只在記憶體錄製，不寫檔。呼叫端在返回或拋錯後讀取 `records`。
錄製沿實際執行路徑觀察，不另跑 guard、成本計算或查詢。

這篇保留當時的比較與取捨，不擁有決策。
公開欄位與生命週期由 [Engine capture](../../lib/zcu_tools/simulate/evolution/README.md#engine-capture) 說明。
Package 邊界與凍結容器的責任見 [ADR-0078](../adr/0078-evolution-module-boundary.md)。

## 問題

當時已有 `Engine.inspect_dispatch`，但它不能還原先前 evolve 實際做過什麼。
它使用正式 classifier，卻不演化 prefix density，也不跑 timeline worker。
查詢還會為候選計算 `scheduling_cost`，不能把再查一次當作零額外工作的錄製。

Caller 可能先把一列延後，worker 才接受或拒絕候選。
共用 prefix 則只在代表 timeline 執行一次，其他列沿用結果。
錄製需要分辨這些情況，也要讓呼叫端在凍結、數值演化或取消失敗後取得已建紀錄。

## 考慮過的選項與比較

### 沿用查詢邊界，觀察實際執行

調查時，`_dispatch` 已有 guard 決定的觀察位置。
Scheduler 在 worker 重新分類 deferred 列，prefix resolver 用普通分類與排程求代表 timeline。
因此選擇沿這三條執行路徑記事件，不重跑 `inspect_dispatch`。
查詢與執行的分工見 [Dispatch inspection](../../lib/zcu_tools/simulate/evolution/README.md#dispatch-inspection)。

[ADR-0078 的黑盒邊界](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)把公開入口放在 package root，dispatch 與 kernel 保留內部責任。
這次遵循此先例，公開 buffer 與共用 metadata 型別，不公開內部 dispatcher。
同份 ADR 的[凍結數值輸入契約](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)允許 opaque metadata，不讓容器解讀場景或選 solver。
這次也沿用此先例，把事件放入既有 metadata，不改 codec 格式。
依 [ADR-0072 的封閉子模塊邊界](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明)，`simulate` 不新增對其他 `zcu_tools` package 的依賴，也不建立通用 collector 或 callback 協定。

### 保存逐輪事件，而非只留最終 family

只覆寫最後 family 時，失敗在 worker 之前就無法分辨未試與已拒絕。
因此選擇每列每輪各留事件，以 `round` 區分 caller 與 worker，以 `family=null` 表示該輪尚未選中。
候選沿用四個 `GuardDecision` 值，保留首次提供順序與該輪最後的決定。
結構上沒有提供的 family 不列入候選，排序成本與 guard 數字也不複製。

Prefix 則記代表 group 列與全部採用者，含跨 batch 座標。
前綴事件只屬於代表 run，不為每個採用者虛構重跑。
當時選定的輸出順序是先列各 prefix run，再列主體，每輪按輸入 point 與 segment 排序。
這樣不會把 worker 執行順序誤當成輸入順序。

### ID 屬於呼叫，buffer 累積多次呼叫

一次 acquire 可以呼叫多次 evolve，因此選擇由呼叫端持有 buffer，跨呼叫追加。
每次 evolve 的新 ID 把 `batch_index` 限定在該次 batches，名稱使用 `<capture_id>-<batch_index>`。
`records` 採 tuple 快照，metadata dict 不深度凍結，呼叫端不得改寫。
同一 buffer 並行傳入另一個 evolve 時，在入口拋 `ValueError`，不等 Engine lock。

當時採用公開 `CaptureBuffer.recording(source, batches, budget, families)`，讓 buffer 自己管理鎖、凍結、完整性標記與事件排序。
Engine 進入 scope 後取得 metadata list 寫事件，不從外面讀 buffer 的私有狀態。
這保留了一個生命週期擁有者，避免另建 state 容器或放寬 private-usage 檢查。
其他呼叫端應把 buffer 傳給 evolve。Python 也允許直接進入 scope，但那只凍結輸入，沒有選路事件，正常退出仍標 `complete=true`。

凍結逐 batch 追加 `complete=false`，本次呼叫正常返回後才全部改為 true。
凍結失敗只保留先前已建紀錄，數值失敗或取消則保留已觀察事件，原例外往外傳。
這讓呼叫端取得部分紀錄，又不把部分結果宣稱為正常返回。

### 沿用 segment shift 的凍結資料

調查時，Engine 只從 `batch.segment_shifts` 註冊 correction。
既有 `physical_input` 保存完整 batch 欄位，`read_physical_input` 會重建 `SegmentShift`。
Source 的註冊表本身不序列化，不能把它與 batch 帶入的 correction 混為一談。
因此當時不增加無法重現旗標，重建條件由 [Engine capture](../../lib/zcu_tools/simulate/evolution/README.md#engine-capture) 說明。

### 保留七個獨立控制，照實回報門檻失敗

`Engine.evolve` 的原有六個參數與新增 `capture` 各自獨立。
把它們分組只會新增一個單一使用者的 options 型別，因此當時選擇保留七個參數。
不加 suppression、不改設定，也不用 kwargs 或 decorator 隱藏簽名。
當時限定只有 `lib/zcu_tools/simulate/evolution/engine.py::Engine.evolve` 的 `PLR0913` 計數 0→1 為非阻擋差異，其他診斷仍須修正。
Ruff 與 gate 仍回報失敗，交付不得稱全綠。
這是本次限定取捨，不是其他函式的先例，例外的審查規則見[程式碼品質](../code-quality.md#審查裁決)。

## 被否決的選項與理由

- 事後重跑 `inspect_dispatch`。它沒有執行 worker 或 prefix density，也會增加查詢工作，不能冒充原次執行紀錄。
- 每列只保存最終 family。這會抹掉 caller deferral，也無法表示停在 worker 之前的失敗。
- 為 prefix 的每個採用者各列一份執行事件。實際只有代表 run 執行，複製會虛構重跑。
- 在 Engine 直接寫檔。這超出既有演化邊界，也讓失敗後保存資料綁定 Engine 的 I/O。
- 合併參數或隱藏簽名以通過 `PLR0913`。這不減少呼叫端需要理解的獨立控制，還會新增型別或間接呼叫方式。

## 留下的問題

寫檔策略、環境的 opt-in 設定與 analyzer 不在這次 Engine 改動內。
錄製 overhead、單次錄製大小、效能與 reference 精度也不在本次驗收內。
錄製不提供 reference error，也不承諾在新 guard、history 或 budget 下重跑會選中同一 family。
若日後新增不經 batch 的 shift 註冊入口，該改動須同時加上無法重建的標記。
再次新增參數或重組控制時，也須重新檢視這次的門檻取捨。

當時選定以公開 `Engine.evolve` 接縫的小模型測試驗證錄製、追加、事件與 telemetry 對照。
同一接縫涵蓋 deferred、跨 batch prefix 採用者、失敗與取消、並行入口拒絕及 segment correction 重建。
合成 kernel 對照有無 buffer 的 guard 與 cost 呼叫次數，不直接測 `recording` 協作 method。
這些是驗證範圍，不是效能或數值精度結論。

## 依據

- 調查起點為本 repo commit `9c169e6d9af99aa62685383c38282b9678bc4160`。以上比較以該版本的查詢、分類、worker、prefix 與凍結 codec 為基礎。
- [Engine](../../lib/zcu_tools/simulate/evolution/engine.py)、[dispatch](../../lib/zcu_tools/simulate/evolution/_dispatch.py)、[scheduler](../../lib/zcu_tools/simulate/evolution/_schedule.py) 與 [prefix resolver](../../lib/zcu_tools/simulate/evolution/_prefix.py) 擁有執行路徑。[Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有公開介面說明。
- [ADR-0078](../adr/0078-evolution-module-boundary.md) 擁有 evolution 黑盒邊界與凍結數值輸入的責任分工。[ADR-0072](../adr/0072-module-boundaries.md) 擁有 `simulate` 的依賴邊界。
- [程式碼品質](../code-quality.md) 說明契約、可觀察驗證與限定例外的審查方式。本篇沒有使用外部來源。

## 後記

2026-11-17，使用者對「在 `Engine.evolve` 定義行加同行 `# noqa: PLR0913  # tolerated: <reason>`」答「B」。`Engine.evolve` 改用這個同行標記，理由寫在原地；只適用這一筆診斷，不改設定，PLR0913 不再回報。本文「不加 suppression」「Ruff 與 gate 仍回報失敗」兩句由此取代；保留七個獨立參數、不合併成 options 型別的判斷不變。
