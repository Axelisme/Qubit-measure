---
date: 2026-11-17
status: accepted
---

**讀者：** framework-dev、agent。查閱 Engine 錄製 buffer 的生命週期與逐輪事件取捨。

# Engine capture 的逐輪選路事件

## 結論

`simulate.evolution.CaptureBuffer()` 保存一次或多次 `Engine.evolve` 的輸入與實際選路。
每次呼叫產生新的隨機 `capture_id`，每個 batch 各留一筆 `FrozenRecord`，含空 batch。
Engine 只記憶體錄製，不寫檔。呼叫端在返回或拋錯後讀取 `records`。
公開欄位見 [Engine capture](../../lib/zcu_tools/simulate/evolution/README.md#engine-capture)。

## 調查與既有邊界

既有 `Engine.inspect_dispatch` 使用正式 classifier，但它不演化 prefix density，也不跑 timeline worker。
查詢會為候選計算 `scheduling_cost`。把查詢再跑一次，不能說明先前 evolve 當次做過什麼。
`_dispatch` 已有 guard 決定的觀察位置；scheduler 在 worker 重新分類 deferred 列；prefix resolver 用普通分類與排程求代表 timeline。
錄製沿這些執行位置觀察，不另跑 guard、成本計算或查詢。

[ADR-0078](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution) 把公開邊界放在 package root，dispatch 與 kernel 保留內部責任。
同份 ADR 的[凍結數值輸入契約](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)允許 opaque metadata，不讓容器解讀場景或選 solver。
本設計沿用這兩項先例。`CaptureMetadata` 與其事件型別供 producer 和 reader 共用，放進原有 metadata，不改 codec 格式。
`simulate` 仍不 import 其他 `zcu_tools` package，也不建立通用 collector 或 callback 協定。

## 為什麼保存事件，不只保存最終 family

Caller 可能把一列延後，worker 才接受或拒絕候選。
若只覆寫最後 family，失敗在 worker 之前時就無法分辨「未試」和「已拒絕」。
因此每列每輪各留事件，`round` 區分 caller 與 worker，`family=null` 表示該輪還沒有選中。
候選保留首次提供順序與該輪最後的 guard 決定，沿用四個 `GuardDecision` 值。
結構上沒有提供的 family 不出現在候選內。排序成本和 guard 數字不複製進錄製資料。

共用 prefix 只在代表 timeline 執行一次。
`prefix_runs` 記代表 group 列與全部採用者，含跨 batch 座標。
前綴事件只屬於代表 run，不為每個採用者虛構重跑。
事件先列各 prefix run，再列主體，每輪按輸入 point 與 segment 排序。Worker 執行順序不影響輸出順序。

## 為什麼 ID 屬於呼叫，buffer 可以累積

一次 acquire 可以呼叫多次 evolve。Buffer 由呼叫端持有，跨呼叫追加。
每次 evolve 的新 ID 把 batch_index 限定在該次 batches，名稱使用 `<capture_id>-<batch_index>`。
`records` 回傳 tuple 快照，metadata dict 不深度凍結，呼叫端不可改寫。
同一 buffer 並行傳入另一個 evolve，入口直接拋 `ValueError`，不等 Engine lock。

公開 `CaptureBuffer.recording(source, batches, budget, families)` 由 buffer 自己管理鎖、凍結、完整性標記和事件排序。
Engine 進入這個 scope，取得 metadata list 來寫事件，不從外面讀 buffer 的私有狀態。
其他呼叫端應把 buffer 傳給 evolve。Python 也允許直接進入 scope，但那只凍結輸入，沒有選路事件，正常退出仍標 complete=true。
這個 method 保留一個生命週期擁有者，避免另建 state 容器或放寬 private-usage 檢查。

凍結逐 batch 追加 `complete=false`，全部執行正常返回才改 true。
凍結失敗只保留先前已建紀錄，數值失敗或取消保留已觀察事件。原例外往外傳。
這使呼叫端在失敗後仍能保存有用輸入，不把部分結果宣稱成功。

## Segment shift 與重建

目前 Engine 只從 `batch.segment_shifts` 註冊 correction。
既有 `physical_input` 保存這些完整 batch 欄位，`read_physical_input` 會重建 `SegmentShift`。
Source 的註冊表本身不序列化，不能把兩者混為一談。
因此目前不增加無法重現旗標。未來若加入不經 batch 的註冊入口，該改動必須同時加標記。
錄製沒有 reference error，也不承諾以新 guard、history 或 budget 重跑時選中同一 family。

## 品質門檻的限定取捨

`Engine.evolve` 的原有六個參數與新增 `capture` 各自獨立。
把它們分組只會新增一個單一使用者的 options 型別。
因此保留七個參數，不加 suppression、不改設定、不用 kwargs 或 decorator 隱藏簽名。
只接受此函式新增的 `PLR0913` 一項為非阻擋門檻差異，其他診斷仍須修正。
Ruff 與 gate 仍會回報失敗，交付不得稱全綠。再次加參數或重組控制時重新檢視。

## 驗證與未做事項

公開 `Engine.evolve` 的小模型測試驗每 batch 的錄製、追加、metadata、guard 事件和 telemetry 對照。
同一接縫驗 deferred、跨 batch prefix 採用者、失敗與取消、並行入口拒絕、segment correction 重建。
注入的合成 kernel 對照有無 buffer 的 guard 和 cost 呼叫次數，確認錄製不增加這兩種工作。
寫檔策略、環境的 opt-in 設定和 analyzer 不在這次 Engine 改動內。效能與 reference 精度也不在本次驗收內。
