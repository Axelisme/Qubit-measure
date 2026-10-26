**讀者：** framework-dev、agent。辨識移除 acquisitions 前後的 run 與 workflow 文件，判斷歷史欄位的讀取與遷移方式。

# 移除 acquisition facts

日期為 2026-10-26，依 ADR-0062 的使用者裁決日期記錄。實作以 `ac79d4883` 合入 main，Git 的 +0800 日期為 2026-10-27。範圍是 native run 的 `docs/metadata`、workflow journal 的 operation、commit `record.json` 的 operations、RPC operation 投影，以及 workflow `summary.json`。Container 版本與量測陣列不變。

## 改前形狀

RunMetadata 與 OperationFacts 都有 `acquisitions` 有序清單。每筆記錄 requested_rounds、completed_rounds、attempts、early_stopped、stop_value 與 retry_reasons，可另帶 simulated_physical_s。Operation 的 attempt 是 operation 序號，清單內的 attempts 才是 acquire 呼叫次數，包含失敗。

以下是 `ac79d4883^1` 的 [metadata 測試](../../tests/experiment/test_record_metadata.py) 中 `test_native_acquisition_shot_time_round_trips_and_omits_absence` 在 seconds=94.0 時產生的 `docs/metadata` 節錄。Snapshot、provenance、device_snapshot 與時間欄位省略：

```json
{
  "run_id": "20261006T010000Z-a1b2c3",
  "experiment": "test/metadata",
  "completion": "unknown",
  "acquisitions": [
    {
      "requested_rounds": 10,
      "completed_rounds": 10,
      "attempts": 1,
      "early_stopped": false,
      "stop_value": null,
      "retry_reasons": [],
      "simulated_physical_s": 94.0
    }
  ]
}
```

Workflow 的 `operation_finished` journal 行將這個清單放在 `operation.acquisitions`。Commit 信封則放在 `operations[i].acquisitions`，RPC 也投影相同清單。Workflow summary 另有 `early_stops` 與 `acquire_retries`，分別計算已交付 facts 中 early_stopped 為真的筆數，以及 attempts 大於 1 的筆數，並非 extra attempts 的總數。

## 改後形狀

新 RunMetadata、OperationFacts、journal、commit 信封與 RPC 都不再產生 `acquisitions`。同一份 metadata fixture 現由 `test_native_load_ignores_retired_metadata_without_rewriting` 驗證，對應節錄為：

```json
{
  "run_id": "20261006T010000Z-a1b2c3",
  "experiment": "test/metadata",
  "completion": "unknown"
}
```

其他歷史事實仍保留。Operation 繼續記錄 kind、iteration、attempt、started_at、finished_at、device_snapshot、outcome、reason、stopped 與 name。Workflow summary 移除 `early_stops`／`acquire_retries`，仍保留 point 統計、physical_recoveries 與 stage durations。

ProgramBuilder 改以執行日誌記錄 acquire retry 與 data-driven early stop。每次 retry 寫 warning，包含原因與下一次 attempt number。Early stop 寫 info，包含 stop value 與完成 rounds。這些日誌不再進入持久化的 operation facts。

## 為什麼改

Workflow 作者可以使用 ProgramBuilder，也可以提供符合 run 簽名的自訂 callable。框架收集的 acquisition facts 只覆蓋 ProgramBuilder 路徑，無法代表 callable 內部的全部工作。將這些 observations 當作通用 operation 事實，會讓保存與 RPC 依賴某一種執行工具。

使用者於 2026-10-26 裁決：「是作者的自由，框架只提供工具。」並要求：「我想把AcquisitionFacts機制刪除或取代」。ADR-0062 另保留職責邊界：「框架不應該知道run裡面是跑acquire或是其他」。因此框架保留 operation 的完成、失敗與時間事實，ProgramBuilder 用日誌說明自己的 retry 與 early stop，作者自行決定科學結果與保存內容。

## 如何辨識

檢查 native 的 `docs/metadata.acquisitions`、journal 的 `operation.acquisitions` 或 commit／RPC 的 `operations[i].acquisitions`。出現該 key 表示含有退役欄位，不應再將它當成現行 operation 契約。歷史 summary 中的 `early_stops`／`acquire_retries` 也是舊摘要的線索。

沒有該 key 不能單獨證明檔案是新產生的，較早的歷史記錄也可能缺欄。Container 仍是 zcu.container major 3，這次沒有升版。不要從 Container 版本或空清單推斷 acquire 次數，也不要把 operation attempt 解讀成 acquire retry 次數。

## 遷移

不遷移，讀取容忍。Experiment metadata reader、workflow journal 與 saved record reader 載入時忽略舊 `acquisitions`，不轉換、不寫回原檔。缺 key 也可載入。[Workflow reader 測試](../../tests/experiment/workflows/test_completed_facts.py) 與 metadata 測試涵蓋空清單、部分歷史物件與字串等舊值，讀取不要求重新驗證退役 observations。

需要調查舊 acquisition observations 時，保留原始文件直接審閱。現行 typed metadata 與 RPC 不再提供它們，也不從其他欄位補造舊 summary 計數。

## 相關 ADR

- [ADR-0062](../adr/0062-experiment-workflow.md#capability-與-driver-limits) 記錄作者自由、operation facts 與 ProgramBuilder 日誌的分工。
- [ADR-0063](../adr/0063-persistence-ownership.md) 區分量測完成、保存與各類文件的 owner。
- [ADR-0080](../adr/0080-datafile-generic-container.md) 定義未變更的 Container 格式，experiment 擁有 metadata 的語意。
