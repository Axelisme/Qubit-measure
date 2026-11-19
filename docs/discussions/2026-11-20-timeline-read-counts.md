---
date: 2026-11-20
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/v2/timeline.py
  - lib/zcu_tools/program/v2/_timeline_macros.py
  - lib/zcu_tools/program/v2/read_axes.py
  - docs/adr/0083-program-macros-and-timelines.md
  - docs/adr/draft/control-flow-completeness.md
  - docs/discussions/2026-11-20-timeline-path-state.md
basis:
  - "repo@db00dbe6e：PathState 已落地；13 個 zcu_lab 呼叫端手動 set_reads_per_shot；read_axes、chain_cycle、chain_readout、rpi 各自比對路徑讀取數"
  - "QICK asm_v2 Trigger.preprocess 逐一累加 ro_chs[ch]['trigs']；QickProgramV2.set_reads_per_shot 直接覆寫該值"
  - "使用者 2026-11-20 的問答；原始掃描紀錄未入庫，必要數字與結論寫在正文"
---

# Timeline 編譯器推算每發 ADC 讀取數

## 結論

編譯器在 macro 包裝層記錄 QICK 每個 ADC 的 trigger 增量，存進 `PathState`，並在路徑會合處比對。Branch 的每個出口要求所有 case（含省略的空 case）讀取數相同。固定次數 Repeat 的出口等於入口加次數乘單輪增量。次數只在執行期知道的 Repeat，body 不能讀 ADC。編譯結束時，結果寫進 `ro_chs[ch]["trigs"]`。`set_reads_per_shot` 只核對，不符就報錯。規則本身寫在 [ADR-0083〈每發 ADC 讀取數〉](../adr/0083-program-macros-and-timelines.md#每發-adc-讀取數)。

## 問題

QICK 在 preprocess 時按原始碼累加 trigger：Branch 的各 case 全部相加，Repeat body 只算一次。這個值不是每發實際執行的讀取數，所以作者 build 後要手動呼叫 `set_reads_per_shot`。`zcu_lab` 有 13 個呼叫端這樣做。手動值和實際觸發不符時，硬體 buffer 形狀錯，CPU 模擬器要到 `finish()` 才發現。四個 consumer（`read_axes`、`chain_cycle`、`chain_readout`、`rpi`）各自重做「每條路徑讀取數相同」的檢查。

使用者問：「那你要怎麼確定每條路徑的readout次數一樣多？」提案是讀取數進 `PathState`、在會合點比對、自動寫入每發讀取數，排在 `PathState` 之後、跳轉之前。使用者回「可以」。

## 考慮過的選項

**會合規則。** 實作前的掃描找到兩個既有測試 program 的 Branch 出口讀取數不同，整發總數卻相同：

| 既有 program | 寫法 |
| --- | --- |
| `test_correlated_branches_keep_one_selector_choice_for_the_whole_path` | 同一 selector 的兩個 Branch 互補：第一個在 case 0 讀，第二個在 case 1 讀 |
| `test_unknown_branch_local_adc_tail_closes_at_runtime_with_constant_shell_wait` | 只有 case 0 讀，其他 case 不讀 |

| 選項 | 內容 | 代價 |
| --- | --- | --- |
| 1 每個 Branch 出口對齊（採用） | 每個出口逐 ADC 比對所有 case | 兩個測試改寫；互補寫法不再支援 |
| 2 逐條可行路徑比對整發總數 | 列舉 selector 相關的可行路徑，只要求整發總數相同 | 要追蹤 selector 之間的相關性，跳轉加入後路徑數會乘開；`zcu_lab` 沒有實驗使用互補寫法 |

使用者選嚴格規則：「3-1」（2026-11-20）。

**計數位置。** 在 timeline macro 包裝層取 QICK `trigs` 的前後差值，涵蓋原生 trigger、module readout，以及 QICK 內部 macro 送出、原始碼看不到的 trigger。另一種做法是沿用 `observe_readout_paths` 只看原始碼中的 Trigger，會漏掉內部 macro 的 trigger。保留一個窄的檢查：`shot_sums` 選到的 ADC，原始碼看得到的 trigger 數必須等於編譯結果；不等時報錯，因為 label 的讀取索引無法跨過看不到的 trigger 解析。這不是第二條路徑相等規則。

**`set_reads_per_shot`。** 保留這個 QICK 方法，值與編譯結果相同時不做事，不同時 raise `ValueError` 並列出兩邊的值。不採「照舊覆寫」，因為手動值會蓋掉缺少的 trigger。

## 被否決的選項與理由

- 逐條路徑比對整發總數：見上表；沒有使用方，且和之後的跳轉合併規則（label 的所有入邊讀取數相同）不一致。
- 繼續要求手動宣告：問題本身就是手動值無法保證和路徑一致。

## 先例與立場

- ADR-0083 的 Branch 規則（case 從同一入口開始、出口合併）：遵循，讀取數照同一方式合併。
- [PathState 討論](2026-11-20-timeline-path-state.md)：遵循，讀取數是 `PathState` 的一個欄位，合併規則留在各結構。
- CPU 模擬器 `Outputs.finish` 逐發比對實際觸發數與 `trigs`：保留，作為 raw ASM 觸發的第二道檢查。

## 落地結果

- 13 個 `zcu_lab` 呼叫端刪除手動宣告；改動前後逐一編譯的 15 次 build，自動值與原手動值全部相同，13 張離線 run 的圖逐 byte 相同。
- 四個 consumer 的路徑讀取數檢查刪除，改依賴編譯器；label、讀取索引與 ADC 設定的檢查保留。
- 被規則取代的 consumer 測試刪除；仍可編譯的既有測試改成每個 Branch 各自平衡，保留原主題。

## 留下的問題

- 直接以 raw ASM 送出的 ADC 觸發不經此記帳，只靠模擬器檢查。
- 跳轉加入後，label 合併沿用同一條相等規則，見 [ADR 草稿](../adr/draft/control-flow-completeness.md)。
