---
date: 2026-10-29
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/acquisition.py
  - lib/zcu_tools/program/v2/simulation/README.md
  - lib/zcu_tools/experiment/ftf/README.md
  - zcu_lab/v2/mux/t1/core.py
  - zcu_lab/v2/two_qubit/guide.md
  - docs/discussions/2026-10-29-readout-classifier.md
basis:
  - "repo@c30bf2bbd；qick 0.2.436（pyproject.toml 釘住版本）"
  - "兩份唯讀可行性調查與設計輸入為未入庫的工作檔（2026-10-29），結論與必要事實寫在正文"
---

# Heralding：取數層 post-selection，不做 FPGA active reset

## 結論

1. Heralding 在 program 的取數層做 post-selection。每發 shot 先做一次 herald 讀出；每個 round 收完後，host 用給定的圓判斷 herald 讀出，只平均通過的 shot。跨 round 依通過數加權，結果等於所有通過 shot 一起平均。不存逐發資料。
2. 不做 active reset，也不做 FPGA 即時判斷。
3. 上層新增實驗組 `heralding/*`，組織方式比照 `mux/*`。先從單 qubit 的 pulse readout 版本做起，再做 mux 版本。
4. 單 qubit 實驗中，每顆 qubit 只看自己的 herald；雙 qubit 序列要所有參與的 qubit 都通過。
5. 框架用 program accessor 提供每點的通過比例；`acquire` 的回傳形狀不變，herald 讀出不出現在回傳裡。heralding 實驗的 Result 存同網格的 `pass_fraction`。
6. 雙 qubit 序列改由取數層依各 tone 的分類器逐發分成四態，再對通過的 shot 平均四態機率，不再存逐發 IQ。
7. Herald 用哪個圓，取自 readout cfg 的 `classifier`（見 [readout classifier 的討論](2026-10-29-readout-classifier.md)）；herald 與最終讀出共用同一個 readout module。

## 問題

FTF 模擬的雙 qubit 序列（conditional phase 等）受熱布居影響：Pe 約 0.45 時，conditional-phase 的對比只剩約 1%。需要讓 qubit 從 g 開始。使用者先提出：「heralding需要完善program的數據即時處理，或是後處理，總之不能只是單純的平均了」。

另一個起因來自模擬器的中途讀出。r1 的標準讀出設定（7079.70 MHz、gain 0.05、1.5 µs）用在中途 herald 時，cavity drive 的 Stark shift 反轉能階順序，被 dispersive validity guard（`lib/zcu_tools/simulate/evolution/secular.py` 的 `ShiftOrderingError`）拒絕。最終讀出不演化 qubit，所以先前沒碰到。

## 考慮過的選項與比較

第一輪 advisor 提的四個問題：

| 問題 | 選項 | 使用者的回答 |
| --- | --- | --- |
| H1 即時處理或後處理 | A 逐發後處理；B FPGA 即時判斷 | 「H1的A只能用來測試可行性，但不能普遍使用，因為儲存過程中每個rep的點的記憶體負擔太大了，最後結果要是是B」 |
| H2 存什麼 | 逐發 I/Q 或只存平均與統計 | 由 H1 推得：只存平均與 herald 統計 |
| H3 判斷邊界 | singleshot 校準的旋轉直線 | 「我傾向使用給定圓作為分界，但你說的直線也可以」 |
| H4 介面 | 時域實驗表單加 herald 選項 | 「理想上希望可以是一個Reset模塊」 |

依 H1=B 做的可行性調查，找到 FPGA 回饋式做法的阻擋：

- QICK v2 有 `read_input`、`cond_jump`、`read_and_jump`，直線門檻有現成路徑；但 tProc 的 ALU 沒有乘法。圓判斷要用非同步的 ARITH 週邊，只回低 32 bit，而且暫存器裡是未除以長度的積分和，換算後平方容易溢位。
- 讀出結束到暫存器可用的延遲，QICK 與 repo 都沒有數值，只能上機量。
- repo 的 v2 bitfile 只有 avg buffer 0–7 接進 tProc，PFB 輸出 6、7 沒接。
- 每發的讀出次數在編譯期固定（QICK `Trigger.preprocess` 累加 `trigs`），retry 到通過會和 streamer 的固定筆數衝突。
- 模擬器照 Module 類別逐一解讀，每個 sweep 點只有一條 timeline，明確不支援資料相依 feedback（`lib/zcu_tools/program/v2/simulation/README.md`、`lib/zcu_tools/experiment/ftf/README.md`）。要支援是架構級改動。

同一次調查也找到一個改變判斷的事實。QICK 的平均本來就在 host 做。`finish_round` 把一整個 round 每發 shot、每次讀出的積分 I/Q 收進 host 記憶體的 `acc_buf`，`_average_buf` 才平均。所以現行的一般量測，每個 round 本來就持有完整的逐發資料。在取數層依 herald 篩選再平均，額外記憶體只有一次 herald 讀出，例如 101 點、1000 reps 的 T1，每個 round 從 1.6 MB 變 3.2 MB，而且不存檔。

第二輪 advisor 把兩條路並排比較：

| | 取數層 post-selection | 類 active reset（FPGA 回饋） |
| --- | --- | --- |
| 圓判斷 | host 浮點，精確 | ALU 無乘法，需 ARITH |
| 吞吐 | 丟掉未通過的 shot；Pe≈0.45 的兩顆 qubit 只留約 30% | 不丟 shot |
| 硬體未知數 | 只多一次讀出 | 延遲、接線、retry 都要上機量 |
| 模擬器 | 中途 projective readout 已支援，herald 與最終讀出共用同一條逐發 outcome 歷史 | 架構級改動 |

使用者的回答：「那不做active reset，讓底層支援post-selection，上層使用新實驗組，比如heralding/*，類似mux」。

第三輪的範圍問題：

- Q1 多 qubit 通過判斷：採建議（單 qubit 各看自己；雙 qubit 全部通過）。
- Q3 只做 mux 或也做 pulse readout：「我反而想先測試在單qubit，因為比較簡單，B」。
- Q4 雙 qubit 序列：現有的 chevron、CZ 振幅、條件相位存逐發 IQ，分析時才分成四態。原因是現行取數層只接受一組純量的 g/e 中心並套用到所有 ADC，輸出每個 ADC 的邊際比例，沒有兩個 tone 之間的逐發配對。使用者：「怎麼會逐發存？？為什麼要實作成這樣？本來就該用分類機制」。
- Q5 通過比例：使用者先說「我不確定要不要存分類比例，感覺acquire結果沒地方放這資訊」。advisor 補充：框架用 accessor 提供，存不存由實驗決定。使用者選 B，存每點的 `pass_fraction`。

## 被否決的選項與理由

- 存逐發 I/Q、事後篩選：每個 rep 的資料存檔負擔太大；只保留給一次性的可行性測試。
- FPGA 即時判斷、active reset：延遲未知、ALU 無乘法、tProc 只接部分 buffer、retry 與固定讀出次數衝突、模擬器要架構級改動；取數層 post-selection 已能在不存逐發資料的前提下做到。
- Reset 模組：改為獨立的實驗組 `heralding/*`，既有實驗不加 herald 欄位。
- 在 `acquire` 內自動掃描 readout module 取得 herald 圓：改由 readout module 提供工具算出型別化規格、作者明確傳入（理由見 readout classifier 的討論）。

## 留下的問題

- FTF r1 讀出下，herald 的 gain 或長度要多小才不觸發 dispersive guard，現有紀錄沒有交換的能階與 margin，需要一次小規模掃描。
- Mux 的 tone 增益在宣告 generator 時固定，同一組 ADC 不能放兩個 MuxReadout；mux 的 herald 只能和最終讀出同功率。模擬時由量測流程挑一個在有效範圍內的讀出增益。
- 模擬端一旦啟用 herald，每個 round 都要產生逐發資料，成本沒有實測。
- 每發多一次讀出會減少 streamer 每次可傳的 shot 數，實機 `avg_maxlen` 未知。
- 模擬 herald 是能量基底的理想投影，沒有量測誘發躍遷，比真實硬體樂觀。

## 依據

- QICK 0.2.436：`qick_asm.py` 的 `finish_round`、`_average_buf`；`asm_v2.py` 的 `read_input`、`cond_jump`、`read_and_jump`、`Trigger.preprocess`。
- repo：`lib/zcu_tools/program/acquisition.py`（`SingleShotMixin`）、`lib/zcu_tools/program/v2/simulation/README.md`、`lib/zcu_tools/experiment/ftf/README.md`、`lib/zcu_tools/simulate/evolution/secular.py`、`bitfiles/qick_216_v2.hwh`、`docs/reference/tprocv2-asm-notes.md`。
- 調查與成本探針為未入庫的工作檔；本文已寫入結論與必要數字。
