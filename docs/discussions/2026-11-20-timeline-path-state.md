---
date: 2026-11-20
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/v2/_timeline_state.py
  - lib/zcu_tools/program/v2/_timeline_macros.py
  - lib/zcu_tools/program/v2/_timeline_repeat.py
  - docs/adr/0083-program-macros-and-timelines.md
  - docs/adr/draft/control-flow-completeness.md
  - docs/discussions/2026-11-20-early-exit-and-control-flow-completeness.md
basis:
  - "repo@abec61d9b：Branch 與 Repeat 各自手寫存取狀態；同日已修 Branch 漏還原 unknown ADC 的缺陷"
  - "Møller & Schwartzbach, Static Program Analysis, https://cs.au.dk/~amoeller/spa/spa.pdf"
  - "MLIR LangRef（Blocks、block arguments）與 MLIR Data-flow Analysis 文件，2026-11 版"
  - "使用者 2026-11-20 的問答；原始工作紀錄與等價檢查檔未入庫，必要原話、數字與結論寫在正文"
---

# Timeline 共用的編譯期狀態：PathState 與 capture／restore

## 結論

Branch 與 Repeat 改用一個共用的 frozen 值 `PathState`，裝下編譯器在某個程式位置上、隨路徑而變的全部狀態。`TimelineBook.capture` 一次讀出全部欄位，`TimelineBook.restore` 一次寫回去。合併規則仍留在各結構裡：Branch 取最大或保留實際執行的 case，Repeat 把一輪的結尾帶到下一輪。

切換路徑用 capture／restore，不改成「每條 timeline 各有一份狀態」，也不改成「用指標切換目前路徑」。使用者接受這個理由，並要求寫進討論紀錄：「這個理由我接受，記得寫進discussion」（2026-11-20）。

這次是行為不變的重構。改動前後各編譯同一批 304 個 program，ASM 與 metadata 逐位元組相同。這批 program 包含既有的 Branch／Repeat 測試，以及 two-qubit RB 的 reference、interleaved 兩組。

## 問題

Timeline 編譯器依序走過每個 macro，一邊走一邊更新狀態：每條具名 timeline 結束在哪裡、每個 generator 和 ADC 上一個事件的時間戳、哪些值要到執行期才知道、哪些 register 已經定義過。下一個 macro 放在哪裡，由這份狀態決定。直線程式只有一份狀態。控制流程會從同一個位置分出幾條路徑，每條路徑都需要一份自己的狀態。

以 Branch 為例。兩個 case 分別播 40 ns 和 200 ns 的 pulse，進入時 `default` 結束在 1.00 µs。編譯器先 capture 入口狀態，再編 case 0 得到 1.04。接著 restore 回 1.00，編 case 1 得到 1.20。最後在出口 join：`align_timeline=True` 取最大，得到 1.20。如果漏了 restore，case 1 會從 1.04 開始編，算出 1.24。這是錯的，因為每發只會走一個 case。

改動前，這份狀態散在大約七個地方。Branch 和 Repeat 各自手寫要存哪些、還原哪些。同日修掉的缺陷就出在這裡：case 0 用了 `t=None` 的原生 trigger，ADC 結尾因此不可定位，但 Branch 在編下一個 case 前逐項還原時漏了 unknown-ADC 集合。結果 Branch 之後的 managed trigger 失去了重疊檢查。漏掉一項不會報錯，只會讓一個檢查失效。之後再加跳轉，同樣的手寫邏輯會出現第三份。

## 考慮過的選項與比較

這套做法對應編譯器理論的 abstract interpretation，也就是 dataflow analysis：每個程式位置有一份抽象狀態，每個 macro 是 transfer function，路徑會合時做 join，不可到達的位置記作 ⊥。PathState 就是這份抽象狀態。Branch 的兩種合併分別是 lattice join 與 SSA 的 phi。Repeat 跨迭代保留的狀態，對應 MLIR `scf.for` 的 `iter_args`。之後的 label 合併所有入邊，對應 MLIR 的 block arguments。

`PathState` 收的欄位：
- 具名 timeline 的結尾；
- QICK generator 與 ADC 的時間戳，以及存在執行期儲存的結尾；
- source／native 時間表達式與 unknown ADC；
- 進行中的 Repeat frame 記下的推進量與 guard；
- producer 已定義的 register 與 selection；
- register pin；
- 每個 ADC 的 QICK source trigger 計數。

已宣告的 module、waveform、register 配置不放進去，因為它們在每條路徑上都相同，只累加、不還原。判斷標準是「換一條路徑要不要還原」。

使用者問：「為什麼要設計成capture/restore，而不是每個timeline維護自己的PathState，這樣切換PathState就不用restore了」（2026-11-20）。回答有兩點。

第一，狀態屬於路徑，不屬於 timeline。Timeline 是同一次執行裡並行的時間線，例如 RB 的 F1、F2，用 fork 分開、用 join 會合。路徑是互斥的執行可能，例如 Branch 的兩個 case，每發只走一條。一條路徑裡有好幾條 timeline。很多狀態不屬於任何一條 timeline：
- 同一個 generator 被兩條 timeline 使用時，重疊檢查要同時看到兩者（ADR-0083：另一條 timeline 不能豁免此檢查）；
- unknown ADC 和讀取數屬於 ADC；
- 已定義的 register、selection 證明屬於程式位置；
- register pin 屬於整個程式的 register 配置。

第二，「每條路徑擁有自己的一份」本來就成立：每個 case、每條之後的跳轉邊都持有自己的 frozen `PathState`。真正的差別在於正在編譯的那條路徑放在哪裡。用指標切換，就得讓所有讀寫都經過 `prog.current_path`。但 QICK 自己的 macro 在 preprocess 時直接讀寫 `prog._gen_ts`、`prog._ro_ts`，trigger 直接遞增 `ro_chs[ch]['trigs']`，producer 也直接改 `prog._time_bounds`。要做到換指標，只能改 QICK 的 macro，或用 property 攔截 QICK 的屬性。Restore 只需要在一個地方把值複製回這些位置。

成本與風險：
- Restore 的次數等於 case 數加上跳轉邊數，每次複製幾個 dict。一個 RB program 編譯約 0.38 s，restore 的成本可以忽略。
- 換指標也有同類的 bug：某段程式繞過 `current_path` 讀到舊欄位。Capture／restore 把風險集中在兩個函式裡。

對先例的立場：
- 遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md)：fork 複製結尾，join 取較大者，不移動 reference；Branch 與 Repeat 的語意不變。
- 遵循 [控制結構完備性的討論](2026-11-20-early-exit-and-control-flow-completeness.md) 與 [ADR 草稿](../adr/draft/control-flow-completeness.md)：使用者選的 K1，先抽共用狀態，再做跳轉。

## 被否決的選項與理由

- **每條 timeline 維護自己的狀態：** 大部分狀態不屬於 timeline，切換路徑時仍然要處理整份。
- **用指標切換目前路徑、不做 restore：** 要改 QICK 的 macro，或攔截 QICK 的屬性。
- **完全的 functional state passing：** 每個 macro 改成 `preprocess(state) -> state`，每個程式位置保存自己的狀態值，這是 MLIR data-flow framework 的做法。這要改寫所有 timeline macro，還要包裝 QICK 的 macro，規模遠超過這次的目標。`PathState` 定成 frozen 值，以後要走這條路可以直接沿用。

## 留下的問題

- `PathState` 現在記的讀取數是 QICK 的 source trigger 計數。Branch 會把各 case 加總，Repeat body 只算一次，所以它不是每發實際執行的讀取數。每發讀取數的推算，以及在 Branch 出口、label 處比對各路徑讀取數，屬於之後的工作。
- `cond_jump`、`jump`、`label`，以及迴圈的 continue／break，會在這個狀態物件上實作，方向見 ADR 草稿。

## 依據

- 實作：[`_timeline_state.py`](../../lib/zcu_tools/program/v2/_timeline_state.py) 的 `PathState`、`TimelineBook.capture`、`TimelineBook.restore`；Branch 在 [`_timeline_macros.py`](../../lib/zcu_tools/program/v2/_timeline_macros.py)，Repeat 在 [`_timeline_repeat.py`](../../lib/zcu_tools/program/v2/_timeline_repeat.py)。
- 等價檢查（工作檔未入庫）：在 `abec61d9b` 與改動後的 tree 上各編譯 304 次，ASM 與 metadata 逐位元組相同。`tests/program` 的 3106 個測試照舊通過。
- [Møller & Schwartzbach, Static Program Analysis](https://cs.au.dk/~amoeller/spa/spa.pdf)，§4.2、§5.1（lattice 與 dataflow）。
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)（Blocks 與 block arguments）；[MLIR Data-flow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)。
