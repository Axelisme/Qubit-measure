---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-10-31-timeline-variable-model.md
  - 2026-11-15-two-qubit-rb-simultaneous-gates.md
  - 2026-11-19-single-qubit-rb-irb-merge.md
basis:
  repo:
    main: 2b76fe05f
    spike: 0d871cbd5
  external:
    - "Møller & Schwartzbach, Static Program Analysis (lecture notes), accessed 2026-11-20"
    - "Cousot, POPL 2024, arXiv:2310.15340, accessed 2026-11-20"
    - "MLIR LangRef and SCF dialect pages, accessed 2026-11-20"
---

# Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性

## 結論

- Two-qubit RB 現在每個 (depth, seed, arm) 在主機上各編一支程式。這次評估改成一個 (seed, arm) 只編一支程式，depth 在板上掃。
- 現有的 `repeat` 和 `branch` 都做不到「到了指定 depth 就結束本發」：條件區塊執行完，會回到後面的程式繼續跑。
- Spike 用 QICK 的 `CondJump`、`Jump`、`Label`，加上手寫的編譯狀態儲存與還原，做出了提早結束本發。逐事件時序和舊程式完全相同。但手抄編譯狀態的寫法很脆弱，不作為正式做法。
- 使用者裁定，這是既有控制結構不完備，而不是新需求需要新機制：抽出 Branch 與 Repeat 共用的編譯期狀態（存、還原、合併），再補上第四種結構「提早結束本發」（K1）。
- 這個方向記在 [ADR 草稿：Timeline 控制結構的完備性](../adr/draft/control-flow-completeness.md)。共用狀態物件的設計，以及 RB 是否改成在板上掃 depth，都還沒決定。

## 問題

### 現在的做法與成本

Two-qubit RB 在主機上依序掃 depth、seed、arm，每個組合編一支程式並取數一次。CZ 量測任務用了 9 個 depth（0 到 48）、12 個 seed、2 個 arm，每輪要編譯並取數 216 次。

### 單 qubit RB 為什麼能在板上掃

單 qubit RB 不是靠提早結束：

- `add_loop("depth")`，從表載入該 depth 的前綴長度；
- 用 register 次數的 `repeat` 播放前綴。每次迭代載入 gate 編號，用 `ComputedPulse` 從 7 個固定 pulse（Id、X90、X180、−X90、Y90、Y180、−Y90）挑一個；
- 每次迭代結束呼叫 `delay_by_timeline()`，編出來是一個常數 `inc_ref`，不需要在執行期做 s14 運算。

單 qubit 的 frame 只會是 90° 的倍數，所以 7 個 pulse 就夠用。Two-qubit 有兩點不同，不能照抄：

- 每個 CZ 都帶任意角度的 local-Z 修正，例如 −3.61°。X90 的相位因此是連續值，不是 7 個候選之一。
- 2026-11-15 定下的並行排程讓 F1、F2 各走自己的時間線，只在 CZ 處同步。Clifford 的長度不固定，兩條時間線在 Clifford 邊界也不對齊。

### 選項的來源

使用者 2026-11-19 提出兩種想法：「關於RB，我的想法有兩種，第一種是存所有可能的相位組合的recovery gate，第二種是執行期再載入對應的phase，後者比較複雜，前者可能很消耗pmem但簡單。」

## 考慮過的選項與比較

### 四個方向

| 選項 | 做法 | 主要成本 |
|---|---|---|
| HW0 | 維持主機迴圈。 | 每輪 216 次編譯與取數。 |
| HW1 | Pulse 的相位在執行期從 frame register 讀取；zcu_lab 產生 two-qubit 的事件表。 | lib 200–400 行、zcu_lab 300–500 行，再加測試與模擬器驗證。 |
| HW2 | 把用到的每個相位都列成候選 pulse。 | 候選數隨 depth 增加，depth 100 時每顆 qubit 約幾百個 pulse 定義。 |
| HW3 | 使用者的第一種想法的最小形式：前綴是共用的直線程式。到了每個要量的 depth 設一個檢查點，迴圈索引等於該點時播預先編好的 recovery、讀取、結束本發；否則繼續播前綴。只存實際用到的 depth。 | 需要「提早結束本發」的控制結構。 |

### HW3 spike 的結果

**現有結構擋住的地方。**
- `repeat(n=register)` 可以讓 recovery 只在某個點執行，但看不到第二條外部時間線（`rb_f2_time: unknown timeline`），執行完也會回到後面的前綴。
- `branch(by="depth")` 也一樣：case 只看得到單一母時間線，執行完同樣回到前綴。

**原型。** 為每個 (seed, arm) 開一個板上 `depth` 迴圈。
- 每個檢查點用 `TEST depth_idx - #point` 判斷，不相等就跳過 recovery。
- 命中時沿 F1、F2 各自的時間線播 recovery、讀取、收尾，再跳到共用的 shell 收尾。
- 沒命中的路徑還原 timeline 結尾、QICK 各 channel 的 timestamp 和 ADC trigger 計數，繼續編前綴。

**時序。** 用 `ir.analyze_timing` 逐路徑和舊程式比對 `(kind, port, cycle, value)`，以及 waveform 參數、body_end、shot_end。646 條路徑、399,037 個事件全部相同。這是編譯出來的排程時間，沒有驗證實板的 tProc latency。

**s14。** 使用者提醒：「HW3-2你要注意，timeline機制沒寫好會導致編譯的結果出現反覆在執行期透過步進s14來決定pulse timestamp，通常可以透過插入inc_ref，也就是delay來解決。這能減少指令消耗。」（2026-11-20）
- 傾印 11 個點（depth 0 到 100）的組語：沒有任何一行用到 s14，共 13 個 `TIME inc_ref`，分別是開頭 1 個、每個檢查點的收尾 11 個、relax 1 個。
- 前綴 pulse 都用 `@T` 立即值。

**PMEM（FTF profile 上限 4096 word）。**

| Depth | Reference | Interleaved |
|---|---:|---:|
| 0–100 全掃，101 點 | 2161 | 2393 |
| 0–200 每 10 取 1，21 點 | 1601 | 2019 |
| 0–200 全掃，201 點 | 4348（超過） | 4743（超過） |

**FTF。** depth 0、1、2、5，每點 8192 shots。新舊程式的聯合基態機率完全相同，抽樣差都在逐點 95% 區間內。終端 IQ 分類的 `gg` 計數差也在區間內。

**效益。**
- 編譯時間：主機迴圈每個 (seed, arm) 1.42 s，HW3 0.38 s。24 個 (seed, arm) 合計約 34 s 對 9 s。
- 取數次數：每輪從 216 次降到 24 次。每次 round trip 的時間沒有在實板上量過，估計只佔總時間幾個百分點。

### 怎麼理解這個機制

使用者質疑 spike 的寫法：「什么叫做编译器快照跟还原快照啊？我怎么觉得你引入了一种相当危险、复杂的机制？你确定它的可扩展性还有可维护性是高的吗？还有我自己觉得它的可读性好像有点糟，有没有什么？ 可以再简单的理解它的机制，用一些已经既有的程式概念之类的。」（2026-11-20）

用既有的概念說：

- Builder 把程式從頭走一次，邊走邊維護「目前位置」：各時間線的結尾、各 channel 的 timestamp、trigger 數。這相當於組譯器的 location counter。
- 提早結束就是 guard clause 的 early return。被跳過的區塊之後，沒跳走的那條路從未執行過區塊內容，所以編譯要從進入區塊前的位置接著走。
- Branch 早就這樣做：`TimelineBranch.preprocess` 在每個 case 編譯前，還原 timeline 結尾、QICK gen/ro timestamp、runtime 結尾、time bounds 和 frames。
- Spike 把同樣的動作手抄了一份，涵蓋三種狀態。以後每多一種編譯狀態，兩處都要改，所以脆弱。

使用者接著問：「那感觉起来用 condition jump 加上 label 应该就可以做到需要的效果，而不引入新的机制吧？」（2026-11-20）

在 tProc 指令層級，答案是對的：spike 用的就是 `CondJump`、`Jump`、`Label`。但如果 builder 不知道有跳轉，lib 有五個地方會算錯：

| 地方 | 不知道有跳轉時會怎樣 |
|---|---|
| Timeline 結尾 | recovery 寫完後游標往後走，後面的前綴從 recovery 結束的時間開始排。 |
| QICK 的 timestamp 與 ADC trigger 計數 | QICK 依語法數 trigger，buffer 會配成每發 21 次讀取，實際只有 1 次。 |
| 每條路徑的收尾 | Shell 的 WAIT 是編譯期常數，只按最後一條路徑算。 |
| 模擬器 | 拒絕 raw jump：`JumpInst ... unsupported instruction effect`。 |
| 讀取解析 | `shot_sums` 報 `expected 21 reads per shot; actual 1`。 |

第一項有一個不用還原的做法：recovery 排在 fork 出來、不 join 回去的新時間線上。ADR-0083 允許 fork 之後不 join，這樣前綴的游標就不會被動到。其餘四項仍需要處理。

### 完備性

使用者的判斷：「我觉得这算是旧有的机制出现的不完备性，而不是新的需求需要新的机制。所以我们可能要讨论一下如何完善旧机制。」（2026-11-20）

查證後，現有的直線、迴圈、分支三種結構，各自處理三件事：

1. **編譯期狀態。** Branch 在 `TimelineBranch.preprocess` 自己存和還原；Repeat 另有 `RepeatFrame` 與 `rebase_native`。沒有一個共用的狀態型別。
2. **每發的讀取數。** QICK 依語法數 trigger。用到 Branch 或 Repeat 時，作者要手動呼叫 `set_reads_per_shot`。lib 有 4 個模組要求這樣做（`read_axes`、`rpi`、`chain_*`），`zcu_lab` 有 13 個檔案在呼叫。
3. **模擬器。** 依 ADR-0083〈Compiled snapshot 與 simulation 入口〉，只認 producer 標好的 LOOP、BRANCH 結構，不從 jump 反推。2026-10-30 的裁決是「通用標記暫緩」、只擴充有限的 LOOP metadata。

Böhm–Jacopini 定理說，循序、選擇、迭代三種結構足以表達任何流程圖程式，但要靠輔助變數記住位置。允許結構化的提早結束，例如 early return 或 multi-level break，就不需要這些變數。現有 builder 有前三種，缺的是提早結束。

每加一種結構，上面三件事都要各寫一次。Spike 手抄狀態，就是把第 1 件又重寫了一份。

| 選項 | 內容 |
|---|---|
| K1（採用） | 先抽出共用的編譯期狀態物件，提供存、還原、合併三個操作。Branch 和 Repeat 改用它，行為不變。再補上「提早結束本發」，它用同一個狀態物件。模擬器多認一種有限的 metadata，這條路徑的讀取數由 lib 計算。 |
| K2 | K1 再加上讀取數全面由 lib 推導，Branch 和固定次數的 Repeat 不再需要手動 `set_reads_per_shot`。會改 13 個實驗的寫法。 |
| K3 | 只做狀態物件的重構，不加新結構，RB 維持主機迴圈。 |

使用者回答：「K1，我關注怎麼設計這個共用機制與結構，使其容易維護，可讀性高，有專業術語對應，寫法又自然。」（2026-11-20）

### 對應的專業術語

以下對應由文獻整理而來，套到 builder 上是推論：

- **抽象解譯（abstract interpretation）。** Builder 是一個依語法走一次的抽象解譯器。編譯狀態是 abstract state，每種結構的編譯規則是 transfer function，分支之後的合併是 join。見 Møller & Schwartzbach《Static Program Analysis》§4.2、§5.1、§5.10（<https://cs.au.dk/~amoeller/spa/spa.pdf>）。
- **不可到達。** 提早結束之後的那個位置，記作 bottom（⊥），SPA §8.2 稱為 `unreachable`。由 join 的定義可得 join(⊥, s) = s，所以「沒跳走的路從進入前的狀態接著走」不需要特例。
- **正常結束與中斷結束分開。** Cousot 把每個語句的語意分成正常結束與 `break` 兩部分，並給出 `[[S1;S2]]^b = [[S1]]^b ∪ ([[S1]]^e ; [[S2]]^b)`（POPL 2024，arXiv:2310.15340，§I.1.1 式 47）。對應到 builder：每個區塊編完回傳兩個狀態，一個是沒跳走的，一個是跳走的。
- **SEME region。** 單入口、多出口的區塊，MLIR LangRef〈Regions〉用這個名稱（<https://mlir.llvm.org/docs/LangRef/>）。沒有提早結束時，每個 builder 區塊都是 SESE。
- **Branch 的兩種合併。** 「取最大結尾」是編譯期的 join；「取實際執行那個 case 的 runtime 結尾」是執行期的 select，對應 MLIR `scf.if` 的結果與 LLVM 的 `phi`。Repeat 跨迭代保留的結尾，對應 `scf.for` 的 `iter_args`。

### 時間線計算的限制

使用者的預期：「我预期比较困难的点是，目前的 timeline 机制是基于编译期计算，然后在无可奈何的情况下，在执行期进行比大小。如果加入了更多的条件跳转，这些 timeline 计算会变得非常复杂。我不确定能自然支援到什么地步，可能在某些地方要规定必须做一些 delay 或 sync，某种程度上限缩了表达上限，但应该够用。」（2026-11-20）

分析如下：

- 跳轉只在一個地方增加困難：多條路徑到達同一個位置，要合併狀態。合併得出常數，就不必在執行期比較。
- 提早結束有兩條邊。沒跳走的那條只有一個來源，狀態就是進入前的狀態，不必合併。跳走的那些都到 shot 收尾，那裡才是匯合點。
- 規定每個 exit 跳走前先 sync：把該路徑的時間線都 join，再用 `inc_ref` 推到自己的結尾。這樣每條路徑到收尾時的狀態都相同，不必在執行期比大小。Spike 的組語裡，exit 跳走前那行 `TIME #249 inc_ref` 就是這個 sync。
- 這和 calling convention 的 function epilogue 同一個概念：函式 return 前恢復 stack pointer 和 callee-saved registers，呼叫端就不必知道函式內部走了哪條路。

提議的第一版限制，使用者尚未裁決：

1. Exit 區塊的長度要在編譯期可知，不含 runtime 長度的 Repeat 或 runtime 結尾。這樣 sync 是一個常數 `inc_ref`。
2. Exit 只能放在 shot body 的最外層，不能放在 Repeat 或 Branch case 裡。放在裡面時，Repeat 跨迭代的狀態與 Branch 的合併都要多處理一種「這條路已經離開」的狀態。
3. Sync 由這個結構自動插入，作者不寫 delay。

RB 的用法三條都符合：recovery 是固定的 gate，長度編譯期可知；檢查點都在 shot 最外層。

## 被否決的選項與理由

- **Spike 手寫的狀態儲存與還原。** 手抄三種編譯狀態，每多一種狀態兩處都要改；使用者認為危險、難讀。改為 K1 的共用狀態物件。
- **在 builder 裡直接插 `CondJump` 與 `Label`，不讓 builder 知道。** timeline 結尾、QICK 計數、收尾、模擬器、讀取解析五處都會算錯。
- **HW2。** 記憶體隨 depth 增長，抵消了在板上掃的好處。
- **K2、K3。** 使用者選 K1。

## 留下的問題

- 共用狀態物件的介面：Branch 與 Repeat 現在各自保存哪些狀態、兩份清單的差異是否藏有 bug，調查進行中。
- 第一版限制的三條規則，待使用者裁決。
- RB 是否在 K1 完成後改成在板上掃 depth（HW3），還是維持主機迴圈（HW0）。HW1 沒有被否決，也沒有再評估。
- Depth 0–200 全掃超過 4096 PMEM。可能的方向是分塊掃描，或把共用的讀取搬到 recovery 後的共用尾段，都沒有量測。
- Guide 要求 depth 互不相同，但 `RBCfg` 只檢查排序，重複 depth 怎麼處理還沒決定。
- `rb_heralded` 是否跟進還沒決定。
- 實板的 tProc latency 與每次 round trip 的時間都還沒量。

## 依據

- Repo：討論時的 main `2b76fe05f`；spike 在 `0d871cbd5` 上執行。
- Spike 的報告、probe 腳本、組語傾印與編譯時間量測放在本機工作檔，未入庫。本文已寫入結論與必要的數字。
- 外部來源（2026-11-20 存取）：
  - Møller & Schwartzbach，《Static Program Analysis》：<https://cs.au.dk/~amoeller/spa/spa.pdf>
  - Cousot，"Calculational Design of [In]Correctness Transformational Program Logics by Abstract Interpretation"，POPL 2024：<https://arxiv.org/pdf/2310.15340>
  - MLIR LangRef：<https://mlir.llvm.org/docs/LangRef/>；SCF dialect：<https://mlir.llvm.org/docs/Dialects/SCFDialect/>
  - LLVM LangRef：<https://llvm.org/docs/LangRef.html>
  - Kozen & Tseng，"The Böhm–Jacopini Theorem is False, Propositionally"：<https://cs.cornell.edu/kozen/Papers/BohmJacopini.pdf>
