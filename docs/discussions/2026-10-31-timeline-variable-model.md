---
date: 2026-10-31
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/v2/timeline.py
  - lib/zcu_tools/program/v2/_timeline_runtime.py
  - lib/zcu_tools/program/v2/_timeline_expression.py
  - lib/zcu_tools/program/v2/ir/passes/loop/unroll.py
  - lib/zcu_tools/program/v2/ir/analysis.py
  - lib/zcu_tools/program/v2/ir/hw_semantics.py
  - lib/zcu_tools/program/v2/ir/instructions.py
  - docs/discussions/2026-10-29-module-to-macro.md
basis:
  - "repo@43813fba3（main）；timeline 程式碼與量測取自尚未併入 main 的分支：wave/timeline-contract/integration 999e42c67 與 a63e1dbf1、lane 06 f318fd448"
  - "openquantumhardware/qick HEAD 80753a9c55142f00b00bc236d91074b5901d9743（firmware/ip/qick_processor/src，2026-10-31 讀取）"
  - "本機 qick Python 套件 0.2.436（tprocv2_assembler.py、asm_v2.py、drivers/tproc.py）"
  - "逐輪裁決記在未入庫的工作檔；本文寫入結論與必要事實"
---

# Timeline 編譯：變數模型、通用 join 與 s14 事件時間

## 結論

1. **宣告不做一致性檢查。** `TimelineProgramV2` 的 `declare_gen`／`declare_readout` 回到 QICK 原生行為：後宣告覆蓋先宣告，作者負責。原本為了檢查而長出的 owner 追蹤、typed normalization、宣告查詢 API 與 DMEM mixin 拆分都不做。
2. **變數只有一種形態。** 變數是一組「軸 → 分量」，值等於各軸分量相加。軸有已知可能值時，分量是沿該軸的一維 numpy 陣列；軸只有範圍時，分量是係數。常數是固定 ID、大小 1 的 `const` 軸。只有一個軸的變數和其他變數沒有形態差別。
3. **計算靠軸比對。** 加、減、常數倍逐軸對齊；範圍等於各軸分量上下界相加，因為各軸獨立，結果精確。除 `const` 外只有同一個已知軸時，max／min、ceil、定點換算沿軸逐點算，結果仍是變數。跨軸的 ceil／定點換算寫成原變數加一個只有範圍的誤差軸。
4. **運算式只有兩種節點：變數與極值節點。** 跨軸的 max／min 是極值節點，分支是變數或相反方向的極值節點，最多保留 m=8 支，其餘併成殘差包絡 R。證明時下界用保留支、上界用保留支加 R。中間候選上限 K=81，max／min 交替深度上限 D=2，整個運算式葉節點上限 81，超過時收合最深的子節點。這些常數可依實測編譯時間調整。軸與 register 綁定都不是節點。
5. **所有合流走同一個 join 演算法。** fork／join、branch 結尾、Repeat、pulse／readout 結尾彙整都經同一個 max。演算法先攤平，再提出公共部分，剔除被支配的分支，剩 k 支才做 k−1 次 runtime max。runtime max 用 4 條指令：`d=x−y; m=d ASR 15; r=d AND m; max=x−r`。
6. **ref 只在作者明確要求時推進。** 常數時間用字面值 `@t`，runtime 時間直接寫進 s14。編譯器記住 s14 目前的內容，同一基準的下一個事件只寫 `s14 += 差值`。
7. **Repeat 拿掉 `advance_ref`。** body 頂層最後一個操作是對循環 timeline 的 `delay_by_timeline()`，且進入時循環 timeline 的結尾是 0，才編成常數 `inc_ref`。其他情況用 s14 時間指針。branch case 內任何推進 ref 的操作都 build 報錯。
8. **IR 只看硬體語意。** 字面值 `s14 += #c` 與 `TIME inc_ref #c` 同樣計入時間窗口；增量來自 register 時窗口未知，不展開。
9. **改名。** `delay_timeline` 改為 `timeline_inc`，`sync` 改為 `delay_by_timeline`。名字含 delay 的方法都會推進 ref。

這些結論在 timeline-contract 收尾時寫入 ADR；本文只記推導與取捨。

## 問題

討論從三個問題開始。

- 使用者懷疑先前的 timeline 意外行為盤點過度工程。宣告查詢一路長出 owner 追蹤、typed normalization 與 parity 測試，最後還要拆 `resources.py` 的 DMEM mixin。
- branch 合流把每個 case 的結尾都在 runtime 取 max。使用者認為成本太高：「如果可以先做一些编译期的运算式整合，应该可以降低执行期的成本。」
- `advance_ref=False` 的 Repeat 在次數來自 register 時編譯成 233 條指令與 32 字 DMEM，同一個循環的 `advance_ref=True` 只要 45 條。

這三件事都指向同一個底層問題：編譯器怎麼表示「執行時才知道的時間」，又怎麼在編譯期證明它們之間的關係。

## 考慮過的選項與比較

### 合流成本

當時的 lowering 先把每個 case 的結尾各自重建進 register，再兩兩取 max，每次 9 條 ALU 指令（SUB、ABS、2×ASR、2×SUB、3×ADD）。表達式層只在兩種情況化簡：變數項完全相同時留常數較大者，或某個 case 以獨立區間支配全部。

advisor 先提 P1（以差值上下界剔除被支配的結尾）與 P2（4 條指令的 max）。使用者指出：「第七点其实不只是 branch 的河流，它事实上是应该是 join 的河流啊。branch 只是结尾自动 join 而已。」接著要求「设计一个多个线性式都可以适用的 join 的算法」。結論 5 就是這個演算法。

例子：三條 timeline 的結尾分別是 e+L(i)+40、e+L(i)+M(j)、e+2·L(i)，其中 L 是 60..256、M 是 0..80。公共部分是 e+L(i)，餘項是 {40, M(j), L(i)}。L−40 恆 ≥20，所以剔除 40。結果是 e+L(i)+max(M(j), L(i))，只剩 1 次 max、4 條指令；原做法是 2 次 max、18 條指令。

### 變數範圍格式的演進

1. 使用者提出：每個變數存有方向的範圍，再加一個獨立 ID，ID 通常取自 loop 名，相同 ID 就不獨立。
2. advisor 補上偏差區間，讓非仿射變數也能用。實作後兩個既有測試失敗。選表值在 129 與 43 之間交替，下界被算成 −40，被非負檢查拒絕。兩張反向表在同一 loop 上取 max，結果逐點恆為 100，卻被當成新 ID，WAIT slack 從 31 變 131。
3. 修正為沿 loop 軸逐點存偏差，同 loop 的運算結果保留 ID。之後又出現跨 loop 案例：`body=max(left+40, right)`，把它當成新 ID 會丟掉 body≥left+40 的關係，ADC 尾端上界從 30 變 50。修正為 max／min 保留成符號式，代數分配後化成 `max(0, min(30, left+70−right))`，上界回到 30。
4. 使用者提出可能值與範圍分開存，接著提出 n 軸、n 維可能值，用來表示兩個獨立變數相加。
5. 使用者再修正：「每軸存各自的一維可能值就好了嘛？而且這不就是線性式嗎？」以及「變數是變數，線性式是線性式，後者應該可以視為變數到變數的計算函數，timeline join的對象都是變數」「常數應該是一個只有一個值與固定id的軸」。最後一次修正：「源變數應該與延伸變數沒有型態上的差別，計算也不用回歸源變數，計算的原理是軸比對，源變數只是軸數為1的變數而已」。結論 2、3 由此而來。

### 極值節點的爆量處理

證明時才會分配 max 節點，runtime 每次 join 只算一次並物化。所以爆量只影響能證明多少，不影響結果正確性。

- **夾逼對 [L, U]**：L 取下界最高的分支，U 取逐軸包絡。使用者認為退化太猛。
- **保留 m 支加殘差包絡 R**：保留上界最高的 m 支，被捨棄的分支併成 R。R 被某支保留支支配時丟掉 R，節點仍精確。夾逼對是 m=1 的特例。
- **K 的取值**：兩個節點相加最多 (m+1)² 個候選。剔除分兩步，先用各支上下界做 O(n) 篩選，再對剩下的分支兩兩相減。兩支相減的成本是各軸長度之和，不是乘積。m=8 時最壞約 6,500 次比較，每次運算是毫秒級；m=16 時約 83,000 次，每次運算約 0.1 秒。所以取 m=8、K=81。
- **巢狀**：同方向巢狀會攤平，但 max／min 交替的深度會隨程式長度增加，葉節點最多 (m+1)^深度。保留支的條件放寬為「每支 ≤ 真實值」，超過深度上限時，把子節點收合成下包絡（當保留支）與單一分支（併入 R）。

### s14 與 ref

調查 qick HEAD 80753a9c5 的 tProc v2 RTL 得到四項事實：

- 程式讀不到 ref。s0..s15 沒有接 ref（`qcore_reg_bank.sv:190-204`），s11 是截成 32 位的 abs−ref。
- `inc_ref` 把 32 位運算元零擴展後加到 48 位的 ref（`qproc_ctrl.sv:294-301`），所以只能前進。`set_ref` 寫入 `{16'd0, dt}`，只能設 2³² cycles 以內的值。
- s14（`s_out_time`）只由 REG_WR 寫入，TIME 不碰它（`qcore_reg_bank.sv:136`）。
- 沒有 `@t` 的 port write，時間是 signext32(s14) 加推入當下的 ref（`qproc_dispatcher.sv:168`）。

所以「循環內推進 ref、出口再還原」不可行。`advance_ref=False` 原本把尾端寫進 s14 的做法，正是 QICK 原生 sweep 時間（`set_timereg`）的用法。

使用者確認通用 s14 方案之前問：「對於簡單實驗，常數timestamp會比s14的指令精簡，你認為這是必要的損失嗎？」結論是不必要。常數時間與 QICK sweep 時間都不受影響，只有原本就是 runtime 的時間改走 s14：每個事件從 2 條指令加 1 個 data register，降到 1 條、不占 register。

使用者接著問明確插入 delay 是否能簡化指令。在 integration a63e1dbf1 上實測：單一 timeline，sweep 長度 delay 後接 3 個 pulse，`delay_timeline` 是 35 條指令、7 個 register，`delay` 是 24 條、4 個 register。推進 ref 後事件回到字面值 `@0`、`@28`、`@56`。代價是 ref 共用，其他尚未 join 的 timeline 的常數時間會變成 runtime。使用者的原則是「我不想自動推進ref_time」，所以由作者決定何時插入。

### Repeat 與 IR 展開

`UnrollLoopPass` 依「時間窗口 − body 指令成本」選展開份數 k，窗口只計字面值 `TIME inc_ref`，餘量 ≤0 時取上限 32。`advance_ref=False` 的 body 沒有 `inc_ref`，所以被判成超載、展開 32 份，加上 32 字 dispatch table（`TEST r10 - #32`）。advisor 在 lane 06 快照上重跑，重現了 233 條指令。

| 方案 | 內容 | 結果 |
|---|---|---|
| timeline 標注給 IR | timeline 把每圈長度標在 IR loop 上 | 使用者：「讓IR知道timeline機制有點冒險」 |
| `CarriedUpdate` | IR loop 顯式宣告每圈 register 增量，展開時合併 | 合併每 2 圈只省 1 條，卻要 dispatch 入口補償；使用者：「似乎單純的重複更新s14就好」 |
| s14 當時間指針 | 每圈 `s14 += 圈長`，IR 把字面值 s14 增量算進窗口 | 採用。每圈 body 2 條，與 True 相同 |
| 循環內推進 ref 再還原 | body 用 `inc_ref`，出口還原 | 硬體不可行 |

拿掉 `advance_ref` 時需要判斷定點。child 提兩案：A 是另寫不配置資源的分析路徑，B 是假設定點後重跑 preprocess。兩者都可能碰到自訂 Macro 的副作用。advisor 改提結構規則：`delay_by_timeline()` 會把結尾歸零，所以只要它是 body 的最後一個操作、進入時結尾也是 0，就一定是定點，不需要分析。代價是 `sync` 放在 body 中間的巧合定點只能編成變數尾端。

### 過度工程的盤點

針對當時尚無實驗使用的四個機制，盤點了只為它們存在的產品碼與 runtime 成本（行數為估計）：

| 機制 | 只為它存在的產品碼 | 用到時的 runtime 成本 | 沒用到時 |
|---|---|---|---|
| Repeat 變數尾端 | 約 +250 行 | 循環 body 2→5 到 15 條，每個循環約 3 個 register | 0 |
| readout config runtime 切換 | 0（QICK 原生機制） | 與發兩次相同 | 0 |
| k=8 定點換算 | 約 +292 行 | 每次換算 7 到 12 條 | 0 |
| 無 DMEM 合流重建 | 約 190 行（骨架原本就有） | 每次 max 9 條 | 0 |

使用者判斷：「目前虽然没使用，但我认为属于合理使用范围啊，所以这些机制还是可以考虑的。」四個機制都保留，只有宣告檢查依「既然做这些检查这么的麻烦，那就干脆不检查了，就跟原生机制一样就好」拿掉。

## 被否決的選項與理由

- **宣告一致性檢查、owner 追蹤、typed normalization**：補丁一層疊一層，改回 QICK 原生行為。
- **網格逐點包絡剪枝、依 loop index 比較斷點來選擇**：前者編譯成本隨網格增長，後者要條件跳轉，一次跳轉約 40 cycle。
- **純量偏差區間**：對非單調的選表過度保守，兩個既有保證因此失效。
- **把跨 loop 的 max 當成新 ID**：會丟掉 max 與其分支之間的關係。
- **n 維可能值網格**：每軸一維加軸比對已能表示同樣的關係，不必存乘積大小的陣列。
- **夾逼對**：退化太快，改用保留 m 支。
- **讓 IR 知道 timeline、`CarriedUpdate`、展開時合併 s14 更新**：耦合 IR 與 timeline，或省得太少。
- **循環內推進 ref 再還原**：tProc v2 無法讀出或回撥 ref。
- **編譯器自動推進 ref**：違反使用者原則，而且共用的 ref 會讓其他 timeline 的常數時間變成 runtime。
- **定點分析或重播 preprocess**：用結構規則取代。
- **名稱 `extend`／`move_ref`**：使用者改用 `timeline_inc`／`delay_by_timeline`，讓名字含 delay 的方法都動 ref。

## 留下的問題

- m=8、K=81、D=2、葉節點上限 81 都是暫定值，要等實作後依實測編譯時間與收合頻率調整。
- QICK pulse 是否像 `trigger` 一樣支援 `t=None` 使用 s14，實作 s14 事件時間前要確認。
- RTL 顯示遲到的事件在絕對時間超過事件時間時觸發，與官方 TRM「遲到事件不會出現」的說法衝突，沒有模擬驗證。
- IR 的 `TimeInst` 宣告 TIME 會讀寫 s14，與 RTL 不符。目前因 `dead_write` 把 s0–s14 視為 volatile、`TimedMergePass` 在 s14 讀寫前 flush，效果只是保守；s14 方案實作時一併修正。
- s14 游標的三種失效情況（QICK 原生 macro 寫 s14、raw `asm_inst` 寫 s14、branch 合流）是否完整，要靠實作與測試確認。
- two-qubit-heralding 的 chevron 實驗在 16 個 register 內編譯失敗，收尾時 9 個 register 是 timeline 結尾。這次的變數模型、join 與 s14 方案在結構上能減輕 register 壓力，但所在分支疊在 two-qubit-heralding 之上，無法先給它用。

## 依據

- tProc v2 RTL：openquantumhardware/qick HEAD 80753a9c5 的 `qcore_reg_bank.sv`、`qproc_ctrl.sv`、`qproc_dispatcher.sv`（行號見上文）。
- 本機 qick 0.2.436：`tprocv2_assembler.py` 的 sreg 別名（s11 `s_usr_time`、s14 `s_out_time`），`asm_v2.py` 的 `set_timereg`／`inc_timereg` 與 `trigger(t=None)` 說明。
- 本 repo：`ir/passes/loop/unroll.py`（k 的選法）、`ir/analysis.py::estimate_body_scheduled_ticks`、`ir/passes/timeline/timed_merge.py`、`ir/passes/dataflow/inc_reg_merge.py`、`ir/instructions.py::TimeInst`。
- 指令數與 register 數由 advisor 在 lane 06 f318fd448 與 integration a63e1dbf1 編譯固定小程式量得。機制成本盤點的行數是函式邊界估計。原始腳本與逐輪裁決記在未入庫的工作檔，本文已寫入所需數字。
