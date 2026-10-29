---
date: 2026-10-29
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/v2/modules/README.md
  - lib/zcu_tools/program/v2/macro/__init__.py
  - lib/zcu_tools/program/v2/simulation/README.md
  - lib/zcu_tools/program/v2/ir/README.md
  - docs/adr/draft/simulation-environments.md
basis:
  - "repo@c30bf2bbd；qick 0.2.436"
  - "可行性調查與兩個丟棄式原型為未入庫的工作檔（2026-10-29），結論與必要事實寫在正文"
---

# Module 改 QICK Macro 與模擬 frontend 改讀 macro

## 結論

1. 延後解析的時間層可行。原型的 `TimelineMacro` 在 QICK `preprocess` 時才解析後繼 macro 的起點；4 組設定（純量、掃 pulse 長度、掃 delay、Join 其中一支掃長度）產生的 `prog_list`、labels、pmem 與現行 Module 版逐位相同，Join 的交錯發射順序也相同。依讀出結果分支的案例（`read_input` 加 `cond_jump` 跳過一個 pulse）compile 通過。原型沒有用到 QICK 私有屬性，也沒有 monkeypatch。
2. 不立即全面遷移 Module。`Module.init()`（宣告 generator、readout、pulse、register、dmem）發生在 macro 的 preprocess 之前，搬不進 macro；完全移除 Module 不可行。
3. 模擬 frontend 改讀 macro 的方向先以第二個原型比較兩種方案：讀 ModularProgram 產生、與舊 Module 幾乎同形的「語意 macro」，或直接解讀 QICK 底層 macro。2026-10-14「保持 Module 語義的解析」的裁決暫不改寫，等原型結果再決定。正式遷移排在 heralding 第一階段之後。

## 問題

使用者在討論 heralding 時提出：「我最近在考慮將Module類降級成qick的Macro，改動範圍挺大，但可能更適合複雜機制」，並列出 Module 的三項功能：「一是在run中建立macro，二是即時決定結束時間，三是提供模擬器與ir需要的頂層資訊，一應該可以參考qick中macro嵌套，逐迭代往底層展開的機制實現，二qick可能不支援，但我們建立一個父類，比如TimelineMacro，提供時間接口也許可行，三也可以在展開時標記頂層訊息供模擬與ir利用，這方向很模糊所以我想先請你調查可行性」。

## 考慮過的選項與比較

可行性調查對三項功能的判定：

| 功能 | 判定 | 機制與限制 |
| --- | --- | --- |
| 在 run 中建立 macro | 可行，已是現況 | `Module.run` 本來就往 `prog.macro_list` 追加 macro；repo 的迴圈 macro 已有巢狀展開。QICK 只對頂層 macro 呼叫 `preprocess`，子 macro 要由父 macro 轉發；巢狀的 OpenLoop 與 tag 不會被登記 |
| 即時決定結束時間 | 部分可行 | QICK 沒有讓 macro 對外交出長度的 API。現行做法在建構階段用 Python 算 `t`；macro 版要新增延後解析的時間引用，Join 要在 preprocess 重做交錯順序。資料相依的結束時間兩種寫法都算不出，只能先同步再歸零 |
| 提供模擬器與 IR 的頂層資訊 | IR 不需改；模擬器部分可行 | IR 已只讀 `MetaMacro` 標記，不讀 Module 樹；把每個 module 的邊界標記塞進 IR 會切碎 basic block。模擬器照 Module 類別解讀，讀取二十多個 Module 屬性 |

調查也指出，`Module.run` 現在就能呼叫 `read_input`、`cond_jump`、`asm_inst`，資料相依控制流不需要 Module 變成 macro 才能寫；真正的收益在時間語意。同一份時間計算目前在 `Module.run`、模擬器的 `vector.py` 與 `timeline.py` 各重做一次，macro 化之後可以改用 QICK preprocess 算定的值。成本面是 modules 約 3100 行、模擬器核心約 1500 行、相關測試約 6000 行，以及約 80 個實驗 `core.py`。

advisor 給的選項是 A 現在做原型、B 晚點評估、C 不做，建議 B。使用者：「改macro可以做原型」。

原型結果出來後，原型建議只推進時間層，並保留同一棵 Module 語義樹給模擬器，另由框架管理的固定編譯路徑產生 timeline macro。模擬 frontend 明確拒絕 macro 版程式，所以沒有 macro 版的 template 可比。

使用者接著提出：「我認為可以考慮讓模擬器用macro以及其他資訊來模擬，轉timeline的資訊應該是夠的，只是要重寫」。advisor 指出這會改變 2026-10-14 的裁決（Frontend 解釋 Module 語意，持有局部 register、sweep、loop、table 與時間排程 context，不建立通用 tProc ISA emulator 或全域 register table），提出 M1（是否改裁決）、M2（先原型或直接開 task）、M3（原型現在做或等 heralding）。使用者：「M1可以嘗試兩種方案，讀modulerProgram格式與舊mdoule幾乎一樣的macro們來模擬，或是直接用底層支援的macro模擬，後者的好處是不侷限於modularProgram，M23A」。

兩種方案都保留 10-14 的保證：只支援封閉的型別集合，未知操作在 backend 前報錯，只有局部 register context，不支援資料相依 feedback。

## 被否決的選項與理由

- 立即全面把 Module 換成 macro：`Module.init()` 搬不進 macro；收益集中在去掉重複的時間計算，不值得一次付出整個 module、模擬器與測試的改寫成本。
- 在展開時把 module 邊界標記塞進 IR 串流：IR 已有需要的結構資訊，額外標記會限制優化並需要改 parser。
- 不做原型、停在調查：使用者選擇用原型驗證。

## 留下的問題

- 第二個原型的結論：語意 macro 與底層 macro 兩種 frontend，各能否對 Repeat、LoadValue 表派發的 Branch、ScanWith、PulseReadout、帶中途 projective 讀出的 MuxReadout 產生與現行相同的 TimelineTemplate；底層方案能否還原 ComputedPulse 的最大候選排程、TableLengthPulse 實際選中的長度、TablePulseReadout 的 init 與回繞、projective 與 terminal 讀出的分類。
- 原型只驗證固定排列、兩支 pulse-only 的 Join 與單維正向掃描；tag 登記、巢狀 loop、其他 Module 與硬體回饋延遲未驗證。保守 rounding 與 Join 重疊的 fallback 仍需保留。
- 是否修改 2026-10-14 的模擬器裁決，待第二個原型回報後由使用者決定。

## 依據

- QICK 0.2.436 `asm_v2.py`：`Macro`、`TimedMacro`、`_make_asm` 的 preprocess 與 translate 順序、`AsmV2.extend_macros`。
- repo：`lib/zcu_tools/program/v2/modules/`（Module 契約與 19 個具體類別）、`lib/zcu_tools/program/v2/macro/`、`lib/zcu_tools/program/v2/ir/`、`lib/zcu_tools/program/v2/simulation/`（`vector.py`、`timeline.py`、`frontend.py`）、`docs/adr/draft/simulation-environments.md`。
- 可行性調查與兩個原型為未入庫的工作檔；本文已寫入結論與必要事實。

## 後記（2026-10-29，第二個原型之後）

更正：上文把方案 1 寫成「讀 ModularProgram 產生、與舊 Module 幾乎同形的語意 macro」，與使用者的意思不同。使用者更正：「不是產生macro，而是使用，我預期實驗作者不怎麼要改寫法，只是module變成繼承macro，program內部解析方式不同而已」。方案 1 是 Module 類別本身繼承 QICK Macro，作者照舊寫 `Pulse(name, cfg)` 等；program 直接把這些物件放進 macro 序列，時間由 QICK preprocess 解析，模擬 frontend 讀這些 module-macro 物件。第二個原型照這個意思做。

第二個原型的結果（工作檔未入庫，結論如下）：

- 9 組設定 × 2 種方案，18 個 TimelineTemplate 與現行 Module frontend 的結果逐欄相等。9 組是純量、掃 pulse 長度、掃 delay、Join 一支掃長度、Repeat、LoadValue 表派發的 Branch、ScanWith、PulseReadout 最終讀出、MuxReadout 中途加最終讀出。兩種 frontend 都禁止讀 `prog.modules`，各自組出 template。
- 方案 2（底層 macro）也能處理沒有 ModularProgram 祖先的程式，但會遺失資訊：兩個不同名稱的 Pulse 共用同一個 pulse cfg 時，registry 只留第一個名稱，第二個 pulse 的 source 還原錯誤（物理欄位相等）。ComputedPulse、TableLengthPulse、TablePulseReadout 需要額外的局部資料流與 template 更新解讀，原型沒有做完，不算成功。
- 作者端程式：標準 zcu_lab 寫法在方案 1 預估 0 行改動；library 端仍要保留 `Module.init()` 的宣告時機、更新 factory，並處理 `run()` 回傳時間的契約、同一 instance 重用與 tag 相容。
- 規模估計：方案 1 的模擬 frontend 約 800–1,200 行，Module／Macro 整合與時間層另約 400–700 行；方案 2 約 1,500–2,500 行，而且需要另訂發射來源的契約才能補回遺失的資訊。
- 原型建議方案 1。是否修改 2026-10-14 的模擬器裁決，仍待使用者決定。

## 後記二（2026-10-29 中午，方向修正）

使用者先選方案 1（「4可以方案一」），接著說明真正的動機：「Module轉macro的目的是我有感目前在原生qick建program的語法上自己建一層module層語法，使用者要多學一次，如果只是類似原生qick的macro擴充，理解與上手難度就很低」，以及「可能要考慮可以與原生macro附用的設計，不然效益不大，前者才是真正的macro化」。

所以目標不再只是「Module 繼承 Macro、作者寫法不變」，而是降低作者要學的東西。advisor 列出兩種作者 API：直接寫原生 QICK program、把我們的 module 當成可在 `_body` 呼叫的 macro；或保留 builder 的 `.add()` 當薄層。使用者：「關於macro, 我預期讓底層qick寫法支援reset之類的操作，上層可以再包一層builder層來簡化寫法，但用那種是作者決定」。

修正後的方向：

- 底層是原生 QICK program 寫法。Reset、Pulse、Readout、Join 等是 QICK Macro，可以和原生指令、原生 macro 混用。
- 上層保留 builder，它產生的就是底層同一組 macro。用哪一層由作者決定；只用 builder 的作者，寫法不變。
- SweepCfg 轉 QICK sweep、retry、early stop 與 round 進度放在底層 program 基底與 acquire 參數，builder 只轉呼叫。
- 模擬器支援的範圍以 program 實際產生的 macro 型別定義，是原生 macro 類別的一個明確子集加上我們的 macro，不以 `pulse()`、`delay()` 等函數名稱定義（使用者：「模擬器要支援的應該是macro的子集才對，不是函數呼叫」）。子集以外的 macro 型別在 backend 前報錯；2026-10-14 裁決的封閉支援集與不做資料相依 feedback 照舊。

這使上一段「作者端 0 行改動」只對 builder 使用者成立。第二個原型方案 2 遇到的發射來源問題（兩個 Pulse 共用 cfg 時 source 還原錯誤），在作者直接寫原生 program 時同樣會出現，需要「每次發射記錄來源」的契約。

下一步是第三個丟棄式原型：T1 用底層與 builder 各寫一次並與現行 `prog_list` 逐位比對、原生 macro 與 Reset／Readout 混用的時間銜接、一個 Join 或 Repeat 例、以 macro 型別子集運作的模擬 frontend，以及實作規模估計。

## 後記三（2026-10-29 下午，第三個原型與時間模型）

### 第三個原型：兩層寫法

依後記二的方向做第三個丟棄式原型（工作檔未入庫，依據 repo@a9b66c782、QICK 0.2.436），結論如下：

- 底層寫法（program 子類在 `_initialize` 宣告、`_body` 發射我們的 macro）與 builder 寫法，對 T1 三組 cfg、一個 Join（兩支交錯發射）、一個 Repeat、以及兩個共用 cfg 的 Readout instance，`prog_list`、labels、pmem、wmem、dmem 與 asm 文字都和現行 `ModularProgramV2` 逐位相同。換 cfg 或 delay 值的負向對照會被檢出。
- 依 macro 型別派發的模擬 frontend 輸出的 `AcquireRequest` 與現行相等；支援子集以外的 macro 型別（例如 body 內的 `ReadInput`、`Resync`）在 backend 前報錯。
- 混用原生 macro 時，原型採「我們的 macro 以前一個的結束當起點，原生 `delay`／`delay_auto` 把這條串接歸零」，手算與實測一致，但這是作者要多記的一條規則。
- 還沒解決：同一 instance 多次發射與 `herald_spec` 的發射掃描、內層 macro 的 tag 與 `get_time_param`、沒有 builder 時的 retry／early stop／進度、原型複製了 QICK `make_program`、Branch／LoadValue／ScanWith／ComputedPulse／TableLengthPulse／MuxReadout 等尚未轉換、模擬器仍呼叫 `total_length` 重算長度。
- 規模估計：底層 program 120–200 行、時間層 200–300 行、19 個 module 類別淨增 300–600 行、builder 30–80 行、模擬 frontend 500–900 行，測試改寫 300–600 行與新增 500–900 行；88 個 `prog_builder(` 使用者預估 0 行改動。

### 為什麼 module 要自己推導時間

使用者問「我忘記為什麼module要推導時間而不用預設的delay auto了」。repo 的 ADR、README 與 commit 訊息沒有記下當初理由；從程式推得：QICK `delay_auto(t, gens=True, ros=True)` 的 `gens`／`ros` 只是布林值（`qick/asm_v2.py` 的 `Delay.preprocess` 經 `get_max_timestamp` 取全部 generator 或 readout 的最大結束時間），是全域屏障，無法表達不同通道重疊。module 自己算時間才能有非 blocking pulse（下一個 module 從同一時點開始）與 Join 的各分支獨立計時（commit 074329ab9「replace NonBlocking with Join」）；掃描長度的捨入也由 module 依 modules README「Swept pulse blocking timing」控制。

### 時間模型的比較

| 模型 | 做法 | 混用原生 | 重疊與 Join |
| --- | --- | --- | --- |
| M1 自有串接（第三個原型） | 我們的 macro 以前一個的結束當起點 | 要多記一條串接規則 | 支援，可與現行逐位相同 |
| M2 參考時間 | blocking macro 以 `delay(自己的長度)` 推進 QICK 參考時間 | 只有 QICK 一套模型 | hard delay 不擋其他通道；多 `inc_ref` 指令，需改用 timeline 等價驗收 |
| M3 `delay_auto` 屏障 | 每個 blocking macro 後接 `delay_auto` | 自然一致 | 無法跨通道重疊，否決 |

使用者：「Program作者應該預設理解qick知識，而我在思考的是，能不能擴充qick的delay_auto，使得其能夠只delay特定的timeline，也就是M1精神使用M2實現」，並更正「我說的timeline不是綁定一個channel, 而是一個抽象的time counter，可以讓不同ch的pulse對齊」。

### 具名 timeline 模型

整理後的模型：timeline 是具名的參考時間，以相對 QICK 參考時間的 offset 記錄，不綁通道；掛在同一條 timeline 的 pulse 不論通道都對齊。

```python
self.pulse(flux_ch, "flux", t=0)                  # 原生、預設 timeline：一段長 flux
self.pulse(drive_ch, "pi", t=0, timeline="q")     # 掛在 q
self.delay_auto(timeline="q")                     # 只把 q 推到 pi 結束；不發指令
self.pulse(ro_ch, "ro", t=0, timeline="q")        # pi 結束就讀出，不等 flux
self.delay_auto()                                 # 原生全域同步：參考時間前進，q 對齊上來
```

使用者確認四點（「1同意 2同意 3應該可以 4背後可能是不同的macro，但方法接口可以考慮設計成一個沒問題」）：

1. 預設 timeline 就是 QICK 參考時間，不帶 timeline 的原生寫法行為不變。
2. 具名 timeline 前進只在 preprocess 記帳、算出各 pulse 的絕對 t，不發 `inc_ref` 指令。
3. 原生 `delay`／`delay_auto()` 照 QICK 推進全域參考時間，具名 timeline 跟著扣減，最早對齊參考時間。
4. 方法介面是原生方法加 `timeline=` 參數，背後可以是不同的 macro 類別。

我們的 module 因此不需要另一套串接規則：blocking Pulse 在目前 timeline 發射後 `delay_auto(timeline=目前)`，非 blocking 不推進，Join 從目前 timeline 分出子 timeline、結束時推到各子 timeline 的最大值。QICK 已有逐通道的公開 timestamp 方法（`get_timestamp`、`set_timestamp`、`decrement_timestamps`）。下一步是第四個丟棄式原型，只用 QICK 公開 API 驗證這個模型、與現行 `prog_list` 的逐位比對、指令數與模擬器的讀法。

## 後記四（2026-10-29 傍晚，第四個原型與定案方向）

### 第四個原型：具名 timeline

依後記三的模型做第四個丟棄式原型（工作檔未入庫，依據 repo@b1cca2691、QICK 0.2.436），只用 QICK 公開 API，沒有私有屬性或 monkeypatch：

- timeline 狀態放在我們的 program 基底，只在 preprocess 變動，每次 compile 重置。覆寫公開的 `decrement_timestamps`，原生 `delay`／`delay_auto()` 推進參考時間時，具名 timeline 跟著扣減並以參考時間為下限；四個混用例子的手算與實測一致。
- 具名 timeline 前進只記帳，不發指令。頂層的 tag 與 `get_time_param` 可用。重疊例子中 readout 在 drive 結束時開始，不等 flux。
- 模擬 frontend 直接讀 macro 已解析的時間，不再自己重算 delay。
- timeline 核心約 100–120 行，比後記三估計的時間層 200–300 行小。

### 掃描與指令數

tProc v2 的時間可以是指令內的字面值，或先放進暫存器。掃描的時間只能用暫存器，每個時間暫存器要初始化、每點加步長、每圈歸位。現行 T1 掃 delay 用一條 `inc_ref r2` 把掃描值併進參考時間，之後的 readout config、trigger、readout pulse 都回到字面值，共 28 條。若同樣的序列留在具名 timeline 上，參考時間不動，掃描 offset 帶進之後每條指令的時間，需要 4 個時間暫存器與對應的搬移、更新、歸位，共 40 條。

因此使用者採用兩項建議（「Timeline的12納入設計，採用建議」）：主流程放在預設 timeline，重疊的 pulse 放具名 timeline；提供 `sync(timeline=...)`，用一次 `inc_ref` 把參考時間移到指定 timeline，其他 timeline 一起換算。

### blocking 與推進長度

使用者：「我考慮移除block屬性，改利用timeline來實現重疊」，之後確認「1要移除」。module 發射後一律推進自己所在的 timeline，重疊全部以 timeline 表達；現行用 `block_mode=False` 的三個實驗（兩個 ac_stark 與 cpmg）遷移時改寫。

推進長度比較了三種：(a) 沿用 `total_length`，pre／波形／post 各自進位，可與現行逐位相同；(b) 取 QICK 記錄的實際結束時間，整段向上取整一次；(c) 同 (b) 但照 QICK 原生四捨五入，原型測得最差提早 1.1 ns，違反「下一個不得早於上一個播完」，否決。使用者問「選b是不是比較自然？ A的優點是什麼？」。(a) 的優點是遷移時可用 `prog_list` 逐位比對、不依賴 QICK 捨入細節；(b) 語意與原生 `delay_auto` 一致、時間只有 QICK 一個來源（不再自算 `total_length`，模擬器也不重算），且較緊湊（T1 的 pi 從 @42 提前到 @39）。

定案（「採用建議」）：採 (b)，遷移分兩步。先以 (a) 規則把 module 轉成 macro，用逐位比對證明轉換不改行為；再單獨切換到 (b)，為每個實驗產出 timeline 差異報告，只允許提前且每個元素最多數個 cycle。最終程式只留 (b)。

### 仍待處理

`wait_auto`／`resync` 對具名 timeline 的語意、多 readout 或 readout 窗口長於 pulse 時 post-delay 的基準、巢狀 tag、同一 instance 多次發射（heralding 的 `occurrence`）、QICK 版本耦合，以及 Repeat、Branch、LoadValue、TableLengthPulse 等的轉換，留給實作 spec。

## 後記五（2026-10-29 晚，A 階段細節與 builder 重新設計）

### A 階段的前置決定

- 新基底直接繼承 QICK 的 program，dmem 緩衝與 route 保護改為新舊共用；可以覆寫 QICK 方法，以可維護為準（使用者：「可以適當的重寫qick方法，只要夠好維護」）。
- reps 與 shot body 的辨識：覆寫 `make_program`，照 QICK 的順序執行，在初始化結束、body 開始、body 結束插入不產生指令的邊界標記，迴圈結構讀 `self.loops`；模擬器只解讀兩個標記之間，外殼的 `final_wait`／`final_delay` 等只驗型別。
- body 內在具名 timeline 上不支援 `wait_auto`／`resync`（第一版報錯）；`make_program` 外殼的 `final_wait` 不受限。
- program 層的控制結構用 `with`：`with self.join() as j: with j.branch(): ...`、`with self.repeat(...)`、`with self.branch(by=...) as b: with b.case(): ...`；dmem 表在 `_initialize` 宣告、body 使用（使用者：「A可以，示意寫法沒問題」）。
- retry、early stop、進度屬於 builder 的功能，自己寫 program 的作者自行實作（使用者：「retry/early_stop等可以算是builder提供的功能，想自己寫program的作者要自己實現對應功能」）。

### builder 重新設計

使用者：「我覺得新原生寫法的可讀性比目前builder的寫法好，對於包裝的工具來說這是很尷尬的，可以考慮重新設計」。現行 builder 做四件事：描述序列（`.add(...)`）、宣告掃描（`declare_sweep`）、執行取數（rounds、retry、early stop、進度、Stop、Schedule 整合）與隔離（deepcopy cfg）。

討論過的方向：

- builder 只當執行器、序列一律寫 program：使用者指出「Builder的優勢是不用init與body分開，以及inline的pulse sequence宣告」，不採。
- 在 program 層 emit 時自動宣告：使用者「我覺得不太好，用program寫法就該接近原生的寫法」，不採。
- 採用：「builder才該設計成一個with scope，內部是像body的寫法」。

```python
with sched.builder(soc, soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay) as b:
    length = b.sweep("length", cfg.length)        # 範圍內任何位置宣告
    b.emit(Reset("reset", cfg.reset))
    b.emit(Pulse("pi_pulse", cfg.pi_pulse))
    b.delay(length)
    b.emit(Readout("readout", cfg.readout))
# 離開 with：依記錄產生 program——宣告放 _initialize、掃描變成迴圈、發射依序放 _body

data = b.acquire(rounds=cfg.rounds, retry=cfg.acquire_retry, stop=...)
```

使用者確認（「可以」）：program 層接近原生、不自動宣告；builder 是先記錄、離開範圍時才產生 program 的 `with` 範圍，錯誤在離開時報出；`acquire` 寫在範圍之後；builder 範圍內的方法與 program 層完全同名（含 `timeline=` 與 `join`／`repeat`／`branch`）；新 builder 與既有實驗的改寫排在模擬器改讀 macro 之後，逐一以 `prog_list` 比對遷移。

## 後記六（2026-10-29 晚，A 階段實作中的修正與 timeline 模型）

依據 repo@f93a2d859 與 QICK 0.2.436。A 階段實作的檢查結果與測試證據在未入庫的工作檔，必要事實寫在本節。

### module 身分與作者入口

- module 是宣告物件。作者在 `_initialize` 建立 module 並用 `self.declare(module)` 宣告，在 `_body` 用 `self.emit(module, timeline=...)` 發射。每次發射產生一個新的 emission macro，時間與 tag 屬於那次發射，同一物件可以發射多次。module 物件本身不放進 macro 序列（使用者：「那用A好了，但代價應該就是會出現很多新macro與module」）。
- 所有 module 共用一個通用 emission macro，各 module 提供自己的 lowering，類別數不會加倍。只有 preprocess 或 expand 行為真的不同時才另立 macro 類別。
- 曾考慮讓 `declare` 直接收 cfg，作者不建 module 物件。現有 cfg 型別只涵蓋 Pulse、Readout、Reset 三族；`Pulse` 另有不在 cfg 裡的參數；`TableLengthPulse`、`LoadWord` 等需要長度表、index 暫存器等非 cfg 參數。所以沒有一致的 `declare(cfg)`。使用者：「A'的機制可能比較複雜，收益相比A比較小，除非有很一致且簡單的declare機制，否則還是顯式module好了」。
- 新基底改為繼承現有的 `MyProgramV2`。第一版直接繼承 QICK 的 program，少了 IR 編譯管線、模擬取數與分類取數的擴充，同一段原生寫法編出的結果和現有 program 不同。constructor 沿用 `MyProgramV2` 的 `reps`、`initial_delay`、`relax_delay`。
- module 的方法在 A 階段新舊並存，新路徑用 `declare` 與 `lower`。`init`、`run`、`block_mode` 在刪除舊路徑時移除。

### 驗收基準改為預期行為

後記四的第一步要求新路徑和現行 module 版逐位相同。實作時遇到三個反例：

- Pulse 後接掃描的 Delay：新路徑每個 module 實際推進一次參考時間，多一條 `inc_ref`。舊路徑在 Python 層把 cursor 和掃描值相加，只有一條。
- SoftDelay：舊路徑用兩個時間暫存器、25 條指令，新路徑用一個、23 條，時間相同。
- Pulse 後接 DelayAuto：新路徑晚 1 個 cycle。舊路徑對 pulse 結束點有兩種算法：cursor 用 `total_length` 向上取整得 44，DelayAuto 卻忽略 cursor，改用 QICK timestamp 四捨五入得 43。後者讓間隔比作者要求的短約 56 ps。

要在 IR 還原舊的暫存器排法，必須讀 QICK 標為內部用途的 sweep 資料，並在編譯期重建 cursor。使用者：「第一件選A，語意與行為一致就可以了，舊設計的怪癖不一定要對齊」「我沒有要求必須與舊的一致，我要求的是與預期行為一致」。

定案：

- 正式測試用新的 IR 時序計算器讀編譯結果，包括暫存器初值、迴圈步長、`inc_ref` 與時間欄位，逐掃描點檢查每個 pulse、trigger 的絕對 cycle 和 shot 結束點，是否等於照 timeline 規則手算的值。
- 新舊比較改成一次性的遷移差異報告，每項差異標明是舊設計的怪癖還是新路徑的缺陷，缺陷要修。
- 指令數和暫存器排法不是驗收項目。合併掃描推進的 IR 優化列為待辦。
- 原生 `resync` 在硬體上讀當下時間再推進，編譯結果無法決定推進量。計算器遇到時報錯，不假設為零。
- 規則 (b) 從最後一步提前到 B 階段，因為 B 本來就要改完所有 module 並移除 block。A 階段沿用 (a)，最後階段只刪舊路徑（使用者：「採用建議」）。

### timeline 模型：不綁定參考時間，只存結尾

使用者指出：「原生qick其實有timeline機制，只是其timeline固定綁定到channel, 且與ref_time是分開的，delay操作的是ref time，然後計算每個ch的timeline的等校尾端時間，而我們新設計的timeline，其實就是不綁定channel的原生timeline」。照這個角度，QICK 的參考時間本身不是一條時間線，後記三第 1 點「預設 timeline 就是參考時間」是我們額外加的。使用者補充「只要存當前結尾就好，不用存開始」，並選擇預設 timeline 不綁定參考時間。

定案的模型取代後記三的第 1、2、4 點，以及後記四的「主流程用預設 timeline」：

- 原生方法完全照 QICK。`delay`、`delay_auto`、`wait_auto`、`resync` 不接受 timeline 參數，只移動參考時間（使用者：「還是保留原生delay的行為，用delay_timelime來實現指定timeline的操作？」）。
- 每條 timeline 只存一個數字：相對參考時間的目前結尾，下限 0，和 QICK 的 channel timestamp 同義。名為 `"default"` 的預設 timeline 也一樣。參考時間移動 d 時，每條 timeline 變成 max(0, 結尾 − d)。
- `emit(m, timeline=X)` 以 X 的結尾作為這次發射的共同起點，放好 m 的所有事件後把 X 推到 m 的結束點。不帶 timeline 時用 `"default"`。
- 帶 timeline 的 `pulse`、`trigger` 放在 X 的結尾加 t，再把 X 推到結束點，相當於 QICK 的 `t="auto"`。不帶 timeline 時照 QICK 從參考時間起算。
- `delay_timeline(t, X)` 把 X 的結尾加 t。`sync(X)` 把參考時間移到 X 的結尾。參考時間只由原生 delay 類指令、`sync` 和下面的結構規則移動。
- 同一條 timeline 上連續的 pulse 依序排列。要並排就用 join 的分支，或另一條 timeline。
- 曾提出的 `delay_auto_timeline` 取消：timeline 只存結尾，「等 X 結束再加 t」和「X 往後推 t」變成同一件事。timeline 不分 generator 和 readout，原生 `delay_auto(gens=, ros=)` 照舊。

```python
self.emit(self.pi)       # "default" 推到 pi 結束，參考時間不動
self.delay_auto(t)       # QICK 原生：等所有 channel 結束再加 t，參考時間移動
self.emit(self.ro)       # 從 pi 結束加 t 開始
```

掃描值碰到下限 0 時，如果各掃描點的正負號不同，結果是折線，一個線性的時間暫存器表達不出來。QICK 自己的做法是：掃描的最小值大於 0 才保留整條，否則整條歸零。在 pulse 帶 `post_delay`、delay 從 0 開始掃的 T1 裡，前幾個點的間隔會少於 `post_delay`。使用者：「Qick原生寫法可能是bug，可以採用你的建議」。定案：無法逐點證明同號時在編譯期報錯，提示改用 `delay_timeline`。能編譯的程式每個點都精確。

結構規則：

- join 從母 timeline 的結尾分出各分支，離開時把各分支結尾的最大值回饋到母 timeline（使用者：「join結尾應該預期把最大時間回饋到母timeline裡」）。
- branch 每條 timeline 取各 case 結尾的最大值。某個 case 移動了參考時間時，較短的 case 補 `inc_ref`。
- repeat 的每次迭代結尾，以及每個 shot 的 body 結尾，把參考時間移到用到的 timeline 的最大結尾，這些 timeline 歸零，讓每次迭代和每個 shot 從同樣的狀態開始。body 結尾先對齊，再進外殼的 `final_wait`、`final_delay`，最後一個 module 的 `post_delay` 才不會被原生 `delay_auto` 忽略。
- 進入 repeat 時，如果有迴圈內沒用到、但還沒結束的 timeline，就報錯並提示先 `sync`，因為每圈推進參考時間都會再扣它一次。
- 由暫存器決定的推進只能用原生 `delay_reg_auto`。所有 timeline 的結尾都必須能證明不超過推進量的靜態部分，否則報錯。

module 的推進：A 階段用 `total_length`。B 階段的 lowering 以 `delay_timeline(post_delay, X)` 結尾，由 timeline 自己記錄的結束點決定推進，module 不必另外回報長度。

不做「每個 channel 一條預設 timeline」，理由有三：「等這個 channel 空了再放」原生已有 `pulse(t="auto")`；channel timeline 會被任何 timeline 上的 pulse 更新，形成第二種更新規則；如果可以寫入，又會改變原生 `delay_auto` 的等待時間。日後需要「對齊到某個 channel 空閒」時，再以唯讀來源加入（使用者：「可以，採用建議」）。

### 模擬器的範圍

C 階段支援的 macro 子集包含原生宣告（`declare_gen`、`add_pulse`、`declare_readout`），原生的 `pulse`、`delay`、`delay_auto`、`trigger`、`send_readoutconfig` 與 `add_loop` 掃描，以及 `delay_timeline`、`sync`、我們的 emission macro 和 `join`、`repeat`、`branch`、`declare_table`、`load_word`、`scan_with`。runtime 暫存器運算與跳轉（包括依量測結果分支的 `read_and_jump`）、`asm_inst`、table macro 以外的 dmem 讀寫、`wait`、`wait_auto`、`resync` 會在產生資料前報錯，並指出 macro 類型與位置（使用者：「採用建議」）。時間直接讀 QICK 解析後的值。

現有實驗都只由 module 組成，直接操作暫存器、dmem 與跳轉的程式碼只出現在我們的 module 裡，所以沒有實驗會因此變成無法模擬。C 階段的驗收要求現有每種 module 的 macro 形式都能模擬。

### 仍待處理

- 掃描值放在具名 timeline 上，之後每個 pulse 都要用暫存器給時間（後記四的 28 條對 40 條）。主流程的掃描等待建議用原生 `delay_auto(sweep)`，要寫進作者指引，也可以做成 IR 優化。
- B 階段 spec：其餘 module 的 lowering、移除 block、規則 (b) 與差異報告。
