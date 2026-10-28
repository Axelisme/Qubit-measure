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
