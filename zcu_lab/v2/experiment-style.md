**讀者：** lab-author、agent。寫或改 `zcu_lab/v2` 的實驗時，照這份選範本、寫 `core.py`、做自查。

# 實驗寫法指南

需要 cfg 欄位命名、存檔軸或 Schedule 取數的契約時，查 [experiment-authoring.md](experiment-authoring.md)。

## 1. 選範本

**範本**是使用者自己寫、示範一種寫法的實驗。新實驗從最接近的範本開始，照它的結構寫：

| 範本 | 示範的寫法 |
| --- | --- |
| [`twotone/rabi/amp_rabi/core.py`](twotone/rabi/amp_rabi/core.py) | 一維硬體掃描，擬合後在 analyze 畫分析圖 |
| [`twotone/ro_optimize/freq_gain/core.py`](twotone/ro_optimize/freq_gain/core.py) | 兩個 `add_sweep` 的二維硬體掃描，用 `branch` 分 g/e |
| [`twotone/power_dep/core.py`](twotone/power_dep/core.py) | 外層電腦端迴圈、內層硬體掃描、即時熱圖 |

範本只從這張表選。表外的實驗即使物理上相近，也只參考它的物理內容，不參考寫法。使用者 2026-11-11 指示：「你要參考慣例那請參考我自己寫的實驗，不要參考agent寫的」。

## 2. 照範本的形狀寫 core.py

`core.py` 由上到下是：`Result`、`AnalyzeOptions`（有分析選項才需要）、`Analysis`、小的訊號轉換函式、`Cfg`、`Exp`（`AXES_SPEC`、`run`、`analyze`）。範本約 170 到 250 行。

`core.py` 是局部的：讀者只看這一個檔，就知道播放什麼序列、即時圖畫什麼、分析算什麼、分析圖畫什麼。

- `run` 從上往下就是實驗：算掃描軸、建即時圖、建 `SignalBuffer` 與 `Schedule`、`add_sweep`、照播放順序 `emit` 每個 pulse 與 readout、`run_acquire`、回傳 `Result`。即時圖的建立與更新函式寫在 `run` 本體。
- `analyze` 用 `zcu_tools.analysis` 的數值工具擬合，例如 `fit_rabi`。分析圖寫在 `analyze` 本體（`twotone/rabi/amp_rabi/core.py:207-227`），每個實驗各畫各的。使用者 2026-10-30 裁定：「可以共用分析工具，但畫圖就有點超過了，不利於客製化」。

## 3. 寫法判斷

**掃描。** QICK 能掃的參數用 `builder.add_sweep`，在同一個 program 裡掃完：頻率、gain、const 與 flat-top 的長度。硬體每個 round 把所有點輪一遍，慢漂移會被平均掉，也不必每點重新編譯。電腦端迴圈（`sched.scan`）用在硬體掃不了的參數，或需要逐列更新即時圖的外層，例如 `power_dep` 的 gain。

**Cfg 欄位的 description。** 用一句話寫這個欄位做什麼。它會顯示在 GUI 表單上，操作的人照它填值。推導與理由寫在實驗的 `guide.md`。範例（`twotone/rabi/amp_rabi/core.py:61`）：

```python
description="Fixed preparation pulse before each gain-swept Rabi pulse; None skips preparation."
```

**預設值。** 用範本的值：`reps=1`、`rounds=1`、`relax_delay=1.0`。

**檢查。** 範圍用 annotation 的 `ge`／`le`。其他檢查，例如 `model_validator` 或跨欄位條件，只在使用者要求時加，需要時先問。使用者 2026-11-02 裁定：「我不喜歡我沒有要求的檢查」、「如果要檢查，也是Form透過annotation的ge/le檢查」。

**回傳值。** `Result` 放要存檔的原始陣列，docstring 寫單位與 shape。`Analysis` 只放使用者要拿去用的數字：`AmpRabiAnalysis` 是 π 與 π/2 的 gain、各自的誤差，以及 fit quality。可能是 `None` 的欄位，格式化成文字前先判斷。

**抽象。** 只有一個實驗用到的邏輯，直接寫在那個實驗的 `core.py`。等第二、第三個實驗也需要時，再設計共同的函式或型別。使用者 2026-11-11 對 gain 專用 sweep 型別的判斷：「特化一個 gain sweep config，感覺就有點過度工程」。

**Docstring 與註解。** `run` 的 docstring 用二到四句：播放什麼序列、回傳什麼、使用時要注意什麼（`twotone/rabi/amp_rabi/core.py:110-116`）。每個階段前寫一行註解說明這段做什麼（`:117`、`:143`）。

## 4. 自查

交付前，下面五條都要成立：

1. 只讀這個 `core.py`，說得出播放序列、即時圖、分析計算與分析圖。
2. `lint-imports` 禁止含 `core.py` 的實驗目錄 import 其他實驗或其他家族的家族層模組，但放行自己家族的 helpers、`support` 及家族 `__init__.py` 匯出子實驗，違規時將所需內容改為本地宣告，或將共用數值邏輯移至 `zcu_tools.analysis` 或 `support`。
3. 每個 cfg 欄位的 `description` 都是一句話。
4. QICK 能掃的掃描參數都用了 `add_sweep`。
5. 每個檢查都是 annotation 的 `ge`／`le`，或使用者要求的。
