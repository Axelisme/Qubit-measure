---
status: accepted
date: 2026-11-12
---

# 0085 — 實驗數值工具的 fitting owner

## 脈絡

部分實驗的分析同時包含資料選擇、數值擬合與畫圖。IRB 的共同 IQ 投影和成對 seed bootstrap 原本由單比特實驗持有，雙比特 RB 也需要這些計算。讓雙比特實驗 import 單比特實驗，會把共用數值能力綁在另一個實驗的目錄。

[[0072]] 將分析工具放在 `analysis`，讓各實驗保留自己的分析選擇與圖。本決策指定哪些現有計算由 `analysis.fitting` 持有，不把整個實驗分析搬進 framework。

## 決策

2026-11-12，使用者在「照建議表搬移選定工具」的方案中回答「A」。當時搬移的六組純數值計算由 `zcu_tools.analysis.fitting` 持有，沿用既有 `shared` 與 `decay` backend，不改算法。下表也列出後來納入的加權 decay fringe。

| 數值工具 | 公開輸入與責任 |
| --- | --- |
| ALLXY | Gate-pair sequence、power error、detune error 的模型入口，以及 sequence、實數訊號的擬合入口。擬合一律求 center、contrast 與兩個 coherent errors，g/e 位準可超出量測極值。 |
| IRB | `depths`、`signals`、`bootstrap_samples`、`bootstrap_seed` 的成對 decay 與 bootstrap 主入口。共同 IQ 投影另有入口，供主 fitter 與即時圖共用。 |
| 雙比特 RB | `survival`、`totals` 的成對 seed 統計入口，以及 `depths`、`means`、`errors` 的共享 decay 擬合入口，接受至少兩組，每臂共用一個 decay。 |
| Gate-count quadrature | `n`、`quadratures`、`variances`、`acquired_shots` 的共同 slope 擬合與 alias 判斷入口。 |
| Phase modulation | `phases`、`squared_rates` 的 sine／cosine 回歸、最大相位與 covariance 傳播入口。 |
| Phase slope | `n`、`phase_degrees`、`variances` 的 unwrap 與加權直線擬合入口。 |
| 加權 decay fringe | `fit_weighted_decay_fringe` 以 `delays`、`observations`、`sigmas` 做逐點加權擬合。各 trace 只共用正頻率，振幅、偏移、相位與衰減時間各自擬合；回傳具名 covariance。Lab 保留類別選擇、有效點判定、符號檢查、ζ、圖與發布。2026-11-24，designer 代裁，使用者未覆核。 |

入口只收數值陣列與既有數值或布林控制參數，不收 lab cfg、Result、Options 或 Analysis。純數值結果型別由 fitting owner 定義，caller 直接引用，不在 lab 宣告鏡像型別。擬合、統計與 IQ 投影入口回傳 fitting owner 的具名結果；ALLXY 的模型入口回傳 scalar Bloch-z。各入口的 docstring 定義 shape、axis、單位與失敗語意，內部小函式不因搬移而公開。

IRB 即時圖會傳入尚未完整的資料，不能呼叫要求四個不同 depth、兩個完整成對 seed、至少 100 次 bootstrap 的主入口。共同 IQ 投影因此另設公開入口，主 fitter 和 live callback 共用同一份實作。這是介面責任的釐清，不是放寬主擬合的接受條件。

RB 與 IRB 的數值工具同置於 [`analysis.fitting.rb`](../../lib/zcu_tools/analysis/fitting/rb.py)，由 `analysis.fitting` 匯出既有入口與結果型別。2026-11-20，使用者對合併模組的選項答「A」。

### RB 圖上的 seed 間標準差

Fitting 計算各臂各 depth 的 seed 間樣本標準差。IQ 路徑先把每個 seed 投影到與均值相同的軸。即時圖用各點已有的 seeds，擬合圖只用擬合所選的完整 seeds。不足兩個 seed 時不畫棒。這個量只供畫圖，不加入實驗 Analyze 回傳，也不取代 heralded 的 binomial 擬合權重。使用者在 2026-11-20 對顯示量、參考線和回傳的三題答「1A 2B 3A」。比較見 [RB 誤差棒](../discussions/2026-11-20-rb-errorbars.md)。

## 實驗保留的責任

Lab 選擇資料、解讀保存軸、建立完整度遮罩，並持有 herald 的 g/g、e/e 符號與逐列接受門檻。雙比特 RB 的 d=4 error／CI 換算仍由實驗解讀。IRB 主入口保留既有 d=2 ratio 與未裁切的負 error，不默默套用另一個維度。

Lab 持有 live callback、axes、畫圖與分析小摘要。雙比特 RB 的五參數 herald bootstrap 也留在 lab。它用同一批 seed identities 重抽原始 survival／totals，再呼叫統計與擬合入口，保留重複 identity 的權重。它不反向拼造 totals，也不另外實作 seed 聚合。

未選的 CKP 共享 Lorentzian 擬合留作本實驗 helper。頻譜 IQ 投影由兩個實驗各自持有。CPMG 的加權 log-log 回歸和批次 transition-rate 擬合由各實驗的 analyze 持有，進度回報也留在 caller。

## 依賴與驗證

依賴方向是 lab → `analysis.fitting`。Lib 不 import lab，實驗不透過另一個實驗取得這些數值工具，不留舊 wrapper 或 re-export。

沿用 `.importlinter` 的 `composition-roots-above-framework-and-lab`、C8、C9、C15、C16 與 `experiment-directories-independent`。這些 contract 分別約束 lab／framework 層次、能力模組無循環、core 不依賴 GUI、lib 不依賴 lab，以及實驗目錄之間的獨立性。不新增或放寬 ignore。

Fitting owner 用合成資料測公開入口，驗證參數回復、統計、covariance 與失敗語意。實驗用公開 analysis stage 比較搬移前後的同一份 typed cfg、Result、options 與 seed。數值 shape、dtype、元素與 PNG 集合須相同，每張圖另做直接判讀。量測序列與存檔格式不因這項 owner 調整而改變。

## 取捨

保留 lab analysis.py 作共用 owner，可以少搬一個檔案，但會留下跨實驗依賴。把整個 analyze 搬進 lib，則會讓 framework 認得 herald、cfg 與畫圖政策。本決策只抽取上表的數值計算，接受新增具名結果與合成回復測試的成本，讓實驗仍可直接修改自己的資料選擇與圖。
