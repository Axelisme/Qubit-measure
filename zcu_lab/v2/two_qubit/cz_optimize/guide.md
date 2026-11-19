**讀者：** operator、agent。準備雙線 CZ 的起點，在 GUI 或 MCP 執行形狀優化並判讀結果。

# CZ 形狀優化

`two_qubit/cz_optimize` 以固定 depth 的 interleaved RB 搜尋 CZ 形狀。每個隨機 Clifford 後接 CZ，最後播放整條 interleaved 序列的 inverse。每次評估重用相同的 seeds，目標是在終端讀到 F1、F2 都是 g 的比例，先按 seed 計算，再對 seeds 平均。

CMA-ES 調 I、Q 各 6 個內部控制點，以及 F1、F2 各一個 local Z 角度。雙線播放同一個 shape。各線的帶號 gain、載波頻率、相位與延遲都不變。流程參考 Leon Ding, *Novel Gates with Superconducting Fluxonium Qubits*（2023）§5.3.4。

## First-round preparation（第一輪準備）

1. 依 [two-qubit 校準指南](../guide.md) 準備 F1／F2 的 X90、完整 mux readout、兩個 role tone 的 GE circle classifier 與 radius，以及雙線 CZ。核對 `f1`、`f2`、`coupler`、`mux` 四個 alias。Tone 位置由 mux group 的 qubit 順序決定。
2. 在 GUI 的 arb 波形對話框，用已校準的 CZ 形狀和時長建立一個 arb 資產。Preview 核對 I、Q、時長與 normalize 設定。把 `cz_f1` 與 `cz_f2` 都設成 `arb` 並選同一份資產，保留其他 pulse 欄位。兩個 slot 指向同一個形狀的兩線 pulse，不是把兩線指向同一個 pulse。
3. 用該 arb 重跑 [CZ amplitude](../cz_amplitude/guide.md) 與 [local Z](../local_z/guide.md)。確認 `$coupler.ext.cz_phase` 裡 F1、F2 的名稱和 correction 都屬於這次校準。
4. Open `two_qubit/cz_optimize`。若 tab 已開啟，按 Reset Form 重新捕捉。核對各 slot、mux tone 順序與 local Z。兩線指向不同的 asset path 時，即使樣本相同，表單也會拒絕執行。

評估 0 不由 CMA 抽樣。它以 arb 的相同時長為準，在等距的 6 個內部節點從資產線性內插 I、Q 值，並使用捕捉到的 local Z 角度。再用 natural cubic spline 連接這些點與兩端的零值。這條樣條不是原始資產的複本。即使沒有改控制點，兩端及節點之間的形狀也可能和資產不同。

## 設定 Run

| 欄位 | 設定與效果 |
| --- | --- |
| `x90_f1`、`x90_f2` | 分別捕捉 `$f1.module.x90`、`$f2.module.x90`。每個 X90 使用與同角色 CZ pulse 相同的 generator。 |
| `drive`、`phase_pair` | 捕捉 `$coupler.module.cz_f1`、`$coupler.module.cz_f2` 與 `$coupler.ext.cz_phase`。 |
| `readout`、`roles` | 捕捉 `$mux.module.readout` 的完整 tones 與 F1／F2 的原始 tone 位置。 |
| `reset` | 預設不設。可選脈衝 reset。若選 `reset/mux_herald`，用同一份完整 mux readout，並把 F1、F2 的 tone 位置都設為 g，例如 `{"0": "g", "1": "g"}`。 |
| `depth` | 每條序列的隨機 Clifford 數，至少 1。先用 [paired RB／IRB](../rb/guide.md) 估計 survival，再選落在衰減中段的 depth。太淺時，多數結果接近起始值。太深時，多數結果接近底值。 |
| `seed` | 3–5 個不同的整數，範圍 0 至 2³¹−1。每次評估使用同一組序列。 |
| `generations`、`popsize` | CMA 世代數至少 1，每代候選數至少 2，預設每代 11 個。 |
| `sigma_knot` | I、Q 控制點的初始標準差，normalized 單位，必須大於 0。控制點限制在 −1 到 1。 |
| `sigma_z_deg` | 兩個 local Z 角度的初始標準差，單位 deg，必須大於 0。角度不設上下限。 |
| `cma_seed` | 控制 CMA 候選抽樣的非負整數。它和選 Clifford 序列的 `seed` 是不同設定。 |
| `reps`、`initial_delay`、`relax_delay` | 每次評估每個 seed 的 shots、第一發前的等待和每發後的等待。時間單位是 µs。 |

低 SNR 時用脈衝 reset 或不設 reset，避免 herald 的分群錯誤影響篩選。有 herald 時，各 seed 的分母是通過篩選的 shots。沒有 herald 時，分母是全部 shots。分母都包含終端分類落在 other 的 shots。若某個 seed 沒有通過 herald 的 shots，Run 會報出 seed 和接受數。先核對 tone 位置、keep 與讀出分群，再調整設定。

## 成本、停止與保存

總共嘗試 `1 + generations × popsize` 次評估。每次可行評估對每個 seed 編譯並取數一支 program，最多取 `評估數 × seed 數 × reps` 發。例如 2 代、每代 4 個候選、3 seeds、256 reps，最多 27 支 program、6912 發。離線 FTF 在 depth 5 和 4 workers 時，每支約 10 秒，可用 program 數估算等待時間。實機時間還受 depth、reps、編譯與等待設定影響。

Run 把候選樣條放在記憶體內播放，不為候選建立 library 資產。如果樣本的最大 `|I+iQ|` 大於 1，跳過取數並將目標 0 回報給 CMA。Live 圖的橫軸是評估序號，圓點是本次目標，虛線是到目前為止最佳的**實測**目標。跳過的候選顯示為 0，但不進入最佳值。

Stop 保留已取得的計數，不把未完成的世代回報給 CMA。Save 記錄本次 cfg、各次評估的 14 個參數、峰值及每個 seed 的 3×3 終端計數。未取得的計數為 NaN。結果檔沒有候選波形樣本。保留 cfg 所用的起點資產及原有 path，之後 Load 再 Analyze 會使用該資產的時長。

## Analyze 與驗收

用 Analyze 讀取目前 Run 或 Load 的結果。圖 `fit` 左側顯示實測的目標和最高點，略過不可行或未完成的評估。右側疊加最佳及起點的 normalized I、Q，橫軸是 µs。摘要列出最佳序號、最佳及起點的目標，以及依 I1… I6、Q1… Q6、F1／F2 local Z 排序的 14 個數字。若缺少 cfg 或沒有可用評估，摘要回報原因，不選最佳。

最高的單次 survival 會因多次比較而偏樂觀。判斷 gate 是否改善，要用 [two_qubit/rb](../rb/guide.md) 重測 paired RB／IRB。使用與優化不重疊的 seeds 及至少 3 個 depths，依驗收的 gate error 與不確定度判斷，不用優化曲線的最高點代替。
