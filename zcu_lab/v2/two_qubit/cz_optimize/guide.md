**讀者：** operator、agent。準備雙線 CZ 的起點，在 GUI 或 MCP 優化、寫回並用 IRB 驗收。

# CZ 形狀優化

`two_qubit/cz_optimize` 以固定 depth 的 interleaved RB 搜尋 CZ 形狀。Run 按 seed 抽一次該 depth 的 Clifford，之後各候選都重用這組序列。每發依序播放可選的 reset、每個 Clifford 和其後的 CZ、整條序列的 recovery、終端讀取；沒有板上 depth 跳轉。目標是終端 Qubit1、Qubit2 都為 g 的比例，先按 seed 計算，再對 seeds 平均。

CMA-ES 調 I、Q 各 6 個內部控制點，以及 Qubit1、Qubit2 各一個 local Z 角度。雙線播放同一個 shape。各線的帶號 gain、載波頻率、相位與延遲都不變。流程參考 Leon Ding, *Novel Gates with Superconducting Fluxonium Qubits*（2023）§5.3.4。

## First-round preparation（第一輪準備）

1. 依 [two-qubit 校準指南](../guide.md) 準備 Qubit1／Qubit2 的 X90 和 X180、完整 mux readout、兩個 role tone 的 GE circle classifier 與 radius，以及雙線 CZ。核對 `qubit1`、`qubit2`、`coupler`、`mux` 四個 alias。Tone 位置由 `$mux.qubits` 的順序決定，清單須包含兩個不同的 qubit 元件。沿用舊 point 時，重綁兩個 qubit alias，並將 CZ slot 與 pulse library key 改為 `cz_qubit1`、`cz_qubit2`。
2. 在 GUI 的 arb 波形對話框，用已校準的 CZ 形狀和時長建立一個 arb 資產。Preview 核對 I、Q、時長與 normalize 設定。把 `cz_qubit1` 與 `cz_qubit2` 都設成 `arb` 並選同一份資產，保留其他 pulse 欄位。兩個 slot 指向同一個形狀的兩線 pulse，不是把兩線指向同一個 pulse。
3. 用該 arb 重跑 [CZ amplitude](../cz_amplitude/guide.md) 與 [local Z](../local_z/guide.md)。套用 local Z 的寫回，讓 `$coupler.ext.cz_phase` 保存 `qubit1`、`qubit2` 與 `correction_qubit1_deg`、`correction_qubit2_deg`。Open／Defaults 會比對元件名及順序，不符時會指出位置。
4. Open `two_qubit/cz_optimize`。若 tab 已開啟，按 Defaults 重新捕捉。核對各 slot、mux tone 順序與 local Z。兩線指向不同的 asset path 時，即使樣本相同，表單也會拒絕執行。

評估 0 不由 CMA 抽樣。它以 arb 的相同時長為準，在等距的 6 個內部節點從資產線性內插 I、Q 值，並使用捕捉到的 local Z 角度。再用 natural cubic spline 連接這些點與兩端的零值。這條樣條不是原始資產的複本。即使沒有改控制點，兩端及節點之間的形狀也可能和資產不同。

## 設定 Run

| 欄位 | 設定與效果 |
| --- | --- |
| `x90_qubit1`、`x90_qubit2` | 分別捕捉 `$qubit1.module.x90`、`$qubit2.module.x90`。 |
| `x180_qubit1`、`x180_qubit2` | 分別捕捉 `$qubit1.module.x180`、`$qubit2.module.x180`。 |
| `drive`、`phase_pair` | 捕捉 `$coupler.module.cz_qubit1`、`$coupler.module.cz_qubit2` 與 `$coupler.ext.cz_phase`。 |
| `readout` | 捕捉 `$mux.module.readout` 的完整 tones。Open／Defaults 將兩個原始 tone 位置填入隱藏的 `qubit1_tone`、`qubit2_tone`。 |
| `reset` | 預設不設。可選脈衝 reset。若選 `reset/mux_herald`，用同一份完整 mux readout，並把 Qubit1、Qubit2 的 tone 位置都設為 g，例如 `{"0": "g", "1": "g"}`。 |
| `depth` | 每條序列的隨機 Clifford 數，至少 1。先用 [paired RB／IRB](../rb/guide.md) 估計 survival，再選落在衰減中段的 depth。太淺時，多數結果接近起始值。太深時，多數結果接近底值。 |
| `seed` | 3–5 個不同的整數，範圍 0 至 2³¹−1。每次評估使用同一組序列。 |
| `generations`、`popsize` | CMA 世代數至少 1，每代候選數至少 2，預設每代 11 個。 |
| `sigma_knot` | I、Q 控制點的初始標準差，normalized 單位，必須大於 0。控制點限制在 −1 到 1。 |
| `sigma_z_deg` | 兩個 local Z 角度的初始標準差，單位 deg，必須大於 0。角度不設上下限。 |
| `cma_seed` | 控制 CMA 候選抽樣的非負整數。它和選 Clifford 序列的 `seed` 是不同設定。 |
| `reps`、`initial_delay`、`relax_delay` | 每次評估每個 seed 的 shots、第一發前的等待和每發後的等待。時間單位是 µs。 |

用 Notebook 組 cfg 時，填完整 readout 的 `qubit1_tone`、`qubit2_tone` 索引。Notebook 內建角色也叫 `qubit1`、`qubit2`，但不會自動綁定 GUI aliases。

每支 X90 和 X180 都要使用同位置 CZ pulse 的 generator，並提供有限的 scalar phase。Clifford 分解先最少 CZ，再最少實體 X 脈衝；一個 X180 算一支校準脈衝。播放相位為設定值減去該角色的 virtual-Z frame。

低 SNR 時用脈衝 reset 或不設 reset，避免 herald 的分群錯誤影響篩選。有 herald 時，各 seed 的分母是通過篩選的 shots。沒有 herald 時，分母是全部 shots。分母都包含終端分類落在 other 的 shots。若某個 seed 沒有通過 herald 的 shots，Run 會報出 seed 和接受數。先核對 tone 位置、keep 與讀出分群，再調整設定。

## 成本、停止與保存

總共嘗試 `1 + generations × popsize` 次評估。每次可行評估對每個 seed 編譯並取數一支 program，最多取 `評估數 × seed 數 × reps` 發。例如 2 代、每代 4 個候選、3 seeds、256 reps，最多 27 支 program、6912 發。離線 FTF 在 depth 5 和 4 workers 時，每支約 10 秒，可用 program 數估算等待時間。實機時間還受 depth、reps、編譯與等待設定影響。

Run 把候選樣條放在記憶體內播放，不為候選建立 library 資產。如果樣本的最大 `|I+iQ|` 大於 1，跳過取數並將目標 0 回報給 CMA。Live 圖的橫軸是評估序號，圓點是本次目標，虛線是到目前為止最佳的**實測**目標。跳過的候選顯示為 0，但不進入最佳值。

Stop 保留已取得的計數，不把未完成的世代回報給 CMA。Save 記錄本次 cfg、各次評估的 14 個參數、峰值、每個 seed 的 3×3 終端計數，以及 Run 開始時讀到的起點資產時長。未取得的計數為 NaN。結果檔不含候選波形樣本。Load 後，Analyze 和寫回提案用保存的時長重建形狀，不讀起點波形檔。若要用原 cfg 再 Run，仍需要起點資產。

Cfg schema 為 3.0。新版 Load 拒絕軸名不相容的舊結果檔，不改動原檔。要讀取原檔，請使用對應舊版。沒有 cfg 時，Analyze 不選最佳，也不提供寫回。

## Analyze 與寫回

用 Analyze 讀取目前 Run 或 Load 的結果。圖 `fit` 左側顯示實測的目標和最高點，略過不可行或未完成的評估。右側疊加最佳及起點的 normalized I、Q，橫軸是 µs。摘要列出最佳序號、最佳及起點的目標，以及依 I1… I6、Q1… Q6、Qubit1／Qubit2 local Z 排序的 14 個數字。若缺少 cfg 或沒有可用評估，摘要回報原因，不選最佳。

有可用的最佳評估時，writeback pane 會列出六筆提案：

| 筆數 | 提案 | Apply 結果 |
| --- | --- | --- |
| 1 | `cz_opt_<hash>` arb 資產 | 發布最佳 I/Q 樣本。`<hash>` 是樣本雜湊的前 12 個十六進位字元。 |
| 2 | `$coupler.pulses.cz_qubit1`、`$coupler.pulses.cz_qubit2` | 把 Run 的兩支 pulse 換成新資產，保留各自的 generator、gain、載波、相位與延遲。 |
| 2 | `$coupler.module.cz_qubit1`、`$coupler.module.cz_qubit2` | 將兩個 slot 指向本次捕捉所解析的 pulse library keys。 |
| 1 | `$coupler.ext.cz_phase` | 保存 Qubit1、Qubit2 的元件名稱與最佳 local Z 角度，單位是 deg。 |

最佳存活率高於評估 0 時，六筆提案全部預選。持平或無法比較時，所有列預設不勾選，note 會說明原因。沒有可用的最佳評估時不提案。寫回從固定的 Run／Load capture 取得兩個 qubit 元件名及 coupler。變更目前的 aliases 不會改動目標。沒有 capture 時，寫回失敗。

先核對 Apply 的目的 entry、point、兩個 pulse 及兩個 local Z 角度。接受結果時，在**同一次 Apply** 選取六筆，讓兩個 slot 和 local Z 跟新波形一起更新。Analyze 不寫入資產或校準。相同 key 與樣本重新 Apply 時，資產回報 `existing`，其餘選取的列照常寫入。Apply 中途出錯時不回滾已完成的寫入。先檢查實際提交的列，再決定是否重試，不要用只寫入一部分的 cfg 驗收。

## 每輪優化與驗收

1. Run 優化，Analyze 並檢查六筆提案。接受這組結果才 Apply 並保存本輪 cfg 與資料。優化在同一組 seeds 上比較多個候選，最高的存活率可能偏高，不能當 gate error。
2. 開啟 plain [`two_qubit/rb`](../rb/guide.md)。若 tab 已開，先按 Defaults，確認兩支 CZ pulse 與 `cz_phase` 都是剛寫入的值。換一組與本輪優化不重疊的 seeds，並選遞增 depths。三個 depths 可檢查取數流程，估計 gate error 則至少需要四個 depths 和兩個完整的 paired seeds。
3. 讀出可區分 g/e 時，可在 RB 選 `reset/mux_herald`。使用完整 mux readout，將 Qubit1、Qubit2 對應的 tone 位置設為 g，例如兩者是 tone 0、1 時填 `{"0": "g", "1": "g"}`。SNR 低時改用脈衝 reset 或不設 reset。優化也遵循同一個 SNR 判斷，低 SNR 時用未 heralded 的終端計數。
4. 比較本輪與前一輪驗收的 gate error 和不確定度，不用優化曲線的最高點代替。若擬合沒有估計值，先排除分析回報的原因，不以該結果判定進步。驗收的 gate error 不再改善時停止。
5. 若驗收仍有改善，回到 `two_qubit/cz_optimize` 按 Defaults，讀入寫回的 CZ 與 `cz_phase`。評估 0 的 12 個控制點是上一輪最佳波形的控制點，兩個 local Z 取自保存的 `cz_phase`。以這組值開始下一輪。
