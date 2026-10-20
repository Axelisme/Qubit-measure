# Local Z

Readout 有 `$mux.module.readout` slot 時引用 library 的完整 tones 與 scalar 設定。缺 slot 時從 blank mux 範本開始，須手填全部必要讀出欄位。

在幅度與 frequency 選定後，選 `ftf/local_z`。同一次 Run 分別量 F1、F2。每個 target 從 `B(00)` 熱初態做 X90、n 次雙線 CZ、projection 與完整 mux readout，另一 qubit 不翻轉。X/Y projection 是兩次獨立 acquisition，不是同 shot 連續量測。沒有 heralding、shot 排除或純態初始化。

## Run 前

1. 綁定 scalar aliases `f1`、`f2`、`coupler`、`mux`。F1/F2 須為不同 fluxonium，coupler 是 transmon，mux 的 qubits 須包含兩者，並與 resonators 逐項對應。
2. Open／Reset 固定捕捉角色與 library。核對兩角色 X90、coupler 的 `cz_f1`／`cz_f2`、完整 mux tones 與各自 GE centers。缺 CZ 或 X90 不能 Run。缺 GE centers 可收 raw，但不能求該 target 的 phase。
3. 兩 CZ pulses 必須用不同 generator，同 frequency、duration、pre-delay。保留已校準的 signed gain 比例、relative phase 與 waveform。X90 generator 必須對應自己的角色。F1/F2 不是 generator 排序，也不是 ADC 編號。
4. 設 gate_counts、reps、rounds。n 至少三點，必須是連續、嚴增、非負整數。預設 `[0,1,2,3,4]`。n=0 是不播 CZ 的基線，不補理想資料。

Run 不重讀目前 aliases 或 library。要改角色或來源，重新 Reset。所有 tones、same-shot 關係及 round-major shots 原樣保存；raw shape 是 `(target, projection, n, rounds*reps, tone)`。取消後已完成 round 留在固定 slot，未完成位置是 NaN，final conversion 不重複 append。

## Phase 與 slope

Preparation 是來源 X90 的 phase φ。X projection 用同 φ 的 X90，Y projection 用 φ+90° 的 X90。分析以來源 GE centers 線性投影取得 P1，再由 qX=2P1X−1、qY=2P1Y−1 求 logical Bloch phase `atan2(qY,qX)`，degree wrap 到 `[-180,180)`。不做 confusion matrix inversion，也不把熱混合當純態。

每個 target 必須有整列完整的 X/Y phases。缺任一點、缺任一完整 round pair、缺 GE centers 或零 contrast 時，該 target 不求 slope，不跨缺口 unwrap。微弱但非零 contrast 不會自動被排除。看 contrast、quadratures 與曲線判斷是否需要重測。

分析對完整列做相鄰主值 unwrap，再用含 intercept 的 ordinary least squares 擬合 `phase_deg = intercept + slope*n`。兩個 target 各自擬合。Slope 是每 CZ 的 logical Bloch phase accrual，correction 是它的負值，不再 wrap。零 slope 可以是有效結果，缺失分量不是零。

此解讀假設相鄰真實 phase 增量滿足 `|Δθ| < 180°`，無法辨認每 gate 額外的 360°。n 連續也不能消除此 alias。操作員要自行確認此假設；分析不新增自動 sampling gate。

`phase` 圖顯示兩 target 的 contrast、wrapped phase、unwrapped phase 與線性 fit。`quadratures_tone_i` 顯示完整每 tone 在兩 target、兩 projection 的平均 I/Q。Summary 保留每點原因、acquired shots、intercept、slope、correction、slope 1σ error 與既有 `fit_quality`。Covariance 以線性模型的 residual variance 估計，不是 shot-noise 或 phase uncertainty 模型。零 residual 不代表物理誤差為零。沒有 χ²、contrast/σ 或 fidelity 門檻，模型適用性由操作員看 residual 與 uncertainty 判斷。

熱態的其他成分仍在資料內，phase 可以偏離線性模型。這是熱 ensemble 的模型估計，不能稱為已純化分支的 local phase，也不能由此推 gate fidelity。

## Preview 與手動 Apply

只有 F1/F2 都得到有效 correction，才提出一筆 `$coupler.ext.cz_phase` compound parameter。它包含 `f1`、`f2` component 名稱與 `correction_f1_deg`、`correction_f2_deg`。缺值不補 0，不拆成兩筆獨立分量。先 preview，再勾選這一列按 Apply Selected。FIT 本身不寫檔。

提案只讀 source cfg 與同次 Run／Load capture。若 F1/F2/coupler 與來源不同，拒絕提案，包括交換 F1/F2。後續消費 phase pair 時也須核對 component 名稱，不能靜默交換沿用。

Correction 用於後續 single-qubit logical frame，不播額外 Z pulse，也不改 CZ 的 constructive relative phase。Program pulse phase 的正負換算要沿既有 frame convention，不能直接假設與 logical Z 同號。Apply 這一列不改任何 pulse、slot 或其他 parameter。

Native Save／Load 保留 raw、axes 與 source cfg。Load 不回填目前 Form，Analyze 不拿目前 Form 補歷史校準。缺 source cfg 可保留 raw，但 analysis／writeback 明確拒絕。改 pulse amplitude、frequency、duration、waveform、relative phase 或 DC 工作點後，要重測 local Z；沒有自動 staleness 系統。

完整操作順序見 [FTF bring-up guide](../guide.md)。本實驗的離線合成走查不等於 FTF 控制鏈 smoke，實際 smoke 另驗，不宣稱論文 fidelity。
