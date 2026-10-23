# Local Z

**讀者：** operator、agent。從 GUI 或 MCP 量兩個 target 的 local-Z slope，審閱並套用 correction pair。

**Last updated:** 2026-10-21

Readout 有 `$mux.module.readout` slot 時引用 library 的完整 tones 與 scalar 設定。缺 slot 時從 blank mux 範本開始，須手填全部必要讀出欄位。

在幅度與 frequency 選定後，選 `two_qubit/local_z`。同一次 Run 分別量 F1、F2。每個 target 從 `B(00)` 熱初態做 X90、n 次雙線 CZ、projection 與完整 mux readout，另一 qubit 維持原熱態。X/Y projection 分別獨立 acquisition，所有 shots 保留原始熱 population。

## Run 前

1. 綁定 scalar aliases `f1`、`f2`、`coupler`、`mux`。F1/F2 須為不同 fluxonium，coupler 是 transmon，mux 的 qubits 須包含兩者，並與 resonators 逐項對應。
2. Open／Reset 固定捕捉角色與 library。核對兩角色 X90、coupler 的 `cz_f1`／`cz_f2`、完整 mux tones 與各自 GE centers。CZ 與 X90 是 Run 的必要輸入。缺 GE centers 時可收 raw，該 target 的 phase 分析則回報缺校準。
3. 兩 CZ pulses 必須用不同 generator，同 frequency、duration、pre-delay。保留已校準的 signed gain 比例、relative phase 與 waveform。X90 generator 必須對應自己的角色。角色身分由 component 綁定與 mux qubits 順序決定。
4. 設 gate_counts、reps、rounds。n 至少三點，必須是連續、嚴增、非負整數。預設 `[0,1,2,3,4]`。n=0 直接量無 CZ 的實測基線。

Run 全程使用 Open／Reset 固定的來源。要改角色或來源，重新 Reset。所有 tones、same-shot 關係及 round-major shots 原樣保存；raw shape 是 `(target, projection, n, rounds*reps, tone)`。取消後已完成 round 留在固定 slot，未完成位置是 NaN，每個已取得 round 只計一次。

## Phase 與 slope

Preparation 是來源 X90 的 phase φ。X projection 用同 φ 的 X90，Y projection 用 φ+90° 的 X90。分析以來源 GE centers 線性投影取得 P1，再由 qX=2P1X−1、qY=2P1Y−1 求 logical Bloch phase `atan2(qY,qX)`，degree wrap 到 `[-180,180)`。線性投影保留原始讀出效果與熱混合。

每個 target 必須有整列完整的 X/Y phases。缺任一點、缺任一完整 round pair、缺 GE centers 或零 contrast 時，該 target 回報 slope 不可辨識並保留原因。Unwrap 只處理完整列；微弱但非零 contrast 仍可進入分析。看 contrast、quadratures 與曲線判斷是否需要重測。

分析對完整列做相鄰主值 unwrap，再用含 intercept 的 ordinary least squares 擬合 `phase_deg = intercept + slope*n`。兩個 target 各自擬合。Slope 是每 CZ 的 logical Bloch phase accrual，correction 是它的負值，保留原數值。零 slope 可以是有效結果，缺失分量則維持缺失。

此解讀假設相鄰真實 phase 增量滿足 `|Δθ| < 180°`，每 gate 額外的 360° 在連續 n 採樣下仍無法辨認。操作員須自行確認採樣假設。

`phase` 圖顯示兩 target 的 contrast、wrapped phase、unwrapped phase 與線性 fit。`quadratures_tone_i` 顯示完整每 tone 在兩 target、兩 projection 的平均 I/Q。Summary 保留每點原因、acquired shots、intercept、slope、correction、slope 1σ error 與既有 `fit_quality`。Covariance 以線性模型的 residual variance 估計，它只描述線性模型的擬合誤差，shot-noise 與 phase uncertainty 須另行評估。零 residual 仍有物理誤差的可能。模型適用性與統計診斷由操作員看 residual、uncertainty 與資料判斷。

熱態的其他成分仍在資料內，phase 可以偏離線性模型。將結果解讀為熱 ensemble 的模型估計；純化分支的 local phase 與 gate fidelity 須另行驗證。

## Preview 與手動 Apply

只有 F1/F2 都得到有效 correction，才提出一筆 `$coupler.ext.cz_phase` compound parameter。它包含 `f1`、`f2` component 名稱與 `correction_f1_deg`、`correction_f2_deg`。任一分量缺失時停止整筆提案並回報原因。先 preview，再勾選這一列按 Apply Selected。只有 Apply Selected 才寫入校準。

提案只讀 source cfg 與同次 Run／Load capture。若 F1/F2/coupler 與來源不同，拒絕提案，包括交換 F1/F2。後續使用 phase pair 時也須核對 component 名稱與順序。

Correction 以 virtual Z 更新後續 single-qubit logical frame，CZ 的 constructive relative phase 維持原值。Pulse phase 的正負換算須依既有 frame convention 核對。Apply 這一列只更新 phase pair，pulse、slot 與其他 parameter 都維持原值。

Native Save／Load 保留 raw、axes 與 source cfg。Load 保持目前 Form 原值，Analyze 只用保存來源。缺 source cfg 時保留 raw，分析與提案則回報來源缺失。改 pulse amplitude、frequency、duration、waveform、relative phase 或 DC 工作點後，要重測 local Z；校準時效由操作員確認。

完整操作順序見 [FTF bring-up guide](../guide.md)。實際 FTF 控制鏈與 gate 表現須用現場資料另行驗證。
