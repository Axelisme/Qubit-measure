# 色散分析數值核心

**Last updated:** 2026-09-27 — headless 前處理與參數搜尋

`analysis.dispersive` 擁有 one-tone 數值前處理結果、electrical-delay 局部精修、sample-point score 與候選參數搜尋。`compute_preprocess` 接受 flux 軸、GHz 頻率軸與形狀為 `(n_flux, n_freq)` 的複數訊號。它回傳 `PreprocessResult`，包含逐列正規化 phase difference、原始座標、逐列及中位 delay、峰值頻率中位數與 smoothing signature。`auto_tune` 接受正規化影像、sample flux、當前參數和 bounds，回傳 GHz 的候選 `(g, bare_rf)`；它不接受或寫入 fit。

`_fast_edelay` 使用 `analysis.fitting.resonance.find_edelay_branch` 尋找共同 delay branch，再由 numba 平行執行逐 flux 的 Kasa circle refinement。這條路徑不與 `analysis.fitting` 的 scalar circle fit 合併；兩者的局部圓擬合方法不同。等距頻率軸的每列結果先對齊共同 alias，再取中位 delay。前處理移除 delay，沿頻率軸以最低強度 1 做 wavelet smoothing，尋找共同圓心，計算 phase difference 的絕對值並逐列正規化。Signature 記錄 smoothing 方法、divisor 與 grid shape，供 app 判斷既有結果是否失效。

Sample-point score 經 `predict_dispersive_at` 使用 `simulate.fluxonium` prediction engine；固定解析度為 qubit dimension 15、cutoff 30、resonator dimension 4。Fast/scqubits fallback 仍由 simulation engine 決定。`sample_score` 在每個 sample flux 取 ground/excited 兩個預測頻率的正規化 phase 最大值，再對 sample flux 平均；查詢點先 clip 至資料範圍，再做雙線性插值。`auto_tune` 先以包含目前候選值的 2D 粗網格找種子，再以 Nelder–Mead 和越界懲罰精修。空 sample flux 會 raise `ValueError`；局部 objective 評估失敗使用有限懲罰。最大值 loss 可能無法辨識 `g`：單一亮帶足以提高分數，即使 `bare_rf` 收斂，`g` 仍可能停在 bound。GUI 負責呈現、接受與匯出候選。
