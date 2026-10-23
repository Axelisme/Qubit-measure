# Conditional Ramsey 與 idle ZZ

**讀者：** operator、agent。從 GUI 或 MCP 量 conditional Ramsey，判讀 idle ZZ 擬合頻差與其限制。

**Last updated:** 2026-10-21

Readout 有 `$mux.module.readout` slot 時引用 library 的完整 tones 與 scalar 設定。缺 slot 時從 blank mux 範本開始，須手填全部必要讀出欄位。

`two_qubit/idle_zz` 比較 spectator 的 I、Xπ 兩個名目分支。Target 可選 F1 或 F2，每個 shot 做 target π/2 → delay → 單一 phase projection → 一次完整 mux readout。所有 shots 直接保留，兩分支是在熱初態上施加不同 pulse 的 ensemble。

## 開啟前確認

先依 [FTF bring-up 指南](../guide.md) 完成 F1/F2 的 Xπ、Xπ/2 與 mux 校準。Aliases `f1`、`f2` 指向不同 fluxonium，`coupler` 指向本對的 transmon，`mux` 指向含有 F1/F2 的 MuxGroup。Group 的有序 `qubits` 必須與 `resonators` 逐項對應。

Open／Reset 固定捕捉這些角色與 library。F1/F2 tone 取 component 在 `qubits` 的位置，畫面固定顯示角色與 tone，供操作員核對。完整 tone 列表保留原順序。缺 pulse 時表單保持 Invalid，缺 readout slot 時可手填與實測條件相符的有效 mux 設定。交換 aliases 後重新 Reset，檢查 pulses、generator 與 tone 對應。

## 設定與 Run

1. 選 target。另一個 fluxonium 是 spectator，兩分支分別施加 I 或 Xπ。每個角色的 Xπ 與 Xπ/2 必須在同一 generator，F1/F2 使用不同 generator。
2. 人工填 detune，單位 MHz。此欄必須手填。其絕對值須明顯大於預期 abs(ζ) 加 qubit 頻率誤差，使兩分支都留在 detune 選定的同側。Detune 與同側前提由操作員確認，單 projection 的資料只能在此前提下解讀。
3. 設 delay 的 µs 範圍及點數。分析至少需要 6 個不同、等距的實際 delay。時間窗要涵蓋數個 fringe 週期，採樣要足以解析振盪。硬體量化後相鄰點碰撞會報錯，須調整採樣間距。
4. 只掃 delay 時，Coupler DC source 留空，各 source 維持原值，結果的單點 DC 座標 0 表示該軸停用。要掃描時，填直接 device 名稱、該 driver 的 working unit，以及已授權的 source values 範圍。支援 A、V 或 native，直接使用所填單位。
5. 核對 reps、rounds、等待時間與完整 mux 積分窗口，再 Run。每個 DC 點依序取得 I、Xπ 的 delay scans。Live 圖顯示每個 delay 已取得的 shots 總數，FIT 再顯示 Ramsey 曲線。

DC 操作設定具名 source，也可能影響其他 loop。工作點、F1/F2 bias 與偏移補償由操作員設定並記錄，結束後 source 維持最後設定。取消或失敗後，先確認裝置當下設定再繼續。

## FIT 與符號

第二個 target π/2 使用第一個 pulse 的 phase 加上 `360 * detune * actual_delay`，phase 單位 degree。分析逐分支取 target tone 的平均 complex IQ，旋轉至 principal real axis，沿用 free-phase decay cosine fitter。

Fitter 給非負 fringe magnitude `m`，單位 MHz。若兩分支確實留在 detune 選定的同側，分析採 `f = sign(detune) * m`，再算 `ζ = f_Xπ - f_I`。負 detune 會讓 f 為負，因此 ζ 必須由 signed frequency 計算。共同的人工 phase ramp 在頻差中抵消，f 僅表示帶此 phase ramp 的 fringe 頻率。

每個 DC 點有兩分支 Ramsey／fit 圖，另有 ζ 圖。Summary 同時列出 magnitude、signed frequency、frequency 1σ error、sign resolution 與原因。ζ 誤差以兩個獨立分支的 frequency covariance errors 做平方和開根號。

**ζ 是兩個名目分支在單頻模型假設下的擬合頻差。** 熱 population 可讓每條曲線混合多個 conditional 頻率。即使 fit 有限，熱混合與單頻模型的偏差仍可能存在。檢查曲線、殘差與 `fit_quality` 的 normalized residual RMS、R²、relative parameter errors。模型是否適用由操作員看圖判斷，統計診斷、contrast 與 fidelity 都留作人工判讀。

### 符號辨識限制

分析在 magnitude 座標比較 m 距離 0 與 `2 * abs(detune)`。正負 detune 都使用這兩個非負邊界。

「近」表示距任一邊界不超過 `max(1/span, frequency_1σ_error)`。span 是該完整分支實際有效 delay 的最大值減最小值，單位 µs，所以 1/span 是 MHz。另一項來自同次 decay cosine fit 的 frequency covariance。這個判準衡量有限時間窗解析度與擬合不確定度。

近邊界可能發生零頻折疊，該支 signed frequency 與 ζ 都回不可辨識，保留可得的 fit／magnitude 與原因。Detune 為 0、有效點數不足、span 非正、振盪未解析、非有限不確定度或數值 fit 失敗也回報 ζ 不可辨識，保留原因。先保存圖與原因，再人工調整量測範圍或校準。

## 取消、保存與重分析

Raw IQ shape 是 `DC × 2 branches × delay × (rounds*reps) × full tones`。Branch 0 是 spectator I，1 是 Xπ。Shot index 依 host round 再依 program reps 排列。每 shot 保留完整 tones、讀出的 normalized ADC units/sample 與原始同 shot 對應。

取消保留已完成 rounds，未取得的位置維持 complex NaN。部分完成的曲線仍可見，只有必要 delay／round 完整的分支才 fit，兩分支都可用才算該 DC 點的 ζ。非有限 acquired IQ 回報執行錯誤。

Save 保留 raw、全部座標與本次 cfg，包括 detune、角色、pulses、完整 readout、DC device 與單位。Native Load 保持目前表單原值。Analyze 只用所載來源 cfg。Cfg 不可用時保留有效 raw，但拒絕這項分析並回報原因。

本實驗只提供診斷資料，pulse、detune 與 DC 校準由操作員決定。實際 FTF 控制鏈與物理 fidelity 須用現場量測另行驗證。
