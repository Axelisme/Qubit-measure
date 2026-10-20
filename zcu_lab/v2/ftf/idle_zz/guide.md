# Conditional Ramsey 與 idle ZZ

`ftf/idle_zz` 比較 spectator 的 I、Xπ 兩個名目分支。Target 可選 F1 或 F2，每個 shot 做 target π/2 → delay → 單一 phase projection → 一次完整 mux readout。不做 heralding、post-selection 或純態初始化。兩分支在熱初態上施加不同 pulse，不代表已準備出純 spectator 狀態。

## 開啟前確認

先依 [FTF bring-up 指南](../guide.md) 完成 F1/F2 的 Xπ、Xπ/2 與 mux 校準。Aliases `f1`、`f2` 指向不同 fluxonium，`coupler` 指向本對的 transmon，`mux` 指向含有 F1/F2 的 MuxGroup。Group 的有序 `qubits` 必須與 `resonators` 逐項對應。

Open／Reset 固定捕捉這些角色與 library。F1/F2 tone 取 component 在 `qubits` 的位置，畫面鎖住角色與 tone，不能手填 index。完整 tone 列表保留原順序。缺 pulse 時表單保持 Invalid，缺 readout slot 時可手填有效 mux 設定，不用 simulator truth 補校準。交換 aliases 後重新 Reset，檢查 pulses、generator 與 tone 對應。

## 設定與 Run

1. 選 target。另一個 fluxonium 是 spectator，兩分支分別施加 I 或 Xπ。每個角色的 Xπ 與 Xπ/2 必須在同一 generator，F1/F2 使用不同 generator。
2. 人工填 detune，單位 MHz。此欄沒有預設值。其絕對值須明顯大於預期 abs(ζ) 加 qubit 頻率誤差，使兩分支都留在 detune 選定的同側。程式不會自動找 detune，也無法從單 projection 證明這項前提。
3. 設 delay 的 µs 範圍及點數。分析至少需要 6 個不同、等距的實際 delay。時間窗要涵蓋數個 fringe 週期，採樣要足以解析振盪。硬體量化後相鄰點碰撞會報錯，不會合併資料。
4. 不掃 DC 時，Coupler DC source 留空。此時不讀寫任何 source，結果的單點 DC 座標 0 只是 inactive sentinel，不表示把 source 設為 0。要掃描時，填直接 device 名稱、該 driver 的 working unit，以及已授權的 source values 範圍。支援 A、V 或 native，不做單位換算。
5. 核對 reps、rounds、等待時間與完整 mux 積分窗口，再 Run。每個 DC 點依序取得 I、Xπ 的 delay scans。Live 圖顯示每個 delay 已取得的 shots 總數，FIT 再顯示 Ramsey 曲線。

DC 操作只設定具名 source，不等於只改 coupler loop flux。操作員自行提供與記錄 F1/F2 bias。程式不補償偏移、不選工作點，也不在結束時自動恢復 source。取消或失敗後，先確認裝置當下設定再繼續。

## FIT 與符號

第二個 target π/2 使用第一個 pulse 的 phase 加上 `360 * detune * actual_delay`，phase 單位 degree。分析逐分支取 target tone 的平均 complex IQ，旋轉至 principal real axis，沿用 free-phase decay cosine fitter。

Fitter 給非負 fringe magnitude `m`，單位 MHz。若兩分支確實留在 detune 選定的同側，分析採 `f = sign(detune) * m`，再算 `ζ = f_Xπ - f_I`。負 detune 會讓 f 為負，因此不能直接把兩支 magnitude 差當 ζ。共同的人工 phase ramp 在頻差中抵消，不把 f 當作 qubit 的絕對 transition frequency。

每個 DC 點有兩分支 Ramsey／fit 圖，另有 ζ 圖。Summary 同時列出 magnitude、signed frequency、frequency 1σ error、sign resolution 與原因。ζ 誤差以兩個獨立分支的 frequency covariance errors 做平方和開根號。

**ζ 是兩個名目分支在單頻模型假設下的擬合頻差。** 熱 population 可讓每條曲線混合多個 conditional 頻率。有限 fit 不代表已排除混合，更不代表得到 pure ZZ。檢查曲線、殘差與 `fit_quality` 的 normalized residual RMS、R²、relative parameter errors。模型是否適用由操作員看圖判斷，不設 χ² lack-of-fit、自動統計 gate、contrast 品質門檻或 fidelity 門檻。

### 符號不可辨識

分析在 magnitude 座標比較 m 距離 0 與 `2 * abs(detune)`。負 detune 也用這兩個非負邊界，不把正的 magnitude 直接與負的 `2 * detune` 比較。

「近」表示距任一邊界不超過 `max(1/span, frequency_1σ_error)`。span 是該完整分支實際有效 delay 的最大值減最小值，單位 µs，所以 1/span 是 MHz。另一項來自同次 decay cosine fit 的 frequency covariance。這是有限時間窗解析度與擬合不確定度的判準，不是物理 fidelity 門檻。

近邊界可能發生零頻折疊，該支 signed frequency 與 ζ 都回不可辨識，保留可得的 fit／magnitude 與原因。detune 為 0、有效點數不足、span 不為正、無可辨識振盪、非有限不確定度或數值 fit 失敗也不輸出 ζ，不以 0 或預設誤差放行。先保存圖與原因，再人工調整量測範圍或校準。

## 取消、保存與重分析

Raw IQ shape 是 `DC × 2 branches × delay × (rounds*reps) × full tones`。Branch 0 是 spectator I，1 是 Xπ。Shot index 依 host round 再依 program reps 排列。每 shot 保留完整 tones，直接使用 `MuxReadout.collect_raw_iq` 的 normalized ADC units/sample，不另解析 raw buffer 或重做 normalization。

取消保留已完成 rounds，未取得的位置維持 complex NaN。部分完成的曲線仍可見，但缺任何必要 delay／round 的分支不做 fit，也不計算該 DC 點的 ζ。非有限 acquired IQ 是執行錯誤，不是假裝取消的 sentinel。

Save 保留 raw、全部座標與本次 cfg，包括 detune、角色、pulses、完整 readout、DC device 與單位。Native Load 不回填目前表單。Analyze 只用所載來源 cfg，不讀目前 aliases、Form 或 library 補歷史缺值。Cfg 不可用時保留有效 raw，但拒絕這項分析並回報原因。

本實驗沒有 writeback，不自動改 pulse、detune 或 DC。離線合成走查只驗證控制與分析契約，不等於實際 FTF smoke 或物理 fidelity 驗收。
