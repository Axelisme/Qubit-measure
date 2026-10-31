# Heralded conditional Ramsey 與 idle ZZ

**讀者：** operator、agent。設定 heralded idle ZZ，判讀條件式 Ramsey 與 herald 通過比例。

`two_qubit/idle_zz_heralded` 是獨立實驗。表單、Run 與保存結果不共用普通版的工作狀態。Target 使用小寫 `f1` 或 `f2`，脈衝、DC 掃描、detune 符號與 fit 限制見[普通版指南](guide.md)。

每個 shot 做可選 reset → herald mux readout → spectator I/Xπ → target π/2 → delay → phase projection → terminal mux readout。Reset 留空時不發射 reset。兩次讀出使用同一個 mux 物件與相同 circle classifiers，herald 後的等待由 readout.post_delay 決定。

## 校準與 Run

Open／Reset 固定捕捉 F1/F2 與原始 full-mux tone 位置。先完成兩角色的 GE Post，F1/F2 都必須有含 radius 的 circle classifier。其他 tone 不要求 classifier。

選 target，填 signed detune 與 delay。分析至少需要 6 個不同且等距的實際 delay。Detune 絕對值須明顯大於預期 abs(ζ) 加 qubit 頻率誤差，兩分支須留在 detune 選定的同側。可選具名 DC source 只按指定工作單位掃描，不補償也不恢復原值。需要 reset 時填入已校準的 cfg，再核對 reps、rounds 與等待時間後 Run。

Live 顯示每個 DC／branch 的 target ⟨σ⟩，只選 herald 判為 g,g 的 terminal 計數。這不是 active reset。取消後保存已完成 rounds，未量點維持 NaN。

## 保存與分析

Sums 是 float64 `DC × 2 branches × delay × 3 × 3 × 3 × 3` 原始計數。類別軸依序是 herald F1、herald F2、terminal F1、terminal F2，各為 g/e/other。全部 81 類都保存，不只保存接受 shots。Rounds 在取數端累加，不保存個別 shots 或 IQ。舊 raw-IQ 格式缺少 sums 時報錯。

分析選 herald g,g 切片，再以普通版相同的 target `⟨σ⟩ = (N_e − N_g) / N_total` 做 decay cosine fit。Other 留在分母。零接受數給 NaN，該支 fit 與 ζ 不可用。完整性由全部 81 類總數判定，不把低 herald 接受數誤判成 acquisition 未完成。

每個 DC 點另畫兩個 branch 的全部 9 類 herald 比例。g,g 是 pass fraction，分母是所有 herald classes 的總數。Pass fraction 是分類接受率，不是 ground-state purity。Classifier 誤差、有限 radius 與熱混合都限制此解讀。

Terminal 分析回傳與普通版相同的 magnitude、signed frequency、covariance error、fit_quality、ζ 與原因。外層分析另有 herald_matrix 與 pass_fraction。ζ 仍是 `f_Xπ − f_I`，不宣稱純 ZZ，也不寫回 calibration。Analyze 只讀保存的 cfg，不查目前 Form 或 library。
