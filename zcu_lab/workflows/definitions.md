# `zcu_lab.workflows` 使用者 workflow

**讀者：** lab-author。查閱 coherence、MIST 與 overnight workflow 的校準、分析和保存契約。

**Last updated:** 2026-10-25，MIST preparation 與依賴時態

本 package 擁有具體 workflow 的 cfg 組裝、校準、分析、區域 history 與 point record。
每個 workflow 是一個 async 函式。Engine 執行 await operations，記錄 tunables revision，並保存作者明確提交的 record。`registry.build_registry` 回傳 workflow 與表單的註冊清單，scripts 組合根將它交給 lib 宿主；import 不啟動量測。
GUI／MCP 只支援 `coherence_fluxdep`、`mist_fluxdep`、`overnight`。不提供舊 standalone 名稱的 alias。

## Coherence

Readout 優化預設 `reps=100`、`rounds=1`，其餘 workflow 量測與共用校準預設 `reps=1000`、`rounds=10`。Early-stop SNR 留空（None）關閉，不改變每輪的 reps。Spectroscopy 門檻預設 50，LenRabi／AmpRabi 預設 30，Coherence lifetime、overnight 與 overnight_mux 的 T1／Echo 預設 20.0。Readout 優化與 MIST 沒有此欄位，分別預設完整執行 1 輪與 10 輪。Singleshot shots 不受此預設影響。

`CoherencePlan` 固定 flux setpoints、明填的 flux_unit、resolved drive/readout modules、delay points 與 historical RunSnapshot。
`measure_t1`、`measure_t2ramsey`、`measure_t2echo` 預設 true，至少開啟一項。全關在 acquisition 前拒絕。
每點先設 flux，再依 frequency、Rabi、readout、可選 singleshot GE、T1、Ramsey、Echo 順序執行。
關閉的 lifetime 不量、不保存，對應的 point 欄位保留 None。

`measure_singleshot` 是固定 plan bool，預設 false，不取代至少開啟一項 lifetime 的要求。
`CoherenceTunables` 分為共用 `cal` 與獨立 `singleshot`、`t1`、`t2ramsey`、`t2echo`。
GE 只要求本點 Rabi 已接受，不再加 Rabi residual 或對比門檻。熱布居會壓低 Rabi 對比，額外門檻會偏向排除熱激發大的點。
`singleshot.min_spacing` 用 plan 的 flux 單位，從上一個實際取得 GE 的點計算絕對距離；距離至少等於它才量。
Schedule failure 不更新間距基準，Completed partial 仍更新。預設間距 0，允許每個合格點及重複 setpoint。
每種 preparation 預設 100000 shots，是暫定模擬值。
GE 使用本點 fitted pi pulse 與接受或保留的合法 readout，relax delay 使用 T1 的 relax_factor × 該點 history-based guess，即使 T1 關閉也沿用此公式。
每個 Completed GE 保存 `singleshot-ge.h5`，partial 也先保存。Workflow 不呼叫 GE analyze，不做分類、擬合或 workflow 結果圖。Core 的 operation 即時圖保留。
主模型有擬合結果時，每個 lifetime 都記兩種模型。T1 記含 baseline 的單指數與雙指數，T2 記指數與零中心高斯包絡；有條紋時兩者都帶 cosine。
`t1.models`、`t2ramsey.models`、`t2echo.models` 記各模型的 lifetime、誤差、residual RMS、quality、BIC 與狀態，圖上疊加兩條可用曲線。
只有原有主模型參與接受、relax 與歷史。BIC 不選接受模型，副模型不改下游值。
非有限、非正 lifetime、誤差大於值、負誤差、quality.invalid 或 lifetime 超過 `max_model_window_factor * max(fitted delays)` 時，模型診斷記 `discarded` 與原因。倍數預設 2，三種 lifetime 各自可調。
有限但 discarded 的值仍保留，非有限值存 null。副模型失敗只記 failed/reason；主模型失敗不嘗試副模型，partial 不擬合。
三份 lifetime history 只收各自原有主模型的接受值，delay grid 保留 alpha=0.5 的 EWMA 與各自初始 guess。
T1 的 relax delay 沿它的 delay guess。T2 的 auto relax 優先用本點接受的主模型 T1，其次 T1 EWMA，無歷史用 t1.guess_us，再套 relax_factor 與 relax_min_us。Fixed relax 使用明填值。
Rabi／readout 的 relax delay 是最近 accepted T1 的五倍；尚無 accepted T1 或 T1 關閉時，用合法 T1 guess。
例如 T1 關閉且 guess=50 µs，兩個校準 stage 等待 250 µs，不補量 T1。

Ramsey 的 detune_ratio 是有限 fringes/step 值，無論 plan 開關都驗證，預設 0.05。
Requested detune 為 ratio/length_step；公開 acquisition result 保留原生量化後的 actual detune。
Echo 的 detune_ratio 預設 0.1，零選純 decay，非零選 fringe；requested detune 同樣是 ratio/length_step。
兩者仍以原有 fringe／whole-trace decay 作主模型，拒絕超過兩倍最大 delay 或 mean absolute error 超過 0.1 倍 fit peak-to-peak 的值。
T1 保留原有 relative error 與 normalized residual RMS 門檻。拒絕 reason 先保留 lifetime_rejection 的原因，再逐項列出失敗 gate 的名稱、實際值、失敗比較與門檻；無法取得的值標為 unavailable。
Pi2 使用所選 Rabi 模式的 fitted pulse，不取 pi duration 或 gain 除以二。Len 模式使用 phase-aware fitted duration 與相同 gain。Amp 模式使用相同固定長度與獨立 fitted pi/2 gain。
T1 的 native cfg 由 `coherence_fluxdep.stages.t1` 建立，T2 的 cfg 科學 helper 暫留 `coherence_fluxdep.t2`。純資料 plan/point carrier 不代表 standalone workflow 入口。

## 獨立校準與 point

共用 calibration 的 `rabi_mode` 預設 `len_then_amp`，也可選 `amp_only` 或 `len_only`。

| 模式 | 掃描與下游 pulse |
| --- | --- |
| `len_then_amp` | 先接受 LenRabi，再固定其 π 長度掃 gain，範圍 0 到 min\(1, 2.5 × 本次 Len gain\)。下游 π 與 π/2 用該長度及 Amp 各自 fitted gain。Amp 未接受時，本點 Rabi 不接受。 |
| `amp_only` | 不跑 LenRabi。固定 `target_pi_length_us`，未填時取 min(host seed π, rabi_min_stop_us / rabi_stop_factor)。首次掃 0 到 `amp_gain_max`，之後以上次 accepted Amp π gain 為中心掃 ±`amp_gain_half_width`，限於 0 到上限。 |
| `len_only` | 保留 Length-Rabi 的 gain 回饋、長度掃描及 fitted π/π2 duration。 |

Amp 共用 `max_rabi_rel_err`、`max_fit_residual` 與 `rabi_max_mean_residual_ratio`。π gain 至少為 `amp_min_gain`，且不超過 `amp_max_window_fraction × 實際掃描上限`。
預設最小 gain 為 0.0001，窗比例 0.9，amp-only 上限 1，追蹤半寬 0.2。
π/2 gain 必須有限且至少為最小 gain，固定長度也須至少 0.03 µs，否則只接受 π，不產生新的 π/π2 pair。
`RabiResult.amp` 保留 gain fit、誤差、固定長度與 `amp-rabi.h5` 引用，拒絕 fit 也保留診斷。
Combined 模式的 `file` 仍指 `rabi.h5`，`len_status` 記 Len 的接受狀態。Amp-only 的 `file` 指 `amp-rabi.h5`。
Amp 的 partial raw 先保存，不擬合、不更新 carry。下游只接受本點產生的 π；需要 π/2 時也必須來自本點。
`Calibrated.drive_gain` 與 `pi_length_us` 一起定義 π。`CalibratedPi2.pi2_drive_gain` 與 `pi2_length_us` 一起定義 π/2，gain 為 None 時沿用 π gain。

`calibrate_frequency`、`calibrate_rabi`、`calibrate_readout` 可分別用 await 呼叫。
`CalibrationPulses` 固定 Rabi/pi drive `qub`、獨立 spectroscopy `probe` 與 pulse-backed readout；各 stage 另收自己的 settings、本點 guess 與 `StageEnv`（plots、snapshot、summary）。
Coherence 與 MIST 的 plan 分別保存兩個 drive。Frequency cfg 複製 `probe_pulse` 的 waveform/長度，頻率仍由 tracker 掃描設定，gain 從 probe 的 gain 起跑，再依線寬自適應。
新表單先引用 captured `$qubit.pulses.qub_probe`，沒有則複製 captured x180，設 freq 0 MHz、gain 0.1、長度 5 µs；缺 x180 時保留必填。
Rabi 與下游仍複製 `qub_pulse`，不受 probe 長度或 gain 影響。Frequency result 的 `probe_gains` 保留每次 attempt 實際用值。
Helper 收 Scope 與 `StageEnv`；frequency 與 Rabi 另明收各自的回饋控制器（tracker、rabi_drive），不收整個 `CalCarry`。
三個 stage 各回 frozen 具名結果，保留 status、reason、accepted scalar 與自己的 immutable native 檔名。
Frequency 接受後，calibrate_point 立即更新 frequency history；Rabi 只在本點 frequency 接受後執行，接受時同樣由 calibrate_point 更新 pi/pi2。
Readout 用本點 accepted π 與其配對頻率，不要求 π/2。兩個 caller 共用 `readout_due(point_index, tun)`，依零基 flux index % ro_every == 0 掃描。
Index 0 自然先掃。到期但本點 Rabi 未接受就 skipped，不順延到下一點。失敗／拒絕不增加掃描，也不改下一個 due index。
Readout 非接受時保留合法 seed／prior accepted readout，仍可量 lifetime 或 MIST。
Rabi 的頻率與所有下游 π pulse 都要求 NOW，只接受本點產出，不借前點 pulse。
T1、GE、readout 與 excited MIST 要求本點 accepted π；Ramsey／Echo 另要求本點可信 π/π2 pair。Ground MIST 不加 π，無須本點 Rabi accepted。
缺少本點 pulse 時，該 consumer 記 skipped 並寫明依賴。本點 π/2 不可信時，不阻擋只需 π 的 consumer。
Point 的 calibration_sources 記 consumer、need=now、source_index、source_flux 與 fallback=False。readout_source 另記沿用的 readout 校準來源，source_index=None 代表固定 host seed。

`CoherencePoint` 保存 frequency、rabi、readout、singleshot，以及獨立的 t1、t2ramsey、t2echo 欄位。T1Result、RamseyResult 與 EchoResult 各自是獨立型別。
各 lifetime 與 GE 的 cfg builder 截斷 DAC gain，將 before/after 保存在自己的 result.gain_clamped。CoherencePoint 不另存 point-level gain_clamped map。Ramsey／Echo 的 gates 記錄原有接受比較，reason 列出未通過的項目。
`singleshot` 保留 measured／skipped／failed、reason 與相對 native 檔名。未開、Rabi 未接受、間距不足或取消皆記 skipped 原因。
完整擷取記 measured，Schedule failure 記 failed 且沒有檔名，partial 記 failed／partial 並保留檔名。未走到此 stage 的點記 skipped／not reached，歷史 record 缺欄位時為 None。
三個 lifetime 欄位未量測或未到達時是 None，包含關閉的項目。開啟但失敗、拒絕或 skipped 時保存具名結果與原因。這是 2026-10-25 的格式切換；沒有真實舊資料，不支援舊 measurements 格式讀取。
Frequency 預設 `frequency_mode="two_step"`。粗掃使用 probe pulse gain，
span 為 `span_mhz * coarse_span_factor`，預設倍率 3。可用 `coarse_gain` 覆寫粗掃 gain。
粗掃先選 predictor 加 IDW 中心最近的候選，其餘依 prominence 降序。
每點最多試 `max_candidates` 個候選，預設 3。重用同次粗掃資料隔離各峰做局部擬合。
局部擬合失敗或 truncated 會換下一候選。沒有可用粗峰時才沿 `qf_widen` 擴窗，仍受候選總數限制。
每個可用粗峰以其擬合中心細掃一次，不再套 detune offset。
細掃 span 預設為 `span_mhz`，可用 `fine_span_mhz` 覆寫。
細掃 gain 預設為自適應 gain 的一半，且不超過粗掃 gain 的一半，可用 `fine_gain` 覆寫。
只有細掃結果進入頻率接受門檻、IDW、可信歷史與線寬回饋。
細掃沒有候選，或最強候選中心離該粗峰的擬合中心超過該粗峰的半 prominence 寬度，會拒絕該粗峰。
細掃擬合失敗或未通過既有品質 gate 也換下一候選。不增加每次掃描點數或 rounds。
細掃失敗不退回粗掃，候選都失敗才觸發整點的 recovery 判斷。
`frequency_mode="single"` 保留原有整窗擬合、gain 與 widening。

`peak_min_prominence` 是投影訊號 ptp 比例，預設 0.2。
`peak_min_distance_mhz` 是候選最小間距，預設 0 MHz。
`peak_min_snr` 預設 5，prominence 與峰高相對 median 都須至少為此倍數的雜訊 σ。
σ 由一階差分 MAD 換算。候選的 `snr` 是峰高相對 median 除以 σ，不是 prominence/σ。
`peak_min_width_samples` 預設 3，是半 prominence 的窗內可觀察取樣間隔數。
降低 SNR 或寬度門檻會同時增加弱峰與雜訊候選。這些統計門檻不保證排除所有雜訊。
每點 Frequency result 的 `scans` 保存 coarse、fine 或 single 階段、
檔名、gain、span、實際中心、全部候選、選定候選及擬合中心。
每次 scan 的 `diagnostics` 記 fitted_center_mhz、center_error_mhz、fwhm_mhz、peak_snr、trace_snr、normalized_rms 與 mae_over_fit_ptp。
Peak SNR 取選定候選相對 median 的高度 SNR。Trace SNR 用完整投影 trace 的 Savitzky-Golay estimator；它是診斷值，不增加接受門檻。
`gates` 記名稱、comparison、threshold、actual、passed 與 applied。非有限或未取得的 actual 為 null，該比較不通過；尚未執行的 fit gate 不虛造結果。
選定候選記 SNR 與 truncation gate。Coarse 另記局部 fit interval 上下界，最終接受 gates 只供診斷，標記 applied=false。
Fine 與 single 記正中心、窗內中心上下界、正 FWHM、非負 center error、error/FWHM、normalized RMS 與 MAE/fit ptp 的既有門檻。Fine 另記最強候選是否存在及其中心到粗峰的距離上限。
每個試過的局部粗峰診斷保存在 `coarse_candidate_diagnostics`，以 selected_peak_center_mhz 對應候選。Failed／partial acquisition 不擬合，保留 fit_failure，無法取得的數值為 null。這些紀錄不參與接受判斷。
`candidate_attempts` 逐一保存試過的粗峰、粗掃檔名、局部擬合中心、細掃檔名與 rejection_reason。
拒絕原因包含沒有細峰、超出粗峰寬度、截斷、擬合失敗與品質 gate；只有接受的候選沒有拒絕原因。
候選的 `truncated` 表示窗內無法觀察完整半 prominence 寬度。
兩步檔名含 coarse 或 fine，single 維持 frequency-NN.h5。
Frequency 保存每個 Completed attempt 的引用，fit 拒絕也保留 native raw。
Schedule Failed／skipped stage 沒有偽造檔名。分析失敗仍保留先保存的 Completed 檔名。工具按 point 的用途欄位讀檔，不掃目錄或猜 effect 次序。
這讓 frequency 接受而 Rabi 失敗的點仍保留部分科學證據。

Carry 與 lifetime history 是 async 函式的區域變數。每點先建可變 point_record，欄位尚未取得時為 None。
Coherence 與 MIST 共用的 calibrate_point（`common/point_calibration.py`）是複合 stage，依序呼叫 frequency、Rabi 與 readout；Rabi 依模式包含 AmpRabi。它正常回傳 PointCalibrationResult，主迴圈再填 frequency/rabi/readout；`CalCarry` 的歷史欄位（last、f01_by_flux、traces、last_amp_gain、readout_source）由 calibrate_point 依各 stage 回傳的結果更新。Expected failed/rejected 仍回傳各 stage 的結果。
校準的 unexpected exception 不回傳 aggregate。例如 frequency 成功後 Rabi 內部 bug，finally 保存的 point 不含 frequency 結果，但 frequency 已存的 h5 仍在磁碟，run 失敗。
主迴圈用 try/finally，在 finally 同步 iteration.commit(point_record) 一次，正常、continue 與例外都經過同一提交位置。
Pause 中斷目前 operation，Resume 重試同一 request；已完成的校準與區域變數不重做。
子層取消讓 operation 回 partial。作者先保存 Completed raw，再檢查 partial_result，保留原有拒絕原因，不進擬合或接受。
量測間若 iteration.cancelled，補 status="skipped" 再 continue，只跳過目前點。Helper 以 return 退出該點。
根 Stop 在 async operation 或 Ask raise ScriptCancelled。Finally 提交已有成果，不改 status，例外繼續向上傳播。
分析 helper 對非有限 measured IQ 回 Failed(reason)，不填值或 mask。Native fitter 的具名 AnalysisFailed 與訊息恰為 `array must not contain infs or NaNs` 的 scipy ValueError 也轉 Failed。Caller 記 status=failed 並保留 raw 檔名。
Shape、grid、detune、cfg 與其他程式錯誤仍 raise。其他 warning 沿原政策處理。投影與接受門檻不變。兩步掃描只讓細掃結果更新 history，single 保留原整窗擬合。
沒有 stage checkpoint、原始檔 reader、檔案追蹤或 rollback 保證。Unexpected adapter、fit、writer 或 plot 例外停止 run，不改標為正常失敗。

## MIST 與 overnight

`mist_fluxdep` 使用相同獨立校準 stage。Plan 固定 reset、drive、MIST disturbance pulse 與 gain grid。
measure_mode 預設 excited，可選 ground 或 both。Ground 省略 pi preparation，不要求 Rabi 成功；both 在同一 flux 點先量 excited 再量 ground。
表單從固定 capture 的 $qubit.t1 取微秒 T1 放入 plan.t1_us。有值時 relax=5*T1，無值時用 tun.relax_factor*tun.guess_us，不另量 T1。Point 的 relax_source 與 relax_delay_us 記來源與實際等待時間。
`MISTPoint.traces` 分別保留每種 preparation 的 success、reason 與 gain_file。頂層 success 要求全部 requested modes 完成，gain_file 指向第一種 preparation。
Raw IQ 保存為 mist-excited.h5、mist-ground.h5。Partial 也先保存，再記 success=False、reason=stopped，不更新 map 成功列或完整 gain 圖。
單列投影為 centred magnitude/(std+1e-12)，常數 trace 得零，不推 threshold。Map 再扣全域中位數、取絕對值並截斷於三倍標準差，保留 raw IQ。
Map 使用 native 實際 gain 與有單位的 flux sample centres，不用 requested extent 或 flux index 替代。
Export 的 --mode excited/ground 選要匯出的 preparation；預設選 ground-only plan 的 ground，其餘選 excited。

`overnight` 維持固定 flux、有限 T1→Hahn-echo pair 與固定開始間隔。
T1 relax 為 `max(relax_min_us, relax_factor*t1_guess_us)`。
Echo relax 為 `max(relax_min_us, relax_factor*T1估計)`，優先使用同一輪剛接受的 T1，未接受時用 `t1_guess_us`，不沿用前輪 fit。
`relax_min_us` 預設 1 µs，允許 0。兩個 delay grid 仍由 `sweep_factor` 乘各自 guess 決定，fit 不改 pulse、readout 或後續輪次的 guess。
Point 的 `echo_relax_delay_us` 記 Echo cfg 實際使用的等待時間，`echo_t1_source` 記 `accepted_t1` 或 `guess`。尚未建立 Echo cfg 時兩欄為 None。
Tunables 的 `uniform` 預設 true，控制 T1 的線性 delay sweep。設 false 時，native T1 在同一 start/stop 視窗內把 delay 集中在衰減段，Echo 不變。
它不呼校準，不讀 context，也不因 Coherence 合併增加 calibration。
每個 Completed member 先保存 raw，partial 不擬合並記原有原因。預期 fit failure 記 reason 且 lifetime/error 為 None。
目前 iteration 未取消時才開始另一 member。取消的 pair 留 status="skipped" 與已取得欄位，再進下一點。
下一 pair 在 max(now, previous actual start + interval) 開始，不做 catch-up bursts。
Pause 保留 pair 的區域進度，Resume 繼續等待中的 operation；Stop 保留既有 commits。
區域變數 `trend_points` 保留具名 scalar pair tuple，不保存 Figure 或 axes。
`overnight.trend.draw_overnight_trend` 直接畫這些 pair，缺值維持 gaps，不重 fit。
每個 pair 開始與完整 pair 提交前更新 `overnight_trend`。圖可能先於 journal commit 顯示，不是提交證據。
只有一個 pair 時，以 actual_start 為中心取 ±plan.start_interval_s，UTC 刻度顯示 HH:MM:SS。多 pair 保留實際時間與自動範圍。
宿主不輪詢 committed records 來呼叫它們。Workflow 透過 `env.plots` 決定資料與更新時機，framework 不保證只畫 journal 已提交的紀錄。
沒有自動保存或從 run 目錄重開趨勢圖。保存責任見 ADR-0062。

`overnight_mux` 保留 overnight 的有限 pair、開始間隔、不追趕排程、取消與 finally commit。
每輪先用 `MuxT1Exp` 同時量所有 participants，再用 `MuxT2EchoExp` 同時量 Echo。它不設定 flux，也不校準。
Echo 的共用 relax 取同一輪 participants 已接受 T1 的最大值乘 `relax_factor`，沒有接受值時用 `t1_guess_us`。兩個 member 套用相同的 `relax_min_us` 下限，來源與實際等待時間保存在 point 的同名欄位。
Plan 保存完整 mux readout 順序、participants 原始索引、每個 tone 的已校準 x180 與獨立 x90、可省略的共用 reset，以及歷史 snapshot。
Mux tunables 與 overnight 的欄位及意義相同。`uniform` 預設 true，只控制 T1 的取樣分布，所有 participants 共用同一條 delay grid。
設 false 時使用 native mux T1 的非均勻 cycle table，Echo 仍為線性。兩個 guess 請分別填最長的 T1 與 Echo。
Native mux 分析逐 tone 擬合，再各自套 overnight 的 relative-error 與 normalized-residual 門檻。
Overnight 與 mux 沿用 Coherence 的雙模型診斷，分別保存在 point 與每個 qubit 的 `t1_models/t2_models`。
T1 的副模型為雙指數，零 detune Echo 的副模型為零中心高斯；圖上兩條曲線都標示。`max_model_window_factor` 只控制診斷 discarded，不改接受值、Echo relax 或趨勢圖。
每輪保存兩個 native 檔名、排程與實際 UTC 開始時間、status，以及每個 participant 的名稱、T1/Echo 接受值、誤差與原因。
`overnight_mux_trend` 的 T1/Echo panels 各為每個 qubit 畫一條線，缺值留 gap。單輪範圍、更新時機與不自動保存的契約同 overnight。

## Run 摘要

四個 workflow 正常結束時 return `common.summary.RunSummary`。框架只在 done 時把它寫到 metadata root 的 `summary.json`，不增加 point commit。
檔案使用框架 encoded record 信封，摘要在 `value`。讀取可用 `saved_run.read_summary(metadata_root)`；`workflow_records`／RPC 不回傳這份檔案。
Stop、未處理的 acquisition／analysis／writer 錯誤都會中止 return，因此沒有 `summary.json`。Finally 的部分 point commits 仍保留，不是完整 run 摘要。

摘要只從本次 run 的 point payload 與 delivered Completed facts 累計，不讀磁碟、raw 檔或重新擬合。
`planned_points` 是 flux 點數或 outer_rounds。`committed_points` 包括所有提交點，`completed_points` 只計 status=completed，`skipped_points` 計 status=skipped。
Stage 的拒絕或失敗不會扣除 completed_points。
`flux_range` 記已提交 point 的 flux setpoints 最小、最大值與 plan 單位；空 run 或不設 flux 的 mux 省略此欄。
`physical_recoveries` 計 frequency.recovery 非 None 的 point，包含被拒絕的 physical refit 嘗試。

`stages` 按 frequency、Rabi、readout 與啟用的 singleshot／T1／Ramsey／Echo／MIST 彙整 accepted、rejected、skipped、failed。
Amp Rabi 若執行則獨立列出。未走到的啟用 stage 算 skipped／not reached，關閉的 stage 不列入。
Coherence lifetime 與 calibration 沿用原有 status。完整 singleshot 的 measured 對應 accepted，partial 仍按 point 的 failed 計數。
Overnight 與 mux member 的 status 直接在結果產生處填寫：接受值為 accepted、完整 fit 未過門檻或 partial 為 rejected、Schedule／預期 fit failure 為 failed、未執行為 skipped。
MIST trace 完整 success 為 accepted、partial 為 rejected、Schedule failure 為 failed、依賴不足或未執行為 skipped。
MIST both 每點計兩個 preparation 結果；mux 每點按 participant 計 T1／Echo。副模型診斷不計為另一個 stage，也不改主模型的狀態。
各 stage 的 rejected_reasons、skipped_reasons、failed_reasons 按原始 reason 字串計數，不分類或改寫原因。

`duration.mean_s/max_s` 先加總同一 point、同一 stage 的 operations 秒數，再算平均與最大值。只有有時間的 point 參與平均。
Frequency 包含該點所有 delivered coarse／fine／widening scans，MIST 包含兩種 preparation。Mux 計一次共用 acquisition 時間，不乘 participant 數。
只計 delivered Completed（含 partial），不包含 Failed 或 Pause discarded attempts；缺少時間就省略 duration。
`early_stops` 計 acquisition facts 的 early_stopped=True 次數，`acquire_retries` 計 attempts>1 次數，不是額外 acquire 次數的總和。
這兩項也只涵蓋 delivered Completed facts。完整 Failed／Pause attempts 留在 point 信封與 journal，不在此摘要重建。

`qubits` 只在 overnight／overnight_mux 列出，各 qubit 的 t1／echo 記 accepted、mean_us、std_us。
平均與母體標準差只取原有主模型的接受值，不混入 rejected／failed 或副模型值。單筆標準差為 0，零筆省略平均與標準差。
單 qubit overnight 的 name 取 snapshot.roles 的單一 qubit 字串，缺失或非單一名稱時為 null，不從 pulse channel 猜。Mux 使用 plan participant 的原有名稱。

## Native、裝置與前端邊界

作者用 zip(plan.fluxes, env.iterations()) 取得每點的 Scope，第一次用 iteration.dir 才建 absolute 保存目錄。
目錄是 data root 下的 iteration-000001，nested child 再按 label 與 sequence 鏡射。Commit 序號另按提交順序增加。
Workflow 在 iteration.run 返回 Completed 後呼既有 exact-path native writer，fit 拒絕仍保存 raw。
Point 引用是相對該目錄的 filename，不保證檔案完整或仍存在。Pause／Stop 可能留下未提交檔案。
Writer 收 `Completed[RunRecord]`，從 value 取得 cfg/result，從 facts 取得 acquisition 時間與 device snapshot。
Caller 顯式傳入 `env.identity` 與歷史 `RunSnapshot`。Saver 只組裝既有 `RunMetadata`，不捕捉新事實。
H5 run_id 是 `<identity.run_id>-<facts.iteration:06d>-<facts.attempt>`。Completed.partial_result 決定 complete 或 stopped，experiment 取 persistence spec.tag。
Provenance 取 workflow identity，不逐 operation 捕捉。Lab 不建立 AcquisitionEvidence。
Framework 只保存 manifest、journal 與 best-effort record.json。
Conversion 的科學型別與輸入驗證留在此 package，不在框架建立 reader 或通用 dispatcher。

## 獨立 run 轉換

`coherence_fluxdep.export.export_run` 支援 Labber 與 samples，`mist_fluxdep.export.export_run` 只支援 Labber。
函式收到 `database` 就寫 Labber，Coherence 收到 `result` 就追加 samples。兩個目的地都給就都寫，未給任何目的地就 raise ValueError。
四個 workflow 的 export 模組都提供同 process 的 `main(argv)`，也可用 `python -m zcu_lab.workflows.<name>.export` 執行。
Overnight 與 overnight_mux 只接受 `--result`，重新匯出已提交各輪的 samples，不寫 Labber。
傳入 run 的 metadata 目錄。CLI 給 `--database <Database>` 就寫 Labber，Coherence 給 `--result <entry result>` 就追加 samples。
不另選格式旗標。沒有任何目的地是 CLI error。
工具在任何寫入前驗共同輸入。Labber 每次在日期樹取新檔。
Merged 輸出依使用者 2026-10-25 裁定移除，見 [ADR-0062](../../docs/adr/0062-experiment-workflow.md#獨立-workflow-轉換工具)。
各 writer 回報自己的 path/status/error。未配置的 Labber path 為 None，已配置的失敗保留 path，不自動重試或寫 ledger。

工具讀固定 manifest，再用 `saved_run.read_records(run, CoherencePoint)` 或 MISTPoint 讀整個 run 的 typed commit list。
每列的 directory 來自 commit 信封，工具只讀 point 中具名的 native reference。Coherence 從 t1、t2ramsey、t2echo 欄位取各量測的原生檔名；None 不產生該量測的 channel。Generic saved_run 不解讀 lab 欄位，也不轉換舊 measurements 格式。
沒有 record.json 的配置目錄不產生 row。Reader 不從 raw 檔推造遺失 record，export receipt 的 missing_records 保留為空 tuple。
壞資料指名 source，不讀 journal、掃 native 候選或重抓目前 GUI。
Coherence 各已量 lifetime 分別輸出 Signal/Time Flux×Index 與 Flux-only lifetime，保留各自 actual delay。
Disabled 項目沒有 channel，rejected fit 保留 raw 與 NaN lifetime，不 union、插值或重 fit。
MIST 使用保存的 actual gain 作為 inner-first Signal 軸，另寫 Flux-only Success。
所有已取得 trace 的 actual gain grid 必須完全相同，否則拒絕並列出兩點的 index 與值。
Plan gain grid 是請求，不拿它與 QICK 量化後的實際值比較。

Coherence 在每個 `iteration.commit(point_record)` 成功後立即 append 一列 sample。Overnight 每輪一列，mux 每輪每個 participant 一列。
目的地是固定 entry 的 `Result/records/components/<qubit>.jsonl`，與 GUI RecordWriteback 相同。
三個 plan 的 `result_path` 由表單從固定 `FormEnv.parameters.capture_storage()` 捕捉，manifest 保存絕對路徑。
None 表示沒有 entry，例如 headless 腳本，不自動寫入；手動 `--result` 可補寫，也可指定搬移後的新 entry。
拒絕、跳過與中止的已提交點仍產生 row，保留 stage status／reason。未到達的 stage 記 skipped 及原因。
主頻率、Rabi、readout 與 lifetime 只填 accepted 值，非接受值為 null。副模型保留各自 status 與完整診斷，不宣稱接受。
Coherence 使用 `frequency`、`rabi`、`readout`、`t1`、`t2ramsey`、`t2echo`、`singleshot` 等巢狀欄位。
Frequency 含 freq_mhz／err_mhz／fwhm_mhz／probe_gain。Rabi 含 mode、π／π2 的 us 與 gain，以及 AmpRabi 診斷。
Lifetime 含 us／err_us／detune_mhz／models，保留雙指數第二個時間常數。GE 記相對 raw 檔名。
`flux` 記 commanded device／value／unit。`devices` 原樣保存最後一次已保存 acquisition 的完整 snapshot。
順序為 frequency、LenRabi、AmpRabi、readout、GE、T1、Ramsey、Echo，包含 rejected 與 partial raw。沒有 raw 時為 null，不另讀儀器。
Overnight 類記 T1／Echo 的 status／reason／accepted 值／models、UTC 排程及開始時間、raw 檔名、Echo relax 來源。
固定 frequency、π／π2 與 readout 保存在 `settings`，不冒充本輪校準。Mux 不設定 flux，`flux` 為 null，座標由讀取端從完整 devices 選擇。
Coherence key 是 `<run_id>/<index>`，overnight 類是 `<run_id>/round/<round_index>`，mux 每個 component 各用同一輪的 key。
自動與手動使用同一個 row producer。手動在任何 append 前準備所有 rows。I/O failure 保留已寫 prefix/count/cause，不 rollback。
Commit 與 append 不是原子交易；append 失敗會讓 run 停止，先前 commit 與 rows 保留，可手動補寫。
Latest-wins 由 record table owner 決定，重匯出更新整列，僅 append 時間 `at` 隨每次寫入改變。GUI、RPC、MCP 沒有 conversion controls。
用法見 [每點 sample 與重新匯出](README.md#每點-sample-與重新匯出)。

`common.devices.set_flux` 讀 live typed cfg，重用 check_flux_unit，再以 model_copy 只改 value，await 完整 setup。
Fake 明填 native；YOKO 明填 A 或 V。Plan／表單不推測、換算單位或預驗 connected membership。
每次 env.run 在 callback 前 capture connected drivers。Native saver 使用該次 Completed.facts，不因 Resume、alias 修改或 Save 刷新 snapshot。

核心只透過當次 RunContext.plots 更新 native typed handles；buffer callback 只送座標與複數資料。
每次 env.run attempt 使用同 host 的獨立 operation Plots，完成後 finish。新 attempt release 並替換舊 operation，workflow 趨勢圖留在 env.plots。
Coherence 與 MIST 的 workflow 校準圖為 `calibration/qubit_freq`、`calibration/len_rabi`、`calibration/readout`。
Frequency 圖上方畫 accepted f01 對 flux，下方畫 flux × detune heatmap 與目前 trace/fit。Detune 相對各點掃描中心。
Rabi 圖畫 flux × pulse-length heatmap，紅線連各 flux 點的 accepted pi duration，側欄畫目前 trace/fit。
Readout 圖以 imshow 畫目前 freq × gain SNR，紅叉標有限 SNR 最大點。Cadence 未到期時保留上一張。
Carry 只保留本次 run 的掃描列與 accepted scalars，不保留 Figure 或 axes。Frequency widening 取代目前點的列，各列保留實際座標。
完整但 rejected 的 fit 仍顯示，分析失敗時顯示 raw I。Partial 不更新 heatmap 或 fit。
三張圖在完成的校準 point 目錄內保存為 `qubit-freq.png`、`len-rabi.png`、`readout.png`，cadence skip 不另存 readout 圖。

Coherence 的 `t1`、`t2ramsey`、`t2echo` 各一張，取代分開的 decay 與 vs_flux 圖。
上 panel 從 accepted history 畫 lifetime 對 flux 的 scatter 與 error bar，下 panel 從 Completed.value.result 的 times/signals 重畫目前 decay 與同座標的 fit。
Rejected fit 標示 rejected，不增加上方 accepted 點。只建立 enabled lifetime 的圖。
每個 point 提交後更新 history panel，合併圖保存為 `<kind>-decay.png`，不另存 summary PNG。
MIST 圖為 `mist/<mode>/gain` 與 `mist/<mode>/vs_flux`。gain-<mode>.png 從 Result 的實際 gains/signals 重畫，map-<mode>.png 從本次 run 保留的 traces 重畫。
所有 PNG 都不取 operation live 圖。Native raw／export 格式不變，overnight 圖不變。
GUI forms、measure headless owner、固定 capture、U7 與 S4 分工見 [forms](forms.md)。
行為比對使用 measure GUI 的 workflow tabs 與離線來源，不另用 agent 專用模擬腳本。模擬仍須使用隔離的 project 與保存目的地，不連接硬體。

Framework 測試使用合成 workflow：`tests/experiment/workflows/` 驗通用 Engine，`tests/gui/app/measure/services/`、`tests/gui/app/measure/remote/` 與 `tests/mcp/measure/` 驗宿主、RPC 與 MCP 接縫。不測本套件的科學流程或預設值，刪掉 `zcu_lab/` 後整套 pytest 仍須通過。
