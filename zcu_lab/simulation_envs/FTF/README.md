# 模擬環境

**讀者：** operator、lab-author、agent。準備 FTF 模擬參數與 entry，並從 GUI 或 Notebook 啟動模擬環境。

**Last updated:** 2026-11-18，每次 acquire 選維度

既有 entry 把 resonator 與 mux group 的 `ext.time_of_fly` 搬到頂層 `time_of_fly`。程式不自動搬移。

這裡放 FTF 模擬參數、entry 樣板與可信 GUI startup 範例。
Startup 的 `PROFILE_PATH` 指向共用 [qick_216_v2.yaml](../profiles/qick_216_v2.yaml)，用 `program.build_soccfg` 建立板子後以 `soccfg=`
交給 FTF。板子使用 v6 gen0–6 與 mixmux8 gen7，dyn readout 0、1 與 mux generator 共用 wave port，以 `tmux_ch` 區分。PFB readout 2–9 共用 ADC、使用 decimation 2，所有 readout 使用 tport trigger。Channel 編號是模擬環境的接線，不是實機 channel 清單。
板子的時脈與型別不放入物理模型 YAML。Startup 與 Notebook 直接呼叫公開 `zcu_tools.simulators.ftf.environment.ftf(...)`，不另設 environment 薄包裝。Startup 讀取相鄰 `ftf.yaml`；Notebook 明示 YAML 路徑。指定檔案不存在或內容有誤時直接報錯，不改讀另一份設定。

`default.yaml` 是 FTF 的標準 preset。`ftf.yaml` 是本機工作副本，不納入 Git。第一次使用時複製一份：

```bash
cp zcu_lab/simulation_envs/FTF/default.yaml zcu_lab/simulation_envs/FTF/ftf.yaml
```

切換到共用 v2 板子時，先保存既有工作副本，再重新複製 `default.yaml`，或手動把 `ftf.yaml` 的 `readout.mux.gen_ch` 改為 7，兩筆 `readout.mux.adc[].ro_ch` 依序改為 2、3。保留自己的物理參數與校準。
既有 `ftf-demo` 不會套用新 template。請手動把 MG12 wiring 改為 `gen_ch: 7`、`ro_ch: [2, 3]`，並核對已保存的 mux library 模組與 tab 輸入。也可先備份，再刪除專案 `result/ftf-demo` 與 `Database/ftf-demo`，重跑 startup 重建 entry。MG12 的 `mixer_freq: 7135` 不變。程式不自動遷移工作副本或 entry，kickstart 範本也不隨這次切板更新。

缺少 `ftf.yaml` 時，`ftf(...)` 直接報錯。完整欄位、單位、合法範圍、三迴路映射與 entry 接線對照見 [FTF YAML v2 欄位指南](field-guide.md)。

`default.yaml` 的背景使用 68／44／308／52.6 MHz 乘性駐波，深度依序為
0.088／0.066／0.044／0.043，搭配 2% 複數 RMS roughness、
−4 dB/GHz 斜率與 3.7 GHz 低頻轉角。既有 `ftf.yaml` 不會自動更新。
要看到新背景，請自行把樣板的 `background.extra_ripples`、`roughness` 與 `band_edges`
補入工作副本的 pulse 及 mux receivers，並同步調整原有 ripple 與 slope，再重啟環境。
`roughness.seed` 固定線路形狀，與量測雜訊的 `numerics.seed` 無關。
欄位定義見 [ADC 背景契約](../../../lib/zcu_tools/simulate/adc/README.md)。

## 建立 entry 與啟動

在 repo root 準備好上面的 `ftf.yaml`，再用 startup script 啟動。不需先執行 entry CLI：

```bash
mkdir -p /absolute/new-ftf-project
uv run --directory . --no-sync -- python -m apps.measure \
  --project-root /absolute/new-ftf-project \
  --script zcu_lab/simulation_envs/FTF/startup.py
```

`--project-root` 須指向既存的絕對目錄。省略時使用 repo root。
Script 在該專案建立 `ftf-demo`。它從 [entry-template.yaml](entry-template.yaml) 加入元件，並從 [entry-library.yaml](entry-library.yaml) 安裝脈衝和讀出模組到 entry 根目錄的 library。接著建立並選取 `initial`，從 global 繼承元件與模組，並初始化 point aliases。
MG12 依序綁定 Q1／R1 與 Q2／R2。
`ftf-demo` 已存在時，script 不套用 template，
直接開啟該 entry 並選取 `initial`，保留之前的修改；`initial` 不存在就報錯。
`result/` 與 `Database/` 只剩其中一個 `ftf-demo` 時是不完整的 entry，script 報錯並指出缺哪一邊，
不重用也不覆寫；兩邊都刪除即重新建立。
首次建立中途失敗可能留下部分 entry，沒有跨檔交易或自動 rollback。

Script 接著建立離線 FTF 環境，安裝 SoC，並登記 `f1_bias`、`coupler_bias`、`f2_bias` 和 `coil`。標準 preset 的 DC 初值將 Q1／Q2 帶到 half-flux 工作點。GUI 可直接從 New Tab 開實驗，不必手動開 entry 或選 point。
Script 透過共用 GUI 策略選擇 `workers=max(1, (os.cpu_count() or 1) - 1)`，保留一核。
`warmup=True` 在背景準備常用 CPU 路徑與可用的 OpenCL 程式，不執行 acquisition。

MCP 的等效入口是
`connect(launch="new", script="/absolute/path/startup.py", project_root="/absolute/new-ftf-project")`。
Attach 不重跑腳本；在 attach 傳 script 會報錯。自訂 script 取代預設腳本，
不先執行預設 startup。`--clean` 跳過所有腳本，不能與 `--script` 同用。

可將 `startup.py`、`entry-template.yaml`、`entry-library.yaml` 與工作副本 `ftf.yaml` 複製到自己的目錄，再指定該處的 startup。Entry 範本與模型設定按 startup 檔案位置定位，不跟隨 shell cwd。板子另由 `PROFILE_PATH` 定位；複製 startup 到其他目錄後，須把它改成共用板子 YAML 的實際路徑。
要改 entry、point 或 tab 名稱，編輯自己的 startup 副本。

Demo point 的 qubit 頻率與 `kappa` 提供頻譜掃描的中心與頻窗。`wiring.lo_flux_ch` 選擇各 qubit 的序列內磁通線。Q1 的 `ext.flux_reset_gain` 和 `ext.flux_reset_length` 是 flux reset 表單的起點。
MG12 的 `flux_dev` 選擇兩顆 qubit 共用的 DC source `coil`。Q1／Q2 的 `flux.period.coil` 使用 A／Φ0，供 global flux 表單換算掃描範圍。
既有 v1／v2 entry 須先用[長期升級入口](../../../docs/parameter-upgrades.md)轉換。Startup 不自動升級。[Kickstart 範本](../../kickstart/FTF/entry-template.yaml) 用於從接線開始的走查。若修改工作副本的物理值或接線，也要核對 entry 範本及已建立的 entry。

Startup 腳本建好初始 alias。`mux` 指向 MG12，mux 表單透過 `$mux` 讀取同序的 qubits 與 resonators。
MG12 的 `mixer_freq: 7135` 是 mux LO 初值，缺值時 mixer 留空必填，不由 tone 頻率或 κ 推算。
`MG12.time_of_fly` 用於 mux 路徑，R1／R2 頂層的 `time_of_fly` 用於各自的 pulse ADC。Kickstart 的 group 與 resonator TOF 保留 null。
`mux/onetone` 缺 slot 時，trigger 初值取 group TOF 加 `5/(2π·min κ)`，length 取 trigger 加固定 1.0 us ADC window。Group TOF 或任一 κ 缺值時，trigger 與 length 留空必填，mixer 不受影響。有 slot 時沿用 slot mixer 與既有 timing 規則。
R1／R2 的 wiring 保留 pulse gen0／ADC0。MG12 的 mux wiring 填 `gen_ch: 7`、`ro_ch: [2, 3]`，依序對應 R1／R2，不讀 resonator 的 pulse ADC。
範本的 `MG12.module.readout` 指向共用 mux 模組。缺 readout slot 時，八張 mux 表單的 Open／Defaults 從已知的 group wiring 補通道，tone gain 補 0.1。文件允許 `wiring: {}`；`ro_ch` 未知時，各 tone ADC 留空必填，其餘缺失的必填輸入仍保持 Invalid。有 slot 時沿既有 library 路徑，細節與核對步驟見[接線對照](field-guide.md#generatoradc-與-entry-接線對照)。
初始 alias `qubit` 指向 Q1，`resonator` 讀取其 resonator。把 `qubit` 改為 Q2 後，`resonator` 跟著切換。重新 Open／Defaults 後，scalar flux 表單從 `$qubit.flux_dev` 取得 Q2 的 DC 裝置名稱，不需要 flux 裝置 alias。`qubit1`、`qubit2` 分別指向 Q1、Q2，`coupler` 是普通 alias，指向 qubit/transmon 元件，不是 Notebook role。

## 設計初猜與 predictor

環境作者可在 `ftf.yaml` 的 `design:` 填寫獨立標稱值。這是設計圖的頻率初猜，不是
`chip` 或 DC 校準真值，也不是量測結果。樣板為 Q1、Q2 各填一筆，數值刻意與物理模型不同。
每筆必填 `ej_ghz`、`ec_ghz`、`el_ghz`、`flux_unit`、`flux_half`、
`flux_period`、`flux_bias`。欄位意義與單位見[欄位指南](field-guide.md#design設計初猜)。

Startup 以 entry-template 的 qubit 名稱驗證 design，並檢查 `flux_unit` 與其
`flux_dev` 的 device 單位一致。未知 qubit 或單位不符會報錯。
Startup 經 resources 的 `save_predictor_file` 建立 entry 的 `predictors/Q1.json`、
`predictors/Q2.json`，只寫缺失的檔案。既有 predictor 與手動調整保留不變。
能量交給 predictor v2 的 `params`。Startup 以 `offset = 0.5 + (flux_bias - flux_half) / flux_period`
換算設計對齊，寫入 `flux.offset`，不做 mod 1。`flux.period` 與 `flux.unit` 以 template 的實際 `flux_dev`
為鍵，保留 signed period 與原生單位。它不讀目前 component 校準，也不從物理模型補值或填入 entry 頻率。

省略 `design` 或填 `design: {}` 時，環境照常啟動，不建立 predictor。
狀態列會顯示「design 未提供，未建立 predictor」。既有檔案不會刪除。
提供的每筆若缺欄位，`ftf()` 載入時會指出欄位路徑，例如 `design.Q1.flux_period`。

MCP 先呼叫 `predictor.list_files` 發現檔案，再使用
`predictor.predict(filename="Q1.json", device_values={"f1_bias": 1.2e-3})` 取得標稱頻率。
樣板的 device 原生單位是 A，這個輸入表示 1.2 mA。初猜仍須由量測驗證。

## 工作副本與調參

樣板更新不會自動改動既有的 `ftf.yaml`。可從 [default.yaml](default.yaml) 採用新的接收雜訊、磁通線與 `numerics.projection_weight_floor`，再重建模擬環境。

各 receiver 的 `noise_psd_adc2_ns` 是每個 I/Q 分量的雙邊白雜訊 PSD。I 和 Q 的雜訊獨立，變異數相同。標準 preset 可用於練習低 SNR 讀出。
保留既有 `ftf.yaml` 設定時，逐筆把舊 2×2 PSD 改成一個數字。兩個對角值相同時，保留原對角值；兩者不同時，指定要給 I 和 Q 共用的 PSD。Loader 拒絕舊矩陣並指出欄位。
變更讀出或雜訊後，以 `singleshot/ge` 重新確認所用通道的 SNR。舊矩陣的校準結果不適用於等向設定。

`default.yaml` 使用 `format: zcu.ftf`、整數 `version: 2`。V1 文件不能載入，不自動轉換既有工作副本，也不自動更新既有 entry。三個當地 DC 電源與共用 coil 都在 `dc_bias.sources`，樣板使用 A。環境每 round 讀取 live DC snapshot，保留靜態磁通掃描。`drive_lines` 的每條線以 `operator: charge | flux` 選電荷或固定 DC 小訊號磁通導數，省略為 charge。Flux operator 不是逐時重算 Hamiltonian 的大振幅 fast-flux 控制。

序列內的 flux pulse 另以 `flux_lines` 宣告，不放在 `drive_lines`。`f1`、`coupler`、`f2` 各可選填一條 single-fluxonium 格式的 `FluxLine`，填入 `name`、`gen_ch`、`flux_phi0_per_gain`，再依需求設定 ns 單位的延遲與濾波。Flux 線的 generator 不可與另一 flux loop、任一 drive line、pulse readout 或可選 mux readout 重複。載入失敗時，錯誤會列出兩端綁定的欄位路徑與 channel。省略或填 `{}` 保留原本設定，各迴路省略或填 `null` 停用該線。`default.yaml` 啟用 F1 和 F2 的 flux 線，entry 範本的 Q1／Q2 接線與它們對應。欄位與單位見 [Flux lines](field-guide.md#flux-lines序列內磁通校準與濾波)。

FTF 的 `DriveLine.operator: flux` 仍只在 DC 附近線性化。Fast-flux 線的校準與該小訊號算符分開，數值接線與限制見 [FTF 契約](../../../lib/zcu_tools/simulators/ftf/environment/README.md)。未綁定 generator 沿用既有處理。採用新樣板時請保留自己的工作副本，逐欄核對，不覆寫尚未保存的本機調整。

無微波的連續 flux motion 走 `adiabatic_map`，即使 bath 含 rates 也保持 populations，只衰減 coherence。只有 ramp 遠短於 T1 才能忽略這項近似，接近 T1 的 ramp 會低估弛豫。T1 reset 須有真正 constant park row 走 `t4`，不把 filter tail 視為 constant hold。物理 adiabatic 條件與 native／local／retained-level 收斂另需驗證。DAC 取樣、flux 量化、其他路由與 DC readout settling 限制見 [Flux lines 指南](field-guide.md#flux-lines序列內磁通校準與濾波)。

啟動腳本會執行 Python，只載入可信來源。修改腳本或 YAML 後須重啟 GUI。腳本失敗時，GUI 保留已完成操作並顯示腳本位置與錯誤，不自動改跑預設腳本。

`default.yaml` 的 circuit energies、couplings 與 resonator frequencies 採論文 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。Channel 數量、型別與時脈由共用的 [qick_216_v2.yaml](../profiles/qick_216_v2.yaml) 提供，不從模型接線推導。Caller 要確保物理模型 YAML 的綁定符合該板子。樣板使用 pulse readout gen0／ADC0、mixmux8 mux readout gen7／PFB readout 2 與 3，以及 f1_charge gen2、f2_charge gen3。共用板子 YAML 提供 8 個 generator 與 10 個 readout，FTF 只綁定其中所需的通道。ADC output sample interval `dt_ns` 由 caller 的板子 YAML 決定，模型 YAML 不設定 ADC clock。F1 與 F2 的快磁通線分別使用 gen1 與 gen4。兩條 charge line 的主迴路係數為 0.1 GHz/gain，coupler 串擾為 0.01，對另一 fluxonium 為 0。這些係數是模擬假設。Entry 範本的 Q1/Q2 分別綁 gen2/gen3，既有 entry 不會自動更新。完整訊號路徑與接線表見 [欄位指南](field-guide.md)。更換樣品時可複製 startup 與 YAML，並調整 Notebook 的 YAML 路徑。

[Notebook 範例](../../../notebook_md/single_qubit.md) 呼叫同一個公開 `ftf(...)` builder，並用 `closing_environment` 關閉引擎與裝置。Startup 在交付前登記 `on_close`。成功登記的 driver 由 GUI 關閉，環境 manager 在 runtime 只借用它們。Callback 先關閉引擎，再移除這些借用並關閉尚未交付的 driver。引擎清理失敗時仍會清理裝置。Startup 中途失敗仍保留 callback。完整組裝與數值限制見 [FTF 契約](../../../lib/zcu_tools/simulators/ftf/environment/README.md)。

### 整數 fluxonium 的三態參考驗收

[Qutrit reference populations](../../v2/singleshot/qutrit_population/guide.md) 的短參考案例使用一套特定模擬接線，不是 `default.yaml` 的通用預設。Pump、Stokes、readout 分別使用 gen1、gen2、gen0；3–2 走 charge drive。名目 readout 為 7460 MHz、0.4 µs const pulse、gain 0.1、0.4 µs ADC 積分、trigger offset 0 µs。轉移頻率仍須由光譜量測取得，不由表單從模擬器回填。FTF 每個 repetition 從設定的熱態開始，這不驗證硬體 reset。

這個短參考案例的驗收指標是平均 preparation-label error ≤ 3%、每個 label error ≤ 6%、training confusion 2-norm condition number ≤ 1.5、最小奇異值 ≥ 0.8。用另一固定 seed 獨立量測驗證選定的讀出設定。早期的 separation 門檻用不同距離量，不能當作 shared-σ 分數的門檻。這些數字不適用於強 STIRAP／Raman drive，也不是實機校準標準。

## 啟用跨 acquisition 漂移

`default.yaml` 的頂層 `drift` 範例預設註解，省略或 `drift: null` 都關閉漂移。
要啟用時，在工作副本 `ftf.yaml` 取消該區塊註解，再重啟環境。
先確認所選的 `chip` 損耗都大於 0，不能對 `q_cap: null` 或值為 0 的 `xqp`／`aphi` 啟用漂移。
F1／F2 可選 `q_cap`、`xqp`、`aphi`；coupler 只有 `q_cap`、`aphi`，`q_ind` 不漂移。
`sigma` 是 ln(loss) 的穩態標準差，`tau_s` 是正秒數。省略某個迴路或損耗只讓該項保持常數。
欄位限制與單位見 [Drift 指南](field-guide.md#drift跨-acquisition-損耗漂移)。這些設定不寫入 entry，kickstart 範本也不變。

三個迴路各自使用獨立 log-OU stream。第一次 acquisition 使用 YAML chip 初值，
後續每次 begin 按 simulated time 更新，整次 acquisition 的 points 與 rounds 共用同一組有效值。
環境級 `SimulationClock` 使用 monotonic wall seconds 加不遞減 offset。
每個成功 round 補足未經 `time_scale` 縮放的 physical duration 與 wall 耗時之間的正差額。
所以 `time_scale=0` 仍推進 simulated physical time，兩次 acquisition 間的閒置時間也計入。
重啟環境會建立新 session，不接續先前的漂移；同一 seed 只有在相同時間序列下才有相同漂移。

啟用漂移後，每次 begin 在 `~/.local/state/zcu-tools/simulation/ftf/<session>/truth.jsonl`
附加三行，`qubit` 分別為 `f1`、`coupler`、`f2`。每行記錄 session、simulated seconds、UTC wall time
與當次損耗，coupler 的 `q_ind`／`xqp` 是 JSON `null`。
關閉漂移不建立 truth log。寫檔失敗會讓 acquisition 報錯，不套用 routing telemetry 的 warning 政策。

`FTFAcquisitionOptions` 從 `zcu_tools.simulators.ftf.environment` 匯入，以 `acquisition=` 傳給 `ftf` 或 `FTFSource`。
`clock` 可注入環境專用時鐘，`truth_log` 可指定另一個 `Path`，相對路徑以 cwd 解析。
`truth_log=None` 使用上面的預設位置。
檔案含模型真值，不進一般 Run／GUI／MCP metadata。呼叫者負責路徑與檔案保留期限。
完整參數見 [acquisition 契約](../../../lib/zcu_tools/simulators/ftf/environment/README.md#acquisition-clock-and-loss-drift)。

## 演化與除錯

`retained_levels` 是耦合系統的維度上限。FTF 在符合條件的 acquire 用一條 pilot timeline 決定演化維度，讀出仍使用上限基底。
少於 8 條 timeline、上限不超過 12 態、需要 projective herald 或含早期 cavity drive 時，整次 acquire 直接用上限。
見 [pilot 規則](../../../lib/zcu_tools/simulators/ftf/environment/README.md#dimension-selection)與 [acquire 摘要欄位](../../../lib/zcu_tools/simulators/ftf/environment/README.md#routing-telemetry)。

GUI startup 選擇 `workers=max(1, (os.cpu_count() or 1) - 1)`，保留一核，至少使用一個 worker。八核主機使用七個 GUI workers。GUI 傳 `warmup=True`，在背景準備常用 CPU 路徑，並依 `retained_levels` 預編譯可用的 OpenCL 程式。準備成功後，第一次 acquisition 不必再等這些編譯。

Notebook 傳 `workers=max(1, (os.cpu_count() or 1) - 2), warmup=True`，保留兩核，並在背景預編譯常用 CPU 路徑與可用的 GPU 程式。八核主機使用六個 Notebook workers。公開 `ftf()` 仍預設 `workers=1, warmup=False`；測試、benchmark、研究與直接呼叫不套用 GUI 策略。

同一個 process 內的環境共用相同維度的 GPU 預編譯，不重複編譯。獨立 warmup 引擎只填磁碟快取，不抽樣或改變正式 device bias。沒有可用 GPU 時維持按需執行的 CPU 路徑。

若有準備工作，`env.warmup.status` 提供進度與錯誤，`env.wait_ready()` 等待結束。若要排除預編譯，先等工作結束再開始計時。Acquire 會自動等待，準備失敗仍按需運算。GUI 關閉時等背景工作結束再釋放裝置，不強制中止 compiler。完整介面見 [FTF 契約](../../../lib/zcu_tools/simulators/ftf/environment/README.md#startup-preparation)。

整個 evolve 期間 native BLAS 限為 1 個執行緒。重疊的 evolve 共用 process 層級的 reference count，第一個進入時設限，最後一個離開時還原。其他程式碼不要在 evolve 期間自行更改 threadpoolctl 限制。沒有演化工作、只有一條 timeline 或只有 1 個 worker 時，不開 joblib pool。模型與 caller-owned 完整 preparation 只保留在記憶體，Engine 的數值 preparation 限一次 evolve。不需要離線建庫或 GPU，也不保存 exact density records。

要查一次模擬結果，在環境 YAML 設定 `numerics.debug_capture: true` 後再 acquire。
在快取的 `simulator-debug/v1` 開啟該次的 `capture-*.json.gz`，可讀 Engine 輸入、每列實際 family 與逐 round DC 值。用該 round 的 `model_identity` 查 `metadata.effective_settings` 中的有效設定。
若 telemetry 開啟，當次 `routing-*.jsonl` 的 `capture_file` 指向這份 manifest，並記下投影結果與各階段耗時。
見 [FTF Debug capture](../../../lib/zcu_tools/simulators/ftf/environment/README.md#debug-capture)。Capture 含模型真值，不放入一般 Run／GUI／MCP metadata。
