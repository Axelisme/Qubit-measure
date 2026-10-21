# FTF YAML values

**Last updated:** 2026-10-22

**讀者：** framework-dev。FTF YAML schema、單位與欄位診斷契約。

`load_ftf(path)` 一次讀取 UTF-8 YAML，`parse_ftf(document)` 驗證記憶體中的文件。兩者回傳 immutable `FTFSettings`。本模組擁有 schema、單位與欄位路徑診斷，不依賴 `simulate`、`device` 或 `program`，也不建立裝置或決定物理適用性。

使用者填表與 entry 接線請讀 [FTF YAML v2 欄位指南](../../../../zcu_lab/simulation_envs/FTF/field-guide.md)。欄位型別與省略語意的程式契約在 [schema.py](schema.py)。[環境 README](../../../../zcu_lab/simulation_envs/FTF/README.md) 提供工作副本與 GUI／Notebook 入口。

文件使用 `format: zcu.ftf`、整數 `version: 2`。`chip` 描述物理值，`dc_bias` 合併具名電源與校準，`drive_lines` 綁定 charge 或小訊號 flux drive lines，`readout` 分開 drive、物理 resonators 與 ADC，`numerics` 集中截斷與 solver。所有迴路 map 必填 `f1`、`coupler`、`f2`，key 指受作用的迴路，不是額外線路。

可省略的 `design` map 以 entry qubit 名稱保存完整的獨立標稱值，不從 `chip` 或 `dc_bias` 補值。每筆必填正值 `ej_ghz/ec_ghz/el_ghz`、A/V `flux_unit`、有限的 `flux_half/flux_bias` 與有限非零 signed `flux_period`。缺欄位在載入時報 `design.<qubit>.<field>`。省略或空 map 表示沒有設計初猜。Startup 負責依 entry-template 驗證 qubit 名稱與 device 單位，再交 predictor 檔案 owner 保存；schema 不建立 predictor 或讀 entry。

`numerics.min_round_duration_s` 是每個成功 binding round 的有限非負最短時間，單位秒。省略為 0.2，設 0 關閉補等，供 profile／benchmark 使用。此值由 `experiment.ftf` 接到 binding，不影響數值算法或 GUI 欄位。Builder 另提供 `ftf(..., min_round_duration_s=0)` 公開覆寫，無須改寫 YAML。

`numerics.telemetry.enabled` 省略為 `true`，記錄每次 acquire 的路由彙總。`numerics.telemetry.directory` 省略或 `null` 時，由環境選擇 project root 下的 `.sim_telemetry/`。明確相對路徑在環境建立時解析。公開方法 `settings.with_telemetry(enabled=False)` 回傳關閉記錄的設定副本，供 `ftf(settings.with_telemetry(enabled=False))` 使用。`directory=Path(...)` 可指定另一個目錄，省略則保留原設定。此方法只建立記憶體中的設定副本，原設定保持不變。欄位、50 MiB 輪替與 warning 語意見 [FTF runtime](../../experiment/ftf/README.md#routing-telemetry)。

錯誤包含欄位路徑。未知欄位、重複 source 名稱或 generator 綁定、非 A/V 的電源單位、無效 limits／dimensions，以及非 positive-semidefinite 的 ADC noise PSD 都在建立資源前拒絕。Loader 要求 v2 文件，對 v1 明確報錯並保留本機工作副本與既有 entry。

`readout.adc.time_of_flight_ns` 是有限非負總讀取線延遲，單位 ns。`readout.resonators.r1.mismatch_phase_rad` 與 `r2.mismatch_phase_rad` 分別是有限散射相位，單位 radians。三個欄位省略都為 0，不接受 null 或非有限值。Resources 只驗證這些值；`experiment.ftf` 接到 `simulate.adc` 的數值模型，由 ADC 決定物理延遲、微波 IQ 與散射響應。

每條 `drive_lines.<name>` 的 `operator` 接受 `charge` 或 `flux`，省略為 `charge`，不接受 null。Charge 使用 `charge_drive_ghz_per_gain` 乘完整電荷算符 n；flux 使用 `flux_drive_phi0_per_gain` 乘固定 DC 工作點的 ∂(H/h)/∂f，f = Phi/Phi0。兩種校準都含三迴路的有限複數 real/imag。每條線只接受所選 operator 的校準欄位，不能錯配、同時提供或省略。

FTF 支援 live DC bias、靜態掃描、charge drive、小訊號線性 flux drive 與 readout。Flux drive 使用固定 DC 工作點的線性模型，適用範圍限於小振幅。`flux_lines` 即使為空 map 也是未知欄位。未綁定 generator 沿用既有處理。共享 `simulate` 引擎的 flux-pulse 路徑維持停用。

`readout.drive` 綁定 pulse generator 與 incident amplitude 校準，`readout.adc` 維持單一 receiver object。
可省略的 `readout.mux` 另綁 `gen_ch`、`input_per_gain` 與 1 至 8 個 receiver 的 `adc` list。
Mux generator 不可與 pulse 或 drive lines 重複，mux 的 `ro_ch` 彼此唯一，也不可與 pulse ADC 重複。
省略或設 `mux: null` 只關閉 mux，不影響 pulse。`readout.drive.kind` 已移除，出現時載入會指出該欄位已移除。
Program 提供 tones 的頻率、gain、phase 與順序，YAML 綁定各路徑的 generator 與 receiver 校準。
兩條路徑共用兩個物理 resonator，不新增 tone-to-qubit mapping。
`readout.receivers` 依 pulse、mux 設定順序回傳全部 receiver，不決定 program 輸出順序。

Generator 數量是 drive/readout 最大 generator index 加一，ADC 數量是所有 `ro_ch` 最大值加一。索引必須是非負整數，不設固定上限。索引空洞不是已綁定的線路。Charge 係數與 resonator coupling 乘 full charge operator，不隱藏 n01 正規化。ADC noise 採 two-sided white PSD，sample covariance 為 PSD/dt_ns，receiver gain 不縮放 noise。這些設定與固定 QICK profile 的對照見欄位指南。

[default.yaml](../../../../zcu_lab/simulation_envs/FTF/default.yaml) 的 circuit energies、couplings 與 resonator frequencies 採 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。
