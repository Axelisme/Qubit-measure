# FTF YAML values

**Last updated:** 2026-10-19，round 最短時間、mux readout 與 charge/flux drive lines

`load_ftf(path)` 一次讀取 UTF-8 YAML，`parse_ftf(document)` 驗證記憶體中的文件。兩者回傳 immutable `FTFSettings`。本模組擁有 schema、單位與欄位路徑診斷，不依賴 `simulate`、`device` 或 `program`，也不建立裝置或決定物理適用性。

使用者填表與 entry 接線請讀 [FTF YAML v2 欄位指南](../../../../zcu_lab/simulation_envs/FTF/field-guide.md)。欄位型別與省略語意的程式契約在 [schema.py](schema.py)。[環境 README](../../../../zcu_lab/simulation_envs/FTF/README.md) 提供工作副本與 GUI／Notebook 入口。

文件使用 `format: zcu.ftf`、整數 `version: 2`。`chip` 描述物理值，`dc_bias` 合併具名電源與校準，`drive_lines` 綁定 charge 或小訊號 flux drive lines，`readout` 分開 drive、物理 resonators 與 ADC，`numerics` 集中截斷與 solver。所有迴路 map 必填 `f1`、`coupler`、`f2`，key 指受作用的迴路，不是額外線路。

`numerics.min_round_duration_s` 是每個成功 binding round 的有限非負最短時間，單位秒。省略為 0.2，設 0 關閉補等，供 profile／benchmark 使用。此值由 `experiment.ftf` 接到 binding，不影響數值算法或 GUI 欄位。Builder 另提供 `ftf(..., min_round_duration_s=0)` 公開覆寫，無須改寫 YAML。

錯誤包含欄位路徑。未知欄位、重複 source 名稱或 generator 綁定、非 A/V 的電源單位、無效 limits／dimensions，以及非 positive-semidefinite 的 ADC noise PSD 都在建立資源前拒絕。V1 不相容，沒有轉換或 aliases；loader 不改寫本機工作副本與既有 entry。

`readout.adc.time_of_flight_ns` 是有限非負總讀取線延遲，單位 ns。`readout.resonators.r1.mismatch_phase_rad` 與 `r2.mismatch_phase_rad` 分別是有限散射相位，單位 radians。三個欄位省略都為 0，不接受 null 或非有限值。Resources 只驗證這些值；`experiment.ftf` 接到 `simulate.adc` 的數值模型，由 ADC 決定物理延遲、微波 IQ 與散射響應。

每條 `drive_lines.<name>` 的 `operator` 接受 `charge` 或 `flux`，省略為 `charge`，不接受 null。Charge 使用 `charge_drive_ghz_per_gain` 乘完整電荷算符 n；flux 使用 `flux_drive_phi0_per_gain` 乘固定 DC 工作點的 ∂(H/h)/∂f，f = Phi/Phi0。兩種校準都含三迴路的有限複數 real/imag。每條線只接受所選 operator 的校準欄位，不能錯配、同時提供或省略。

FTF 支援 live DC bias、靜態掃描、charge drive、小訊號線性 flux drive 與 readout。Flux drive 不逐時重算 H(Phi(t))，不提供大振幅 fast-flux controls。`flux_lines` 即使為空 map 也是未知欄位。未綁定 generator 沿用既有處理。共享 `simulate` 引擎的 flux-pulse 路徑不因新增 operator 而啟用。

`readout.drive.kind` 省略或填 `pulse` 時，`readout.adc` 是既有的一個 receiver object。
填 `mux` 時，`adc` 必須是 1 至 8 個 receiver 的 list，`ro_ch` 不可重複。
這包含只有一個 tone 的 mux，不依 receiver 數量推斷 compile profile。
Program 提供 tones 的頻率、gain、phase 與順序，YAML 只綁定 receiver 校準。
所有 tone 共用同一 drive 與兩個物理 resonator，不新增 tone-to-qubit mapping。
預設 pulse kind 不加入序列化結果，既有 pulse settings 與 replay identity 保持不變。

Generator 數量是 drive/readout 最大 generator index 加一，ADC 數量是所有 `ro_ch` 最大值加一。索引空洞不是已綁定的線路。Charge 係數與 resonator coupling 乘 full charge operator，不隱藏 n01 正規化。ADC noise 採 two-sided white PSD，sample covariance 為 PSD/dt_ns，receiver gain 不縮放 noise。這些設定與固定 QICK profile 的對照見欄位指南。

[default.yaml](../../../../zcu_lab/simulation_envs/FTF/default.yaml) 的 circuit energies、couplings 與 resonator frequencies 採 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。
