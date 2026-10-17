# FTF YAML values

**Last updated:** 2026-10-17，v2 欄位指南

`load_ftf(path)` 一次讀取 UTF-8 YAML，`parse_ftf(document)` 驗證記憶體中的文件。兩者回傳 immutable `FTFSettings`。本模組擁有 schema、單位與欄位路徑診斷，不依賴 `simulate`、`device` 或 `program`，也不建立裝置或決定物理適用性。

使用者填表與 entry 接線請讀 [FTF YAML v2 欄位指南](../../../../zcu_lab/simulation_envs/FTF/field-guide.md)。欄位型別與省略語意的程式契約在 [schema.py](schema.py)。[環境 README](../../../../zcu_lab/simulation_envs/FTF/README.md) 提供工作副本與 GUI／Notebook 入口。

文件使用 `format: zcu.ftf`、整數 `version: 2`。`chip` 描述物理值，`dc_bias` 合併具名電源與校準，`drive_lines` 綁定 charge microwave lines，`readout` 分開 drive、物理 resonators 與 ADC，`numerics` 集中截斷與 solver。所有迴路 map 必填 `f1`、`coupler`、`f2`，key 指受作用的迴路，不是額外線路。

錯誤包含欄位路徑。未知欄位、重複 source 名稱或 generator 綁定、非 A/V 的電源單位、無效 limits／dimensions，以及非 positive-semidefinite 的 ADC noise PSD 都在建立資源前拒絕。V1 不相容，沒有轉換或 aliases；loader 不改寫本機工作副本與既有 entry。

FTF 支援 live DC bias、靜態掃描、charge drive 與 readout，不建構 flux lines 或 flux controls。`flux_lines` 即使為空 map 也是未知欄位。未綁定 generator 沿用既有處理，沒有額外 flux-pulse 專用判定或報錯。共享 `simulate` 引擎仍保留 flux-pulse 能力，FTF 不使用。

Generator 數量是 charge/readout 最大 generator index 加一，ADC 數量是 `ro_ch` 加一。索引空洞不是已綁定的線路。微波係數與 resonator coupling 乘 full charge operator，不隱藏 n01 正規化。ADC noise 採 two-sided white PSD，sample covariance 為 PSD/dt_ns，receiver gain 不縮放 noise。這些設定與固定 QICK profile 的對照見欄位指南。

[default.yaml](../../../../zcu_lab/simulation_envs/FTF/default.yaml) 的 circuit energies、couplings 與 resonator frequencies 採 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。
