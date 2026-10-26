# Components

**Last updated:** 2026-10-27，離線 flux v1→v2 與 runtime 版本分開

**讀者：** lab-author、agent。查詢 component kind、角色與字串引用。

新增 model、註冊與表單引用見 [component 教學](../tutorial/01-components.md)。Component 保存校準值、wiring 與 module 引用，不宣告 program 的執行型 cfg。Core 的 module 欄位使用 `Pulse.Cfg`、`Readout.Cfg`、`MuxReadout.Cfg` 等型別；寫法見 [實驗教學](../tutorial/03-experiment.md#2-在-corepy-定義資料與量測)。`register_all` 註冊以下六個 kind，import 不會註冊。

| Kind | 代表什麼 | 哪些實驗讀它 |
| --- | --- | --- |
| `resonator` | Readout resonator 的頻率、linewidth 與 wiring。 | `onetone/*`、`twotone/*`、`jpa/*` 與 mux tone 初值。 |
| `group/mux` | 1 至 8 個有序 resonators 與可選、等長且不重複的 qubits，擁有 mux wiring、module 槽與 global coil 校準。 | `mux/*` 與 `two_qubit/*` 的表單。 |
| `device/current_source` | Flux bias 來源的 component 記錄，只有共用 ext、module，沒有 wiring。 | 不代替 driver；flux 表單從實驗元件的 flux_source 取得裝置初值，Run 使用選定的已連線 driver。 |
| `amplifier/jpa` | JPA pump frequency、power、flux bias 校準及可空的 flux_source／rf_source 裝置名稱。 | `jpa/*` 的表單與 writeback。 |
| `qubit/fluxonium` | 共用 qubit 校準加上 EJ、EC、EL。 | `twotone/*` 與其他讀 qubit 校準的表單；mux／FTF 表單也可經 group 讀取。 |
| `qubit/transmon` | 共用 qubit 校準加上 EJmax、EC、無因次 d。 | 讀 qubit 校準的表單；FTF 可將它綁為 coupler。 |

## Component 文件版本

`register_all` 同時註冊 `parameter_upgrades.PARAMETER_SCHEMA_POLICY`，目前為 `schema_version: 1`。Setup、point 與 entry template 都使用這個整份文件版本，容器版本不變。結構變更的版本宣告、完整舊格式分類與純函式步驟由本 package 擁有。通用執行與 I/O 由 resources 擁有，見 [ADR-0063](../../docs/adr/0063-persistence-ownership.md#component-文件版本與明確升級)。

長期入口是 `scripts/upgrade_parameters.py`。`parameter_upgrade_plan()` 保留完整分類與 flux v1→v2 純步驟，未標版文件只在分類成功後視為 v1。離線 target 為 2、runtime 仍為 1，過渡期只在副本驗收，v2 產物待 runtime 升 2 才能載入。`flux_upgrade.py` 擁有四類分類、local／mux 共用轉換與 source 名稱語法檢查，不做 I/O 或 device catalog 查詢。`storage_mapping` 經中性完整文件 batch callback 重用轉換，長期步驟不依賴 migrations。四類形狀、參考點假設與手動修復見 [flux 升級指南](../../docs/flux-migration.md)，runner 操作見 [升級命令](../../docs/parameter-upgrades.md)。

## Qubit flux frame

Qubit 的 `flux` 預設為 `{}`，以 device 名稱對應 `QubitFluxCalibration`。
每筆必填有限的 `half`、`integer` 與非零 `period`，使用該 device 的原生單位。
其他 contributing devices 都停在各自 integer 時，這兩個 bias 分別是 half／integer 參考點。
`period` 可為負，表示增加 bias 會減少 flux。
載入保留三欄，`support/flux.py` 在使用時檢查 `abs(integer-half) = abs(period)/2`，失敗指出 device。
Helper 重用共用 device 名稱驗證，不改寫有效名稱。
這是新增欄位，目前仍使用 schema version 1，舊 flux 欄位與 mux `coil_flux` 暫時保留。

## Wiring

`BuiltinComponent` 只提供 `ext`、`module` 與未知欄位 migration。Wiring 由各 kind 宣告，不從基底繼承。

| Kind | Wiring 欄位 | 用途 |
| --- | --- | --- |
| `resonator` | `ch`、`ro_ch` | 單 tone generator 與 ADC。 |
| `qubit/fluxonium`、`qubit/transmon` | `ch`、`lo_flux_ch` | Drive generator 與 local flux channel，由 `QubitBaseSchema` 宣告。 |
| `group/mux` | `gen_ch`、`ro_ch` 列表 | Mux generator 與逐 tone ADC，`ro_ch[i]` 對應 `resonators[i]`，長度必須相同。 |
| `amplifier/jpa`、`device/current_source` | 無 | 表單從元件的 flux_source／rf_source 推導裝置名稱，再使用選定的已連線 driver，不讀 component channel。 |

Mux wiring 不從各 resonator 的單 tone wiring 推導。FTF 的 R1／R2 使用 pulse ADC0，MG12 則使用 mux ADC1／ADC2。

Mux generator 未知時可省略 `gen_ch` 或填 null，但 group 必須提供 wiring 與等長的 `ro_ch` 列表。未宣告的 wiring 輸入沿既有規則移到 `ext.migrated`，不會成為有效通道。設計依據見 [ADR-0072](../../docs/adr/0072-module-boundaries.md#使用者元件與模擬環境的放置)。

## 角色與引用

內建角色為 `qubit → qubit/*`、`resonator → resonator`、`qubit1 → qubit/*`、`qubit2 → qubit/*`。Notebook 省略 roles 時用有序 shorthand `qubit, resonator`，預設焦點匹配 `qubit/*`。GUI 使用獨立 alias 表，不由這些角色自動綁定，見 [角色與失敗](../../lib/zcu_tools/resources/entry/README.md#角色與失敗)。

| 字串欄位 | 填入的名稱 |
| --- | --- |
| Qubit `resonator` | Readout resonator 元件，例如 `R1`，kind 為 `resonator`。 |
| Qubit `flux_source` | 自己的 DC 線 device 名稱，例如 `Q1_bias`，不代表已連線。 |
| Mux group `flux_source` | Global coil 的 device 名稱，例如 `coil_bias`，不受 qubits 是否提供影響，不代表已連線。 |
| JPA `flux_source` | JPA bias 的 DC device 名稱，可為 null，不代表已連線。 |
| JPA `rf_source` | JPA pump 的 RF device 名稱，可為 null，不代表已連線。 |
| Resonator `amplifier` | Readout amplifier 元件，例如 `J1`，內建 kind 為 `amplifier/jpa`。 |

Qubit 的 `flux_half`／`flux_period`／`flux_int` 保存自己的 DC 線校準，單位由 `flux_unit` 指定。`wiring.lo_flux_ch` 仍是 fast-flux 通道，不是 DC device。

Mux group 的 `coil_flux: dict[str, CoilFluxCalibration]` 保存每顆 qubit 對 global coil 的校準，預設為 `{}`。Key 必須是 group `qubits` 中的名稱，不依列表位置對齊。每筆 `CoilFluxCalibration` 有 `half`、`period`、`int`，分別代表半整數磁通的 coil bias、一個磁通量子的 bias span、整數磁通的 coil bias。三欄各為有限數值或 null，預設 null，使用 coil device 的原生單位。尚未校準的 qubit 可省略整筆，部分校準可只填已知欄位。未知 qubit key 會拒絕。qubits 為 None 或空時，coil key 條件只允許 `{}`；既有 qubits／resonators 等長規則仍適用。

這些字串不在載入時預檢整張引用圖；解析時才查目標。單位見 [component 欄位單位](../README.md#component-欄位單位)，共用欄位與 ext 政策見 [component 教學](../tutorial/01-components.md#1-宣告欄位)，registry 與 field specs 契約見 [元件與定義](../../lib/zcu_tools/resources/entry/README.md#元件與定義)。
