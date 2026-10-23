# Components

**Last updated:** 2026-10-23，DC 局部線與 global coil 校準

**讀者：** lab-author、agent。查詢 component kind、角色與字串引用。

新增 model、註冊與表單引用見 [component 教學](../tutorial/01-components.md)。`register_all` 註冊以下六個 kind，import 不會註冊。

| Kind | 代表什麼 | 哪些實驗讀它 |
| --- | --- | --- |
| `resonator` | Readout resonator 的頻率、linewidth 與 wiring。 | `onetone/*`、`twotone/*`、`jpa/*` 與 mux tone 初值。 |
| `group/mux` | 1 至 8 個有序 resonators 與可選、等長且不重複的 qubits，擁有 mux wiring、module 槽與 global coil 校準。 | `mux/*` 與 `two_qubit/*` 的表單。 |
| `device/current_source` | Flux bias 來源的 component 記錄，只有共用 ext、module，沒有 wiring。 | 不代替 driver；flux 實驗讀 device alias 指向的已連線裝置。 |
| `amplifier/jpa` | JPA pump frequency、power 與 flux bias 校準。 | `jpa/*` 的表單與 writeback。 |
| `qubit/fluxonium` | 共用 qubit 校準加上 EJ、EC、EL。 | `twotone/*` 與其他讀 qubit 校準的表單；mux／FTF 表單也可經 group 讀取。 |
| `qubit/transmon` | 共用 qubit 校準加上 EJmax、EC、無因次 d。 | 讀 qubit 校準的表單；FTF 可將它綁為 coupler。 |

## Wiring

`BuiltinComponent` 只提供 `ext`、`module` 與未知欄位 migration。Wiring 由各 kind 宣告，不從基底繼承。

| Kind | Wiring 欄位 | 用途 |
| --- | --- | --- |
| `resonator` | `ch`、`ro_ch` | 單 tone generator 與 ADC。 |
| `qubit/fluxonium`、`qubit/transmon` | `ch`、`lo_flux_ch` | Drive generator 與 local flux channel，由 `QubitBaseSchema` 宣告。 |
| `group/mux` | `gen_ch`、`ro_ch` 列表 | Mux generator 與逐 tone ADC，`ro_ch[i]` 對應 `resonators[i]`，長度必須相同。 |
| `amplifier/jpa`、`device/current_source` | 無 | 表單透過 device alias 使用已連線 driver，不讀 component channel。 |

Mux wiring 不從各 resonator 的單 tone wiring 推導。FTF 的 R1／R2 使用 pulse ADC0，MG12 則使用 mux ADC1／ADC2。

Mux generator 未知時可省略 `gen_ch` 或填 null，但 group 必須提供 wiring 與等長的 `ro_ch` 列表。未宣告的 wiring 輸入沿既有規則移到 `ext.migrated`，不會成為有效通道。設計依據見 [ADR-0072](../../docs/adr/0072-module-boundaries.md#使用者元件與模擬環境的放置)。

## 角色與引用

內建角色為 `qubit → qubit/*`、`resonator → resonator`、`qubit1 → qubit/*`、`qubit2 → qubit/*`。Notebook 省略 roles 時用有序 shorthand `qubit, resonator`，預設焦點匹配 `qubit/*`。GUI 使用獨立 alias 表，不由這些角色自動綁定，見 [角色與失敗](../../lib/zcu_tools/resources/entry/README.md#角色與失敗)。

| 字串欄位 | 填入的名稱 |
| --- | --- |
| Qubit `resonator` | Readout resonator 元件，例如 `R1`，kind 為 `resonator`。 |
| Qubit `flux_source` | 自己的 DC 線 device 名稱，例如 `Q1_bias`，不代表已連線。 |
| Mux group `flux_source` | Global coil 的 device 名稱，例如 `coil_bias`，不受 qubits 是否提供影響，不代表已連線。 |
| Resonator `amplifier` | Readout amplifier 元件，例如 `J1`，內建 kind 為 `amplifier/jpa`。 |

Qubit 的 `flux_half`／`flux_period`／`flux_int` 保存自己的 DC 線校準，單位由 `flux_unit` 指定。`wiring.lo_flux_ch` 仍是 fast-flux 通道，不是 DC device。

Mux group 的 `coil_flux_half`／`coil_flux_period`／`coil_flux_int` 保存每顆 qubit 對 global coil 的校準，使用 coil device 的原生單位，依 qubits 順序排列。整欄 null 表示未知，list 內的 null 表示該 qubit 未校準。qubits 為 None 或空時，任一非 null 校準欄位都拒絕，包含空 list，錯誤訊息為「coil 校準需要 qubits 順序對齊」。qubits 有值時，非 null 校準 list 必須與 qubits 等長。

這些字串不在載入時預檢整張引用圖；解析時才查目標。單位見 [component 欄位單位](../README.md#component-欄位單位)，共用欄位與 ext 政策見 [component 教學](../tutorial/01-components.md#1-宣告欄位)，registry 與 field specs 契約見 [元件與定義](../../lib/zcu_tools/resources/entry/README.md#元件與定義)。
