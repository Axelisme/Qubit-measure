# Components

**Last updated:** 2026-10-21，kind、角色與字串引用索引

新增 model、註冊與表單引用見 [component 教學](../tutorial/01-components.md)。`register_all` 註冊以下六個 kind，import 不會註冊。

| Kind | 代表什麼 | 哪些實驗讀它 |
| --- | --- | --- |
| `resonator` | Readout resonator 的頻率、linewidth 與 wiring。 | `onetone/*`、`twotone/*`、`jpa/*` 與 mux tone 初值。 |
| `group/mux` | 1 至 8 個有序 resonators 與可選、等長且不重複的 qubits，擁有 mux module 槽。 | `mux/*` 與 `ftf/*` 的表單。 |
| `device/current_source` | Flux bias 來源的 component 記錄，只有共用 ext、wiring、module。 | 不代替 driver；flux 實驗讀 device alias 指向的已連線裝置。 |
| `amplifier/jpa` | JPA pump frequency、power 與 flux bias 校準。 | `jpa/*` 的表單與 writeback。 |
| `qubit/fluxonium` | 共用 qubit 校準加上 EJ、EC、EL。 | `twotone/*` 與其他讀 qubit 校準的表單；mux／FTF 表單也可經 group 讀取。 |
| `qubit/transmon` | 共用 qubit 校準加上 EJmax、EC、無因次 d。 | 讀 qubit 校準的表單；FTF 可將它綁為 coupler。 |

## 角色與引用

內建角色為 `qubit → qubit/*`、`resonator → resonator`、`qubit1 → qubit/*`、`qubit2 → qubit/*`。Notebook 省略 roles 時用有序 shorthand `qubit, resonator`，預設焦點匹配 `qubit/*`。GUI 使用獨立 alias 表，不由這些角色自動綁定，見 [角色與失敗](../../lib/zcu_tools/resources/entry/README.md#角色與失敗)。

| 字串欄位 | 填入的名稱 |
| --- | --- |
| Qubit `resonator` | Readout resonator 元件，例如 `R1`，kind 為 `resonator`。 |
| Qubit `flux_source` | Flux bias 來源，例如 `f1_bias`；用於 device alias 時須對應 connected-device 名稱，不代表已連線。 |
| Resonator `amplifier` | Readout amplifier 元件，例如 `J1`，內建 kind 為 `amplifier/jpa`。 |

這些字串不在載入時預檢整張引用圖；解析時才查目標。單位見 [component 欄位單位](../README.md#component-欄位單位)，共用欄位與 ext 政策見 [component 教學](../tutorial/01-components.md#1-宣告欄位)，registry 與 field specs 契約見 [元件與定義](../../lib/zcu_tools/resources/entry/README.md#元件與定義)。
