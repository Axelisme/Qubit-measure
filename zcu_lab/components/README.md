# Components

**Last updated:** 2026-10-27，runtime v3 與 retained v1→v2→v3 升級

**讀者：** lab-author、agent。查詢 component kind、角色與字串引用。

新增 model、註冊與表單引用見 [component 教學](../tutorial/01-components.md)。Component 保存校準值、wiring 與 module 引用，不宣告 program 的執行型 cfg。Core 的 module 欄位使用 `Pulse.Cfg`、`Readout.Cfg`、`MuxReadout.Cfg` 等型別；寫法見 [實驗教學](../tutorial/03-experiment.md#2-在-corepy-定義資料與量測)。`register_all` 註冊以下六個 kind，import 不會註冊。

| Kind | 代表什麼 | 哪些實驗讀它 |
| --- | --- | --- |
| `resonator` | Readout resonator 的頻率、linewidth 與 wiring。 | `onetone/*`、`twotone/*`、`jpa/*` 與 mux tone 初值。 |
| `group/mux` | 1 至 8 個有序 resonators 與可選、等長且不重複的 qubits，擁有 mux wiring、module 槽與 global coil 來源。 | `mux/*` 與 `two_qubit/*` 的表單。 |
| `device/current_source` | Flux bias 來源的 component 記錄，只有共用 ext、module，沒有 wiring。 | 不代替 driver；flux 表單從實驗元件的 flux_dev 取得裝置初值，Run 使用選定的已連線 driver。 |
| `amplifier/jpa` | JPA pump frequency、power、flux bias 校準及可空的 flux_dev／rf_source 裝置名稱。 | `jpa/*` 的表單與 writeback。 |
| `qubit/fluxonium` | 共用 qubit 校準加上 EJ、EC、EL。 | `twotone/*` 與其他讀 qubit 校準的表單；mux／FTF 表單也可經 group 讀取。 |
| `qubit/transmon` | 共用 qubit 校準加上 EJmax、EC、無因次 d。 | 讀 qubit 校準的表單；FTF 可將它綁為 coupler。 |

## Component 文件版本

`register_all` 同時註冊 `parameter_upgrades.PARAMETER_SCHEMA_POLICY`，目前為 `schema_version: 3`。Setup、point 與 entry template 都使用這個整份文件版本，容器版本不變。結構變更的版本宣告、完整舊格式分類與純函式步驟由本 package 擁有。通用執行與 I/O 由 resources 擁有，見 [ADR-0063](../../docs/adr/0063-persistence-ownership.md#component-文件版本與明確升級)。

長期入口是 `scripts/upgrade_parameters.py`。`parameter_upgrade_plan()` 保留完整分類與 v1→v2→v3 純函式步驟。未標版文件只在完整 v1 分類成功後視為 v1。升級 target 與 runtime 都為 3。Runtime 不接受舊版，也不在載入時升級。`flux_upgrade.py` 驗證名稱語法與歷史校準，不做 I/O 或 device catalog 查詢。V2 保留每台儀器的 half／integer／period 與 `flux_source`。V2→v3 才改名為 `flux_dev`，並保留主儀器的 half 分支，扣除其他儀器的 integer／period，不做 mod 1。V1 coil 只保留 signed period，原 half／int 進報告；v2 中間值為 integer 0、half = period／2。\n\n`storage_mapping` 經中性完整文件 batch callback 重用轉換。完整有效的歷史校準取代繼承 offset 與同儀器 period，其他 period 保留。同批撞鍵、partial、非法名稱、數值與 spacing 仍拒絕。缺名產生不同的 `unnamed_device`／`unnamed_coil`，兩個入口逐目的位置提醒改名。Native 升級仍拒絕混版文件。長期步驟不依賴 migrations。版本形狀見 [flux 格式歷程](../../docs/storage-format-history/component-flux.md)，runner 操作見 [升級命令](../../docs/parameter-upgrades.md)。

## Qubit flux frame

Qubit 的 `flux` 使用 `QubitFluxSchema`，預設為 `{offset: null, period: {}}`。
`offset` 是所有儀器輸出為 0 時的 reduced flux，以 Φ0 為單位；null 表示尚未校準。
`period` 以 device 名稱對應有限、非零的 signed period，使用各儀器原生單位／Φ0。
負 period 表示增加 bias 會減少 flux。有 period 但當下沒有值的儀器不貢獻 flux，period 仍保留。
正向計算與 half／integer 投影由 `analysis.flux` 提供。
`QubitFluxCalibration` 只用於擬合點與歷史轉換，不是 runtime component 的儲存形狀。
歷史步驟以 `validate_spacing(device_name)` 檢查 `abs(integer-half) = abs(period)/2`，失敗以 `ValueError` 指出 device。
此檢查使用 relative tolerance `1e-9`、absolute tolerance `0`，不改寫校準值。

## Wiring

`BuiltinComponent` 只提供 `ext`、`module` 與未知欄位 migration。Wiring 由各 kind 宣告，不從基底繼承。

| Kind | Wiring 欄位 | 用途 |
| --- | --- | --- |
| `resonator` | `ch`、`ro_ch` | 單 tone generator 與 ADC。 |
| `qubit/fluxonium`、`qubit/transmon` | `ch`、`lo_flux_ch` | Drive generator 與 local flux channel，由 `QubitBaseSchema` 宣告。 |
| `group/mux` | `gen_ch`、`ro_ch` 列表 | Mux generator 與逐 tone ADC，`ro_ch[i]` 對應 `resonators[i]`，長度必須相同。 |
| `amplifier/jpa`、`device/current_source` | 無 | 表單從元件的 flux_dev／rf_source 推導裝置名稱，再使用選定的已連線 driver，不讀 component channel。 |

Mux wiring 不從各 resonator 的單 tone wiring 推導。FTF 的 R1／R2 使用 pulse ADC0，MG12 則使用 mux ADC1／ADC2。

Mux generator 未知時可省略 `gen_ch` 或填 null，但 group 必須提供 wiring 與等長的 `ro_ch` 列表。未宣告的 wiring 輸入沿既有規則移到 `ext.migrated`，不會成為有效通道。設計依據見 [ADR-0072](../../docs/adr/0072-module-boundaries.md#使用者元件與模擬環境的放置)。

## 角色與引用

內建角色為 `qubit → qubit/*`、`resonator → resonator`、`qubit1 → qubit/*`、`qubit2 → qubit/*`。Notebook 省略 roles 時用有序 shorthand `qubit, resonator`，預設焦點匹配 `qubit/*`。GUI 使用獨立 alias 表，不由這些角色自動綁定，見 [角色與失敗](../../lib/zcu_tools/resources/entry/README.md#角色與失敗)。

| 字串欄位 | 填入的名稱 |
| --- | --- |
| Qubit `resonator` | Readout resonator 元件，例如 `R1`，kind 為 `resonator`。 |
| Qubit `flux_dev` | 自己的 DC 線 device 名稱，例如 `Q1_bias`，不代表已連線。 |
| Mux group `flux_dev` | Global coil 的 device 名稱，例如 `coil_bias`，不受 qubits 是否提供影響，不代表已連線。 |
| JPA `flux_dev` | JPA bias 的 DC device 名稱，可為 null，不代表已連線。 |
| JPA `rf_source` | JPA pump 的 RF device 名稱，可為 null，不代表已連線。 |
| Resonator `amplifier` | Readout amplifier 元件，例如 `J1`，內建 kind 為 `amplifier/jpa`。 |

三種元件的 `flux_dev` 都是 flux 的主要（預設）控制儀器，不參與 φ 計算，也不必出現在 flux 校準中。Form 在 Open／Reset 取此欄位作可覆寫的裝置初值，null 使裝置欄位留空。

Qubit 的 `flux` 保存自己的 local、global coil 與其他 contributor 校準。Mux group 只保存 `flux_dev`，不另擁有 qubit 校準。`wiring.lo_flux_ch` 仍是 fast-flux 通道，不是 DC device。

這些字串不在載入時預檢整張引用圖；解析時才查目標。單位見 [component 欄位單位](../README.md#component-欄位單位)，共用欄位與 ext 政策見 [component 教學](../tutorial/01-components.md#1-宣告欄位)，registry 與 field specs 契約見 [元件與定義](../../lib/zcu_tools/resources/entry/README.md#元件與定義)。
