# FTF YAML v2 欄位指南

**Last updated:** 2026-10-17

本指南對照 [default.yaml](default.yaml) 與 [entry-template.yaml](entry-template.yaml)。啟動方式見 [環境 README](README.md)。所有可接受的 YAML 欄位都列在下文；欄位驗證由 [resources.ftf](../../../lib/zcu_tools/resources/ftf/README.md) 負責，實際組裝與演化限制見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md)。

樣板的 circuit energies、couplings 與 resonator frequencies 採用《FTF Architecture for Quantum Computing》第 5 章 Table 5.1 的 Device A。DC、微波、接收器與數值設定包含本機手動調整值，不能當作其他晶片的校準結果。樣板可執行，不代表截斷已收斂或模型適用於所有工作點。

## 先分清楚三種設定

- `chip` 描述 F1、coupler、F2 的物理參數。Flux noise 與耗散也是物理模型的一部分。
- `dc_bias`、`drive_lines`、`readout` 描述 live DC 電源、charge microwave 接線與讀出訊號路徑。DC 電源設定靜態磁通工作點，微波線透過 charge operator 驅動迴路。
- `numerics` 決定基底維度、系統保留維度、solver 精度與亂數種子。它不描述晶片性質。

FTF 支援 DC bias、不同 round 間的靜態磁通掃描、charge drive 與 readout。它不建構 flux line 或 flux controls，沒有 gate 內快速 flux-pulse 路徑。V2 沒有 `flux_lines`，即使填 `{}` 也會按未知欄位拒絕。這不是額外的 flux-pulse 偵測機制；未綁定 generator 沿用既有拒絕行為，沒有 flux-pulse 專用報錯。

共享 `simulate` 引擎仍保留 flux schedule、source-line filter 與 flux-tail Dyson 能力，FTF 不使用這些控制。不要把 `chip` 的 flux-noise 或 flux-line loss 參數當成 flux-pulse 接線。

## 文件結構與共通規則

```yaml
format: zcu.ftf
version: 2
chip: # Physical circuits and couplings.
  # ...
dc_bias: # Static offsets and live DC devices.
  # ...
drive_lines: # Named charge microwave lines; may be omitted or empty.
  # ...
readout: # Shared drive, two physical resonators and one ADC.
  # ...
numerics: # Basis sizes, solver accuracy and root RNG seed.
  # ...
```

這是結構示意，不是可直接載入的完整文件。YAML key 順序不影響載入。表中的「必填」表示省略會報錯，沒有隱藏預設值。「有限數」排除 NaN 與正負無限大，可帶正負號，除非該列另有限制。整數欄位不可填小數、字串或布林值。所有 record 都拒絕未知欄位，錯誤會指出欄位路徑。

| 頂層欄位 | 用途與合法內容 | 必填／省略語意 |
| --- | --- | --- |
| `format` | 格式識別字串，只接受 `zcu.ftf`，無單位 | 必填 |
| `version` | 格式版本，只接受整數 `2`，無單位 | 必填；v1 不相容，字串 `"2"` 也不接受 |
| `chip` | 物理電路 record，見下一節 | 必填 |
| `dc_bias` | 靜態磁通 offset 與具名電源 record | 必填 |
| `drive_lines` | 名稱到 charge drive line 的 map | 可省略或填 `{}`，都停用 charge 控制；不可填 `null` |
| `readout` | `drive`、`resonators`、`adc` 三個必填 record | 必填，固定一個 drive、兩個 resonator、一個 ADC |
| `numerics` | `truncation`、`solver` 與可選 `seed` | 必填 |

`dc_bias.sources` 與 `drive_lines` 的 map key 是非空字串名稱。前者成為 live device 名稱，後者識別微波線，兩類名稱不可重複。`f1_bias`、`coupler_bias`、`f2_bias`、`microwave` 是樣板選用的名稱，不是新增的 YAML 欄位。

所有三迴路 map 都必須含 `f1`、`coupler`、`f2`，沒有作用也要明寫零。這三個 key 指受影響的迴路，不是另外三條線。各 map 的單位由外層欄位決定：

| Map 路徑 | 每個迴路的內容 | 必填／省略語意 |
| --- | --- | --- |
| `dc_bias.offset_phi0` | 有限數，Phi/Phi0 | 三個值皆必填 |
| `dc_bias.sources.<name>.flux_per_unit` | 有限 signed Phi0/A 或 Phi0/V | 三個值皆必填 |
| `drive_lines.<name>.charge_drive_ghz_per_gain` | 每個迴路一筆 `{real: ..., imag: ...}`，GHz/gain | 三筆 record 皆必填，每筆兩個分量也必填 |
| `readout.resonators.r1.charge_coefficients_ghz`、`r2` 的同名欄位 | 有限 signed GHz | 每個 resonator 的三個值皆必填 |

## Chip：物理電路與耦合

`chip.f1`、`chip.coupler`、`chip.f2` 都是必填 record。能量以 H/h 的 GHz 表示，不是 angular frequency，也不是 entry 常用的 MHz 頻率欄位。

### F1 與 F2

下表同時適用於 `chip.f1` 與 `chip.f2`。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `ej_ghz` | Josephson energy EJ/h | 有限數，GHz，> 0 | 必填 |
| `ec_ghz` | Charging energy EC/h | 有限數，GHz，> 0 | 必填 |
| `el_ghz` | Inductive energy EL/h | 有限數，GHz，> 0 | 必填 |
| `q_cap` | 電容損耗的 constant Q | 無因次，有限數，> 0 | 省略或 `null` 停用此損耗 |
| `q_ind` | 電感損耗的 constant Q | 無因次，有限數，> 0 | 省略或 `null` 停用此損耗 |
| `xqp` | Quasiparticle fraction；模型使用 gap 3.4e-4 eV | 無因次，有限數，≥ 0 | 省略為 0，停用此機制；不接受 `null` |
| `aphi` | 此迴路獨立的 flux-noise amplitude | Phi0/sqrt(Hz)，有限數，≥ 0 | 省略為 0，停用此 noise；不接受 `null` |

### Coupler

`chip.coupler` 是 fixed-theta SQUID 模型，不套用 fluxonium 的 `q_ind` 或 `xqp`。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `ejmax_ghz` | 最大 Josephson energy，EJ1 + EJ2 | 有限數，GHz，> 0 | 必填 |
| `ec_ghz` | Charging energy EC/h | 有限數，GHz，> 0 | 必填 |
| `asymmetry` | abs(EJ1 − EJ2)/(EJ1 + EJ2) | 無因次，有限數，0 ≤ 值 ≤ 1 | 必填 |
| `ng` | Offset charge，以 Cooper pair 數表示 | 無因次，有限 signed 數 | 省略為 0；不接受 `null` |
| `q_cap` | 電容損耗的 constant Q | 無因次，有限數，> 0 | 省略或 `null` 停用 |
| `resistance_ohm` | Charge-line real impedance，用於耗散 | Ohm，有限數，> 0 | 省略或 `null` 停用 |
| `flux_resistance_ohm` | Flux-line real impedance，用於耗散 | Ohm，有限數，> 0 | 與 `mutual_phi0_per_a` 成對提供數值；兩者皆未設定數值時停用，各自可省略或填 `null` |
| `mutual_phi0_per_a` | 該耗散模型使用的 signed mutual coupling | Phi0/A，有限數，可為 0 | 與 `flux_resistance_ohm` 成對提供，不能只填其中一個 |
| `aphi` | Coupler 獨立的 flux-noise amplitude | Phi0/sqrt(Hz)，有限數，≥ 0 | 省略為 0；不接受 `null` |

這裡的 flux-line loss 只描述耦合到環境的耗散，不會建立可播放 pulse 的控制線。

### 跨電路耦合與溫度

下列欄位直接放在 `chip`。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `g_f1_coupler_ghz` | F1 與 coupler 的 full-charge coupling | 有限 signed GHz | 必填，0 表示無此耦合 |
| `g_f2_coupler_ghz` | F2 與 coupler 的 full-charge coupling | 有限 signed GHz | 必填，0 表示無此耦合 |
| `g_f1_f2_ghz` | F1 與 F2 的 full-charge coupling | 有限 signed GHz | 必填，0 表示無此耦合 |
| `temperature_k` | 共通 bath temperature | 有限數，K，≥ 0 | 必填 |

耦合係數乘 `n_i * n_j` 的完整 charge operators，沒有依工作點的 `n01` 再正規化。填入 transition-level 的 g01 而不換算，會改變模型。

## DC bias：live 電源與靜態磁通掃描

`dc_bias.offset_phi0` 與 `dc_bias.sources` 都必填。`sources` 是具名電源 map，schema 允許空 map，此時磁通只剩 offset。樣板建立三個獨立 live 電源，每個電源都可能透過串擾影響三個迴路。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `offset_phi0.f1`、`.coupler`、`.f2` | 各迴路無電源貢獻時的 reduced-flux offset | Phi/Phi0，有限 signed 數，不限於 [0, 1] | 三個都必填 |
| `sources.<name>.unit` | 建立 device 時固定的輸出單位 | 字串 `A` 或 `V` | 必填，不能使用 mA 或 mV 字串代替 |
| `sources.<name>.limits` | 電源可用範圍，包含兩端 | 該 source unit，有限二元素 `[min, max]`，min < max | 必填 |
| `sources.<name>.initial` | 啟動時套用一次的電源值 | 該 source unit，有限數，必須在 limits 內 | 必填，不是每 round 重設值 |
| `sources.<name>.ramp_step` | Device ramp 的正步幅 | 該 source unit，有限數，> 0 | 必填，不是時間步長或 pulse duration |
| `sources.<name>.flux_per_unit.f1`、`.coupler`、`.f2` | 各迴路對此電源的 signed DC 校準 | unit 為 A 時是 Phi0/A，為 V 時是 Phi0/V；有限數 | 三個都必填，0 表示無作用；不另填校準 unit |

每 round 開始前，環境讀取 live DC snapshot，檢查單位與 limits，並固定該 round 的工作點。每個迴路的磁通為：

```text
loop_flux_phi0 = offset_phi0 + sum(live_source_value * flux_per_unit)
```

例如樣板的 `f1_bias` 是 A 單位。若把它設成 0.001 A，F1 增加 0.1282 Phi0，coupler 增加 0.002 Phi0，F2 減少 0.001 Phi0。其他 source 的貢獻再加上去。`f1_bias` 不是「只作用於 F1」的保證。

樣板三個電源的 limits 都是 `[-0.01, 0.01]` A，等於 ±10 mA。`ramp_step: 0.000001` A 等於 0.001 mA。一般靜態掃描改的是 live source 值，不必為每個點重寫 `initial`。同一 round 內的快速磁通波形不在這個介面內。

## Drive lines：charge microwave 校準

每一筆 `drive_lines.<name>` 代表一條 charge microwave line。樣板只有 `microwave` 一條，共用 generator 2，並同時作用於三個迴路。若要多條獨立線，每條都要有不同名稱與不同 `gen_ch`，且不得占用 readout generator。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `gen_ch` | 綁定的 generator index | 無單位，整數 ≥ 0，整份文件中不得重複綁定 | 每條線必填 |
| `charge_drive_ghz_per_gain` | 三迴路的複數 charge-drive 校準 map | 含 `f1`、`coupler`、`f2` 三筆 record | 每條線必填，不接受位置 list 或 amplitude/phase |
| `charge_drive_ghz_per_gain.<loop>.real` | 複數係數的實部 | 有限 signed GHz/gain | 每個迴路必填 |
| `charge_drive_ghz_per_gain.<loop>.imag` | 複數係數的虛部 | 有限 signed GHz/gain | 每個迴路必填 |

對一個迴路，令 `c = real + i*imag`，程式的 signed gain 為 `g`、complex envelope 為 `w`。驅動項採 `Re[c*g*w*exp(i carrier)] * n`，carrier 包含程式指定的頻率與 phase。它乘完整 charge operator `n`，不隱藏 `n01` 正規化，也不另藏輸入 RWA 振幅因子。`imag` 的符號會影響 quadrature，不要把 `real`、`imag` 改成絕對值。

只要不重做微波校準，一般使用者可保留樣板係數。樣板的三個迴路都是 `real: 0.1`、`imag: 0.0` GHz/gain，不是 F2 僅有小串擾。把某迴路兩個分量都填 0，才表示該線不驅動那個迴路。頻率、phase、signed gain 與 waveform 由 program 提供，不在這裡新增 YAML 欄位。

## Readout：drive → 物理 resonators → ADC

訊號路徑是 `readout.drive` 的 incident field 進入兩個物理 resonators 共用的 reciprocal、co-located two-port line，再由 `readout.adc` 轉成接收樣本。兩個 resonator 不是兩個 ADC，也不是兩條獨立 readout drive。

Resonator 的頻率、full-charge coupling 與 linewidth 決定物理響應。ADC 的 gain、phase、offset 與 noise 決定接收模型。把 ADC gain 調大不會改變 resonator photon number、物理 linewidth 或另外設定的 ADC noise PSD。

FTF 在 readout 使用該 DC 工作點的模型與 energy-basis populations。Readout 開始前的 charge controls 會演化狀態；readout 期間及之後的 charge controls 不納入讀出計算。每個 point 必須有恰好一個 active readout pulse，可對同一個綁定 ADC 開多個 trigger windows。模型假設 weak dispersive readout，強驅動下的適用性要另行確認。

### Drive

`readout.drive` 的兩個欄位都必填。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `gen_ch` | 共用 readout generator | 無單位，整數 ≥ 0，不可與任一 charge drive 重複 | 必填 |
| `input_per_gain` | Program gain 到 incident amplitude 的轉換 | sqrt(photons/ns)/gain，有限 signed 數，可為 0 | 必填，沒有猜測值 |

Carrier、envelope 與 phase 由 program 提供。`input_per_gain` 是物理入射場的刻度，不是 ADC gain。

### Resonators

`readout.resonators` 必須恰好有 `r1`、`r2` 兩筆 record。下表適用於兩者。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `frequency_ghz` | Bare resonator frequency | 有限數，GHz，> 0 | 必填 |
| `charge_coefficients_ghz.f1`、`.coupler`、`.f2` | Resonator 對三迴路 full charge `n` 的耦合 | 有限 signed GHz，0 表示無該耦合 | 三個都必填，不是 g01，也沒有 bias-dependent n01 重正規化 |
| `kappa_external_mhz` | 與共用 readout line 的 external linewidth | kappa_external/(2*pi)，MHz，有限數，≥ 0 | 必填 |
| `kappa_internal_mhz` | Internal loss linewidth | kappa_internal/(2*pi)，MHz，有限數，≥ 0 | 必填 |
| `thermal_occupation` | 殘餘 thermal photon occupation nth | 無因次，有限數，≥ 0 | 省略或 `null` 時，以共通 temperature 與 bare frequency 計算 Bose occupation；明寫 0 才停用此 occupation |

每個 resonator 的 external + internal linewidth 都必須 > 0，且兩個 resonator 至少一個的 external linewidth > 0。轉成 photon decay rate 時使用 `2*pi*kappa_mhz/1000`，單位 1/ns；不要把 angular decay rate 直接填進 MHz 欄位。

### ADC

`readout.adc` 描述唯一接收通道，不描述物理 resonator。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `ro_ch` | ADC readout index | 無單位，整數 ≥ 0 | 必填 |
| `gain` | 場到 ADC sample 的轉換 | ADC sample units / sqrt(photons/ns)，有限 signed 數，可為 0 | 必填，不縮放另加的 noise |
| `phase_deg` | 額外的 receiver phase | degree，有限 signed 數 | 省略為 0；不接受 `null` |
| `offset_adc` | 每個 sample 的 I、Q offset | ADC sample units，有限二元素 `[I, Q]` | 省略為 `[0, 0]`；不接受 `null` |
| `noise_psd_adc2_ns` | Two-sided white receiver noise spectral density | ADC²*ns，有限、對稱、positive-semidefinite 的 2×2 matrix，順序為 I、Q | 必填；零矩陣與 singular matrix 合法，不能省略代表無 noise |

矩陣的對角項是 I、Q 的 PSD，非對角項是相關 PSD，兩個非對角值必須相等。非對角項可為負，但整體仍須 positive-semidefinite。這個矩陣不是每個 sample 的 covariance，也不是三迴路 map。

```text
Cov(sample) = noise_psd_adc2_ns / dt_ns
```

`dt_ns` 是固定 mock QICK profile 的 ADC output sample interval，以 ns 計。YAML 不提供 ADC clock 或 `dt_ns` 欄位。提高 sample rate 會縮短 dt，單樣本白雜訊 variance 因此增加。`gain` 不縮放這份 noise；要改接收雜訊，直接改 PSD，而不是把 `input_per_gain` 或物理 linewidth 當成 noise 旋鈕。

## Numerics：native、local、system 與 solver

`numerics.truncation` 與 `numerics.solver` 都必填。截斷有三層，先在 native basis 建立單元件，再保留各自的 local levels，最後在耦合系統保留 `retained_levels`。增加 system 維度不能補回先前已丟棄的 local 或 native states。

### Truncation

`numerics.truncation.f1`、`.coupler`、`.f2` 三筆 record 都必填。

| 路徑，相對於 `numerics.truncation` | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `f1.native_cutoff`、`f2.native_cutoff` | Fluxonium oscillator basis dimension | 無單位，整數 ≥ 2 | 各自必填 |
| `f1.levels`、`f2.levels` | 各 fluxonium 的 local retained dimension | 無單位，整數 ≥ 1，且 ≤ 各自 native_cutoff | 各自必填 |
| `coupler.charge_cutoff` | SQUID charge basis 的 ncut，charge states 為 −ncut 到 +ncut | 無單位，整數 ≥ 1，native dimension 是 2*ncut+1 | 必填，不是直接填 dimension |
| `coupler.levels` | Coupler 的 local retained dimension | 無單位，整數 ≥ 1，且 ≤ 2*charge_cutoff+1 | 必填 |
| `retained_levels` | 耦合系統的 system retained dimension | 無單位，整數 ≥ 1，且 ≤ f1.levels * coupler.levels * f2.levels | 必填 |

樣板是 native dimensions 24、17、24，local levels 各 4，因此 local product 是 64，再保留 16 個 system levels。這只說明維度關係。判斷收斂時，應分別增加 native、local、system 維度，並比較關心的頻率、matrix elements 或量測結果。

### Solver 與 seed

| 路徑，相對於 `numerics` | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `solver.rtol` | DOP853 relative local error tolerance | 無因次，有限數，> 0 | 必填 |
| `solver.atol` | DOP853 absolute local error tolerance | Solver state 的數值尺度，有限數，> 0 | 必填 |
| `solver.max_step_ns` | 積分最大步長 | ns，有限數，> 0 | 必填 |
| `solver.bath_segment_ns` | 固定 bath 更新區間 | ns，有限數，> 0 | 必填；實際取最近的 2 ns，halfway ties to even，至少一個 2 ns tick |
| `seed` | Root RNG seed | 無單位，整數 ≥ 0 | 可省略或 `null`；環境啟動時產生一次 root seed，不是每 round 重新產生 |

FTF 預設走分段引擎的快速算法，不是所有演化都逐步執行 DOP853。這些 solver 欄位控制需要數值積分的路徑，無法代替截斷或物理模型的適用性檢查。調小 tolerance、縮短 step 或 bath segment 之後，仍須比較輸出確認精度。明寫 seed 有助於固定抽樣身份；跨 dependency 或 RNG 版本不保證相同結果。

## Generator、ADC 與 entry 接線對照

所有 channel index 都從 0 開始。數量不是綁定條數，而是最大索引加一：

```text
generator_count = max(readout.drive.gen_ch, all drive_lines.gen_ch) + 1
readout_count = readout.adc.ro_ch + 1
```

`gen_ch` 與 `ro_ch` 屬於不同索引空間，可以同為 0。每個 generator 只能綁一條 charge 或 readout line；共用線上的兩個元件共用同一個綁定，不是再綁兩次。Clock、DDS、firmware channel type 與 memory limits 使用固定 mock QICK capability profile，YAML 不提供 `channels` 或改 profile 的欄位。索引通過 schema 不代表任意 pulse 都符合該 profile。

樣板保留 readout gen0、microwave gen2，所以推導出 3 個 generator 與 1 個 readout。Gen1 是未綁定空洞，不能拿來播放另一條控制線，也不表示保留了 flux-pulse 能力。使用未綁定 generator 仍按既有行為拒絕。

[Entry 範本](entry-template.yaml) 的具體對照如下，entry 路徑以 `components` 下的元件為起點。

| 用途 | FTF YAML | Entry 欄位或引用 |
| --- | --- | --- |
| F1 live DC | `dc_bias.sources.f1_bias`，`unit: A` | `Q1.flux_source: f1_bias`，`Q1.flux_unit: A` |
| Coupler live DC | `dc_bias.sources.coupler_bias`，`unit: A` | `coupler.flux_source: coupler_bias`，`coupler.flux_unit: A` |
| F2 live DC | `dc_bias.sources.f2_bias`，`unit: A` | `Q2.flux_source: f2_bias`，`Q2.flux_unit: A` |
| Shared charge microwave | `drive_lines.microwave.gen_ch: 2` | `Q1.wiring.ch: 2`、`Q2.wiring.ch: 2` |
| Shared readout drive | `readout.drive.gen_ch: 0` | `R1.wiring.ch: 0`、`R2.wiring.ch: 0` |
| Shared ADC | `readout.adc.ro_ch: 0` | `R1.wiring.ro_ch: 0`、`R2.wiring.ro_ch: 0` |
| Resonator 身分 | `readout.resonators.r1`、`.r2` | `R1`、`R2`；`Q1.resonator: R1`、`Q2.resonator: R2` |

Coupler 有靜態 DC source，但沒有 `lo_flux_ch` 綁定。不要因為看到 `coupler.flux_source` 就替它填快速 flux channel。Shared microwave 的 coupler 係數仍會作用於模型中的 coupler，與 entry 是否有獨立 coupler pulse channel 是不同的事。

FTF YAML 和 entry 是兩份獨立文件，loader 不自動互改或互相比對。修改 source 名稱、unit、gen_ch 或 ro_ch 時，必須同步修改使用中的 entry 引用與 wiring。修改 energies 時也要同步 entry 的 EJ/EC/EL 或 EJmax/EC/d。樣板 `r1/r2.frequency_ghz` 是 7.19/7.08 GHz，entry 的 R1/R2.freq 是 7190/7080 MHz。Qubit frequency 隨 bias 改變，entry 保留 `null`，不能直接拿 EJ 當 transition frequency。

## 採用 v2 與調參順序

1. 保留自己的工作副本，再以 [default.yaml](default.yaml) 的 v2 結構整理設定。V1 文件不能載入，沒有自動轉換或舊 key aliases。舊預覽若仍含 `flux_lines`，必須自行移除，空 map 也不合法。不要直接覆蓋還沒保存的本機調整。
2. 先核對物理模型、DC source 名稱、A/V 單位、limits 與 offset。用 live DC 值選靜態工作點，只有校準改變時才調 `flux_per_unit`。
3. 對照上表核對 YAML 與 entry 的 source 引用及 channel。環境只建立 live device，entry 不保存 driver；套用新範本不會自動更新既有 entry。
4. 一般量測調整 program 的 frequency、phase、gain 與 waveform。只有重做微波或接收校準時，才修改 `charge_drive_ghz_per_gain`、`input_per_gain`、ADC gain/noise 等刻度。
5. 分別檢查截斷與 solver 收斂。Schema 驗證只檢查格式、數值與接線限制，不會替你判斷物理適用性或精度。
6. 修改 YAML 或環境後重啟 GUI／重建環境。`load_ftf` 只讀一次，不會 hot reload，也不會覆寫本機 `ftf.yaml` 或既有 entry。
