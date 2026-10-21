# FTF YAML v2 欄位指南

**Last updated:** 2026-10-21，segment 主分支腔場繼承

**讀者：** lab-author、framework-dev、agent。依 YAML 欄位、單位與模型限制設定 FTF 環境並對照 entry 接線。

本指南對照 [default.yaml](default.yaml) 與 [entry-template.yaml](entry-template.yaml)。啟動方式見 [環境 README](README.md)。所有可接受的 YAML 欄位都列在下文；欄位驗證由 [resources.ftf](../../../lib/zcu_tools/resources/ftf/README.md) 負責，實際組裝與演化限制見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md)。

樣板的 circuit energies、couplings 與 resonator frequencies 採用《FTF Architecture for Quantum Computing》第 5 章 Table 5.1 的 Device A。DC、微波、接收器與數值設定包含本機手動調整值，不能當作其他晶片的校準結果。樣板可執行，不代表截斷已收斂或模型適用於所有工作點。

## 目錄

- [先分清楚三種設定](#先分清楚三種設定)
- [文件結構與共通規則](#文件結構與共通規則)
- [Chip：物理電路與耦合](#chip物理電路與耦合)
- [DC bias：live 電源與靜態磁通掃描](#dc-biaslive-電源與靜態磁通掃描)
- [Drive lines：charge 與小訊號 flux 校準](#drive-linescharge-與小訊號-flux-校準)
- [Readout：drive → 物理 resonators → ADC](#readoutdrive--物理-resonators--adc)
- [Numerics：native、local、system 與 solver](#numericsnativelocalsystem-與-solver)
- [Generator、ADC 與 entry 接線對照](#generatoradc-與-entry-接線對照)
- [採用 v2 與調參順序](#採用-v2-與調參順序)

## 先分清楚三種設定

- `chip` 描述 F1、coupler、F2 的物理參數。Flux noise 與耗散也是物理模型的一部分。
- `dc_bias`、`drive_lines`、`readout` 描述 live DC 電源、charge/flux 接線與讀出訊號路徑。DC 電源設定靜態磁通工作點，drive line 透過所選 operator 驅動迴路。
- `numerics` 決定基底維度、系統保留維度、solver 精度與亂數種子。它不描述晶片性質。

FTF 支援 DC bias、不同 round 間的靜態磁通掃描、charge drive、小訊號 flux drive 與 readout。Flux drive 使用固定 DC 工作點的一階線性算符。這個近似適用於小訊號驅動，不支援 gate 內逐時重算 H(Phi(t)) 的大振幅 fast-flux。Loader 將 `flux_lines` 視為未知欄位，即使填 `{}` 也會拒絕。使用未綁定的 generator 也會失敗。

`chip` 的 flux-noise 與 flux-line loss 參數描述物理雜訊與耗散。FTF 的 flux-pulse 接線由 `drive_lines` 宣告。共享 `simulate` 引擎另有 flux schedule、source-line filter 與 flux-tail Dyson 能力，這些控制在 FTF 中停用。

## 文件結構與共通規則

```yaml
format: zcu.ftf
version: 2
chip: # Physical circuits and couplings.
  # ...
dc_bias: # Static offsets and live DC devices.
  # ...
drive_lines: # Named charge/flux drive lines; may be omitted or empty.
  # ...
readout: # Shared drive, two physical resonators and bound ADC receivers.
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
| `drive_lines` | 名稱到 charge/flux drive line 的 map | 可省略或填 `{}`，都停用 drive 控制；不可填 `null` |
| `readout` | `drive`、`resonators`、`adc` | 必填，一個 drive、兩個 resonator；pulse 使用一個 ADC object，mux 使用 receiver list |
| `numerics` | `truncation`、`solver` 與可選 `seed` | 必填 |

`dc_bias.sources` 與 `drive_lines` 的 map key 是非空字串名稱。前者成為 live device 名稱，後者識別微波線，兩類名稱不可重複。`f1_bias`、`coupler_bias`、`f2_bias`、`f1_charge`、`f2_charge` 是樣板選用的名稱，不是新增的 YAML 欄位。

所有三迴路 map 都必須含 `f1`、`coupler`、`f2`，沒有作用也要明寫零。這三個 key 指受影響的迴路，不是另外三條線。各 map 的單位由外層欄位決定：

| Map 路徑 | 每個迴路的內容 | 必填／省略語意 |
| --- | --- | --- |
| `dc_bias.offset_phi0` | 有限數，Phi/Phi0 | 三個值皆必填 |
| `dc_bias.sources.<name>.flux_per_unit` | 有限 signed Phi0/A 或 Phi0/V | 三個值皆必填 |
| `drive_lines.<name>.charge_drive_ghz_per_gain` | 每個迴路一筆 `{real: ..., imag: ...}`，GHz/gain | charge line 必填三筆 record 與所有分量 |
| `drive_lines.<name>.flux_drive_phi0_per_gain` | 每個迴路一筆 `{real: ..., imag: ...}`，Phi0/gain | flux line 必填三筆 record 與所有分量 |
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

樣板三個電源的 limits 都是 `[-0.02, 0.02]` A，等於 ±20 mA。`ramp_step: 0.000001` A 等於 0.001 mA。一般靜態掃描改的是 live source 值，不必為每個點重寫 `initial`。同一 round 內的快速磁通波形不在這個介面內。

## Drive lines：charge 與小訊號 flux 校準

每一筆 `drive_lines.<name>` 代表一條獨立 generator 線。樣板有 `f1_charge` gen2 與 `f2_charge` gen3，均為 charge。每條線都要有不同名稱與不同 `gen_ch`，且不得占用 readout generator。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `gen_ch` | 綁定的 generator index | 無單位，整數 ≥ 0，整份文件中不得重複綁定 | 每條線必填 |
| `operator` | 選擇驅動算符 | `charge` 或 `flux` | 省略為 `charge`，不接受 null |
| `charge_drive_ghz_per_gain` | 三迴路的複數 charge 校準 map | 每筆含有限 signed `real`、`imag`，GHz/gain | charge line 必填，flux line 不可提供 |
| `flux_drive_phi0_per_gain` | 三迴路的複數 flux 校準 map | 每筆含有限 signed `real`、`imag`，Phi0/gain | flux line 必填，charge line 不可提供 |

校準 map 必含 `f1`、`coupler`、`f2`，不接受位置 list 或 amplitude/phase，也不接受 null。每條線只填所選 operator 的校準欄位，不可同時提供兩個。各迴路的獨立係數照舊表達串擾。

對一個迴路，令 `c = real + i*imag`，程式的 signed gain 為 `g`、complex envelope 為 `w`。驅動項是 `Re[c*g*w*exp(i carrier)] * O`，carrier 含程式的 frequency 與 phase。Charge 的 O 是完整電荷 n，不隱藏 n01 正規化或輸入 RWA 振幅因子。Flux 的 O 是 FTF 現用 Hamiltonian 的 ∂(H/h)/∂f，f = Phi/Phi0，單位 GHz/Phi0。Phi0/gain 係數乘此算符後仍為 GHz/gain。兩種 operator 都保留 `imag` 的符號，Y quadrature 使用負虛部。

Flux operator 在每 round 的 live DC 工作點用解析導數求值，經固定 local basis 與 retained joint basis 投影。Fluxonium 使用固定 oscillator 座標，H/h 的 Josephson 項是 `-EJ*cos(phi+2*pi*f)`，導數為 `2*pi*EJ*sin(phi+2*pi*f)`。非對稱 SQUID 使用 fixed theta，導數為 `pi*EJmax*[sin(pi*f)*cos(theta)+d*cos(pi*f)*sin(theta)]`。它不是單純相位算符，也不能在 asymmetry 非零時省略 SQUID 的 sin(theta) 項。

這是 DC 附近的小訊號線性近似，drive 期間不隨振幅重算 H(Phi(t)) 或 basis。大振幅 fast-flux 不在此契約內。DC 校準仍由 `dc_bias.sources` 提供，樣板沒有交流 flux line。

樣板的兩條 charge line 各對主迴路填 real 0.1、對 coupler 填 real 0.01、對另一 fluxonium 填 0，imag 全為 0，單位 GHz/gain。這些是模擬假設，不是實測校準。把某迴路兩個分量都填 0 才表示不驅動該迴路。Frequency、phase、signed gain 與 waveform 由 program 提供。

## Readout：drive → 物理 resonators → ADC

訊號路徑是 `readout.drive` 的 incident field 進入兩個物理 resonators 共用的 reciprocal、co-located two-port line，再由 `readout.adc` 轉成接收樣本。兩個 resonator 不是兩個 ADC，也不是兩條獨立 readout drive。

Resonator 的頻率、full-charge coupling 與 linewidth 決定物理響應。ADC 的 gain、phase、offset 與 noise 決定接收模型。把 ADC gain 調大不會改變 resonator photon number、物理 linewidth 或另外設定的 ADC noise PSD。

FTF 在 terminal readout 使用該 DC 工作點的模型與 energy-basis populations。Terminal 起點是每個 point 的 readout generator 上最晚 active block 的起點，首個 ADC trigger 不得早於它。Pulse 模式每個 point 必須恰好有一個 active terminal readout pulse。Mux 模式允許 1 至 8 個共用 terminal 起點的 tone，每個 tone 有自己的 ADC。各 ADC 可有自己的 trigger windows，同一 shot index 的全部 ADC 共用一次 joint-state 抽樣，receiver noise 各自獨立。所有 tone 都進入同一條 line，每個 ADC 解調總場，不把 tone 固定對應到 r1 或 r2。

### Cavity

Readout generator 上的非終端脈衝是腔驅動，必須在 terminal 起點前或恰好於該起點結束。跨越 terminal 起點的脈衝會引發 `ValueError`。腔驅動使用相同的 `input_per_gain` 換算，由 `simulate.adc` 求解線性、能階相依的複數腔場。每個聯合能階 s、resonator r 的光子數是 `n[s,r] = |alpha[s,r]|²`。Stark 能量頻移是 `sum_r chi[s,r] * n[s,r]`，單位 GHz。這裡的 chi 是模型的能階相依腔頻減裸腔頻，不是另外輸入的校準值。能階分支的輸出場差異產生 photon-shot-noise 去相位，包含共用線路的干涉與各腔的 internal loss。這份驅動去相位與殘餘 thermal-photon 去相位分開計算。

FTF 將腔驅動起迄加入演化切分，對每個最終 segment 使用平均 Stark 頻移與平均驅動去相位率。段內不逐時套用瞬時場。脈衝結束後保留 ringdown，不清空腔場。Readout 開始前的 charge/flux drive controls 仍會演化狀態，terminal readout 期間及之後的 drive controls 不納入讀出計算。

FTF 依序推進各 segment 的 density 與腔場。每段開始時，所有能階的條件場都繼承上一段主分支的段末場，再依各自能階演化。第一段從零場開始。主分支取段首 density 中 population 最大的能階，相同時取最低能量。這讓探測段保留 prepared 分支留下的腔場。疊加態只攜帶一個主分支的場是半古典近似。這個逐 segment 平均並繼承主分支場的模型，對疊加態與 finite qubit drive 不描述完整的 qubit 與腔場量子糾纏。

CKP 兩個 preparation 分支的腔中心分裂由 χ/κ 決定。κ ≫ χ 時，兩分支中心趨近。這個趨勢不保證有限探測強度下的擬合中心精度。

同條件的 CKP 與獨立兩能階 qubit × 兩腔 Fock 截斷 Lindblad 比較仍未通過。保存的掃描使用 1 mK、Φ=(0.44, 0.20, 0.49)Φ0、κ/(2π)=1.1 MHz、|χe−χg|≈0.418869 MHz。脈衝為名義 π 1 µs、prefill 1.5 µs、probe 4 µs，有限探測強度 ΩT=1.049885。掃描包含 25 個腔頻率、51 個 qubit 頻率與 g／e 兩種 preparation，每點 4096 reps、1 round。

以 finite estimator 擬合，Run 的 e−g 腔中心差為 −23.6374 kHz，同條件獨立 Lindblad 基準為 −134.5712 kHz。兩者相差 110.934 kHz，是原定 scaled SE 5.9727 kHz 的 18.57 倍。Fock 截斷與頻率網格的收斂變化均小於 local SE 2.8627 kHz，仍未達到中心差在誤差內一致的驗收條件。此次 Lorentzian 擬合所得的有效 κ 不等於 bare κ，且對 fit window 敏感，不能保證 κ 在 fit error 內與 bare 值一致。

此模型可用於半古典趨勢與量測流程驗證，不應把 finite-drive CKP 的中心精度視為完整 Lindblad 結果或校準依據。目前沒有已證明能保證通過此精度驗收的 χ/κ、驅動強度或脈衝時序數值適用界。

模型假設 weak dispersive coupling，不包含 MIST 躍遷或其 onset。只施加腔驅動、沒有 qubit drive 時，這些新增頻移與去相位不改變能階 populations。若 Stark 頻移讓能量排序反轉，演化會報錯並指出 point 與 segment，不重排能階。未報錯不代表強驅動、截斷或色散近似已通過適用性驗證。

### Drive

`readout.drive.kind` 只接受 `pulse` 或 `mux`，省略為 `pulse`。Pulse 使用既有 v6 generator／v2 ADC profile，mux 使用固定 mixmux8／PFBv4 profile。這是離線編譯能力，不代表連接真機，也不提供 clock override。

`readout.drive` 的兩個欄位都必填。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `gen_ch` | 共用 readout generator | 無單位，整數 ≥ 0，不可與任一 drive line 重複 | 必填 |
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
| `mismatch_phase_rad` | 此 resonator 散射項相對直通項的相位 | radians，有限 signed 數 | 省略為 0；不接受 `null` 或非有限值 |
| `thermal_occupation` | 殘餘 thermal photon occupation nth | 無因次，有限數，≥ 0 | 省略或 `null` 時，以共通 temperature 與 bare frequency 計算 Bose occupation；明寫 0 才停用此 occupation |

每個 resonator 的 external + internal linewidth 都必須 > 0，且兩個 resonator 至少一個的 external linewidth > 0。轉成 photon decay rate 時使用 `2*pi*kappa_mhz/1000`，單位 1/ns；不要把 angular decay rate 直接填進 MHz 欄位。

`mismatch_phase_rad` 分別控制 r1、r2。微波 IQ 中，散射項乘 `exp(+i*phi)`，符號與 hanger 分析模型一致。零相位保留對稱線形，非零相位可產生不對稱線形。它不旋轉整個訊號，也不改 linewidth、shared-line damping 或量子演化。整體接收相位仍由 ADC 的 `phase_deg` 控制。

### ADC

`kind: pulse` 時，`readout.adc` 是一個 object。`kind: mux` 時，它是 1 至 8 個 receiver object 的 list，ro_ch 必須唯一，只有一個 tone 也使用 list。以下欄位對每個 receiver 都適用。

```yaml
readout:
  drive: {kind: mux, gen_ch: 0, input_per_gain: 10.0}
  resonators: # Preserve the existing physical r1/r2 values.
    # ...
  adc:
    - ro_ch: 0
      gain: 100.0
      noise_psd_adc2_ns: [[1.0, 0.0], [0.0, 1.0]]
    - ro_ch: 1
      gain: 100.0
      noise_psd_adc2_ns: [[1.0, 0.0], [0.0, 1.0]]
```

這是 readout 結構示意，不是完整設定。Tone 的頻率、gain、phase、播放長度與 ADC window 仍由 program 的 MuxReadoutCfg 提供。YAML list 的順序不改輸出順序，raw 與 acquire 結果沿 program tone 的宣告順序排列。

`readout.adc` 描述唯一接收通道，不描述物理 resonator。

| 欄位 | 用途 | 單位與合法範圍 | 必填／省略語意 |
| --- | --- | --- | --- |
| `ro_ch` | ADC readout index | 無單位，整數 ≥ 0 | 必填 |
| `gain` | 場到 ADC sample 的轉換 | ADC sample units / sqrt(photons/ns)，有限 signed 數，可為 0 | 必填，不縮放另加的 noise |
| `time_of_flight_ns` | 總讀取線延遲，同時決定波形抵達時間與 electronic-delay phase | ns，有限數，≥ 0 | 省略為 0；不接受 `null`、負數或非有限值 |
| `phase_deg` | 額外的 receiver phase | degree，有限 signed 數 | 省略為 0；不接受 `null` |
| `offset_adc` | 每個 sample 的 I、Q offset | ADC sample units，有限二元素 `[I, Q]` | 省略為 `[0, 0]`；不接受 `null` |
| `noise_psd_adc2_ns` | Two-sided white receiver noise spectral density | ADC²*ns，有限、對稱、positive-semidefinite 的 2×2 matrix，順序為 I、Q | 必填；零矩陣與 singular matrix 合法，不能省略代表無 noise |

`time_of_flight_ns` 把物理輸出延後 τ，包含 lookback 的前導無訊號、ringup、ringdown 與 shaped envelope。它同時產生 `exp(-i*2*pi*f*τ)`，f 使用絕對 carrier frequency。τ = 400 ns 時，額外相位斜率約為 -2.513 rad/MHz。沒有獨立 edelay 旋鈕，也不需要為了此延遲修改 pulse 或 trigger metadata。

ADC 先延後物理場，再解調並取共軛，轉成微波分析的 I+iQ。接著套 gain 與 `phase_deg`，加入 offset，最後加入 noise。前導無訊號區仍有 offset 與 noise。共軛不翻轉 offset 或 noise 的 I/Q covariance，receiver phase 也不旋轉 noise。

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

`gen_ch` 與 `ro_ch` 屬於不同索引空間，可以同為 0。每個 generator 只能綁一條 charge、flux 或 readout line；共用線上的兩個元件共用同一個綁定，不是再綁兩次。Clock、DDS、firmware channel type 與 memory limits 使用固定 mock QICK capability profile，YAML 不提供 `channels` 或改 profile 的欄位。索引通過 schema 不代表任意 pulse 都符合該 profile。

樣板保留 readout gen0、f1_charge gen2、f2_charge gen3，所以推導出 4 個 generator 與 1 個 readout。Gen1 是未綁定空洞，不能拿來播放另一條控制線，也不表示保留了 flux-pulse 能力。使用未綁定 generator 仍按既有行為拒絕。

[Entry 範本](entry-template.yaml) 的具體對照如下，entry 路徑以 `components` 下的元件為起點。

| 用途 | FTF YAML | Entry 欄位或引用 |
| --- | --- | --- |
| F1 live DC | `dc_bias.sources.f1_bias`，`unit: A` | `Q1.flux_source: f1_bias`，`Q1.flux_unit: A` |
| Coupler live DC | `dc_bias.sources.coupler_bias`，`unit: A` | `coupler.flux_source: coupler_bias`，`coupler.flux_unit: A` |
| F2 live DC | `dc_bias.sources.f2_bias`，`unit: A` | `Q2.flux_source: f2_bias`，`Q2.flux_unit: A` |
| F1 charge line | `drive_lines.f1_charge.gen_ch: 2` | `Q1.wiring.ch: 2` |
| F2 charge line | `drive_lines.f2_charge.gen_ch: 3` | `Q2.wiring.ch: 3` |
| Shared readout drive | `readout.drive.gen_ch: 0` | `R1.wiring.ch: 0`、`R2.wiring.ch: 0` |
| Shared ADC | `readout.adc.ro_ch: 0` | `R1.wiring.ro_ch: 0`、`R2.wiring.ro_ch: 0` |
| Resonator 身分 | `readout.resonators.r1`、`.r2` | `R1`、`R2`；`Q1.resonator: R1`、`Q2.resonator: R2` |

Coupler 有靜態 DC source，但沒有 `lo_flux_ch` 綁定。不要因為看到 `coupler.flux_source` 就替它填快速 flux channel。兩條 charge line 的 coupler 串擾係數仍會作用於模型中的 coupler，與 entry 是否有獨立 coupler pulse channel 是不同的事。

FTF YAML 和 entry 是兩份獨立文件，loader 不自動互改或互相比對。修改 source 名稱、unit、gen_ch 或 ro_ch 時，必須同步修改使用中的 entry 引用與 wiring。修改 energies 時也要同步 entry 的 EJ/EC/EL 或 EJmax/EC/d。樣板 `r1/r2.frequency_ghz` 是 7.19/7.08 GHz，entry 的 R1/R2.freq 是 7190/7080 MHz。Qubit frequency 隨 bias 改變，entry 保留 `null`，不能直接拿 EJ 當 transition frequency。

## 採用 v2 與調參順序

1. 保留自己的工作副本，再以 [default.yaml](default.yaml) 的 v2 結構整理設定。V1 文件不能載入，沒有自動轉換或舊 key aliases。舊預覽若仍含 `flux_lines`，必須自行移除，空 map 也不合法。不要直接覆蓋還沒保存的本機調整。
2. 先核對物理模型、DC source 名稱、A/V 單位、limits 與 offset。用 live DC 值選靜態工作點，只有校準改變時才調 `flux_per_unit`。
3. 對照上表核對 YAML 與 entry 的 source 引用及 channel。環境只建立 live device，entry 不保存 driver；套用新範本不會自動更新既有 entry。
4. 一般量測調整 program 的 frequency、phase、gain 與 waveform。只有重做 drive 或接收校準時，才修改 `charge_drive_ghz_per_gain`、`flux_drive_phi0_per_gain`、`input_per_gain`、ADC gain/noise 等刻度。
5. 分別檢查截斷與 solver 收斂。Schema 驗證只檢查格式、數值與接線限制，不會替你判斷物理適用性或精度。
6. 修改 YAML 或環境後重啟 GUI／重建環境。`load_ftf` 只讀一次，不會 hot reload，也不會覆寫本機 `ftf.yaml` 或既有 entry。
