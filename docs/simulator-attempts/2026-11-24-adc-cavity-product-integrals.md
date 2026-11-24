---
date: 2026-11-24
status: adopted
commit: 7825137c6169ce3e78e6564ac9b021883686a059
modules:
  - lib/zcu_tools/simulate/adc
discussion: none
superseded_by: null
---

# ADC 共用腔場乘積積分

**讀者：** framework-dev、agent。查找 chain 讀出共用腔場積分的收益與限制。

## 核心思路

原路徑對每對狀態的乘積生成元各算一次增廣矩陣指數。新路徑對小型平坦區間先計算各態腔場的短步 Taylor 係數，再一次積分所有腔場乘積。把同一段的積分與各態傳播子一起倍增，直到區間終點。不符合規模或數值守衛時仍走原路徑。改動在 [ADC 積分器](../../lib/zcu_tools/simulate/adc/_quadratic.py)，沒有改變演化或 FTF 選路。

## 實測

先在 `4a1d3228d` 用 `py-spy record` 取樣 `chain/len_rabi` 的一次完整 11 格 Run，從既有 telemetry 讀 core 分項。使用下述同一場景清單及離線設定。取樣在 readout 看到 `field_product_integral`，整體最大分項是 evolve。`t1`、`t2echo`、`t2ramsey` 沒有完整函式取樣。舊普查 `593426c2f` 的同設定三列各 Run 一次，180 s 到期堆疊都落在這個積分函式。這只說明到期瞬間，不能判定三列各自最大的 readout 函式。

正式配對比較 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a` 與 `2a571d68297db69c9419099c478e60fc5aa5d362`。使用[實驗 Run 牆鐘](../../tools/README.md#實驗-run-牆鐘)的離線 fixture，從 `ftf-demo/initial` 開表。FTF seed 為 123，`time_scale=0`，每對使用同 CPU slot、單 worker、冷 cache，沒有 profiler。Scan 一律覆寫為 11 格。Echo 的等待終點設為 465 µs，Ramsey 為 180 µs、detune 為 0.027777777777777776 MHz。其餘欄位沿用 Form 與 fixture。當時的場景清單與原始收據未入庫，清單 SHA-256 為 `931fa4f599f4db8dc82ac9363cb1068ac6f94ce9286e6e1baec34545bce57a1f`。

下表每個版本各一次完整 11 格 Run。Core 秒數加總 compile、evolve、readout、materialize。

| 場景 | Readout 秒，前 → 後 | Core 秒，前 → 後 |
| --- | ---: | ---: |
| `chain/len_rabi` | 20.772 → 8.279 | 106.060 → 93.163 |
| `chain/t1` | 25.379 → 10.914 | 122.660 → 107.872 |
| `chain/t2echo` | 60.675 → 27.216 | 192.778 → 158.262 |
| `chain/t2ramsey` | 46.503 → 19.446 | 135.638 → 108.336 |
| `chain/amp_rabi` | 6.478 → 4.768 | 67.767 → 66.059 |

`twotone/chain` 用同一對 commit 與設定，掃 253.812–341.672 MHz，各派一次 11 格 Run。兩側都到 180 s wall deadline，沒有完整 core 或 Result。當下的堆疊分別落在 `_rwa_numeric._resolve_retention` 和 Floquet period 的 `_build`。兩處都在演化準備，不能據此估計各步成本或判斷加速。

## 為何採用

上表五對 Run 的 Result 與 shot moments 最大絕對差都是 0，掃描軸與 shot counts 不變。四個原定完成場景的 readout 都下降。補測的 `amp_rabi` 也下降，但 core 只降 2.5%。這是相同輸入的回歸比較，不是獨立物理精度驗證。

## 留下的線索

表內五場的最大分項仍是 evolve，候選的 core 均未達 0.1 s／格。此法減少小型平坦區間的重算，和[先重用相同 projective span](./2026-11-21-projective-span-reuse.md)不同。它不處理 `twotone/chain` 的演化準備。
