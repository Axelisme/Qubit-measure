---
date: 2026-11-24
status: adopted
commit: 5e43af2c76bf40e6b2c314ee79af7362f07cd590
modules:
  - lib/zcu_tools/simulate/adc
discussion: ../discussions/2026-11-24-adc-raw-flat-jump-exception.md
superseded_by: null
---

# ADC flat jump 只算兩態基線

**讀者：** framework-dev、agent。查找 ADC raw materialize 在平坦讀出跳轉時省掉的工作與實測範圍。

## 核心思路

每個跳轉的腔記憶修正只讀來源態與目的態。原路徑在每個跳時先算所有狀態的基線場，再取這兩項。當請求不回傳 trace、所有 pulse 是 flat、狀態多於兩態且當下有 jump 時，[ADC raw materialize](../../lib/zcu_tools/simulate/adc/_integrated_paths.py) 只算這兩態。每態公式、carrier 加總順序與腔記憶更新不變。其他情形仍走原路徑，不改 trace、window 或 [raw plan 的 acquire 壽命](./2026-11-22-adc-raw-plan-acquire-owner.md)。

## 實測

先在 `4a1d3228d` 用 `py-spy record` 分別取樣 `chain_freq`、`chain_power` 的一次完整 11 格 Run。使用下述同一場景清單及離線設定。既有 telemetry 顯示 materialize 是兩場最大分項；取樣在該階段看到 `_baseline`、`evaluate` 與 `_event_states`。這輪只定位，不作正式速度比較。[腔場乘積積分的時域三列](./2026-11-24-adc-cavity-product-integrals.md#實測)另有到期堆疊及正式前後配對，但沒有完整函式取樣。

正式配對比較 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a` 與 `5e43af2c76bf40e6b2c314ee79af7362f07cd590`。從[實驗 Run 牆鐘](../../tools/README.md#實驗-run-牆鐘)的 `ftf-demo/initial` 離線 fixture 開表，`chain_freq` 只覆寫頻率點數，`chain_power` 只覆寫增益點數，均設 11 格。四次 Run 串行完成，FTF seed 123、`time_scale=0`、同 CPU slot、單 worker、每次冷 cache，無 profiler。場景清單與原始收據未入庫，清單 SHA-256 為 `931fa4f599f4db8dc82ac9363cb1068ac6f94ce9286e6e1baec34545bce57a1f`。

| `twotone/ro_optimize` 場景 | Materialize 秒，前 → 後 | Core 秒，前 → 後 | Core 下降 |
| --- | ---: | ---: | ---: |
| `chain_freq` | 61.162 → 48.688 | 67.716 → 55.282 | 18.36% |
| `chain_power` | 60.791 → 48.478 | 67.333 → 54.952 | 18.39% |

各欄是單次完整 11 格 Run 的總和。Core 包含 compile、evolve、readout 和 materialize。兩對 Result 與 shot moments 的最大絕對差為 0，掃描軸與 shot counts 不變。

## 為何採用

兩個場景都通過採用門檻：core 至少下降 5%，且 materialize 同向下降。同一對 commit 另用公開 ADC raw 接縫比對合成多態純 flat 輸入。兩輪實際跳轉命中兩態路徑，固定 identity 的 256 個 int64 值逐個相同。這次試驗沒有加入快取、改變抽樣或將統計量改成直算。

## 留下的線索

候選的兩場 core 仍約 5 s／格，距 0.1 s／格的目標還遠。Materialize 仍佔兩場的大半。未量 `rpi_freq`、`rpi_length` 或 `auto`；它們只有落到相同的 flat 跳轉條件才可能受惠，不能用這兩對 Run 推估收益。
