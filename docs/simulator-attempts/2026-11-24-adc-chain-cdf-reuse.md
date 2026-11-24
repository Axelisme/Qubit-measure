---
date: 2026-11-24
status: adopted
commit: 1d82a507d8024d1ed8961e2b3431c8d8b84ed144
modules:
  - lib/zcu_tools/simulate/adc
discussion: none
superseded_by: null
---

# ADC 跨 round 重用 chain CDF

**讀者：** framework-dev、agent。查找 ADC 在哪個 preparation 重用 CDF、抽樣驗收與兩個 chain 場景的耗時。

## 核心思路

[ADC raw path sampling](../../lib/zcu_tools/simulate/adc/_path_sampling.py)原先每輪對相同的 initial 和 transition 機率重算累積和。現在 [acquire 內的 raw owner](2026-11-22-adc-raw-plan-acquire-owner.md)在同一 active generation 對每個事件只算一次 CDF，跨 round 讀取。換 preparation 或結束 acquire 會退役原 generation。A→B→A 重綁會為 A 開新 generation，不沿用第一次的 CDF。逐列浮點加法、state 抽樣、事件間的 cavity CTMC 和最後的 receiver noise 順序不變。

不同事件仍各建一份 plan。把絕對時間改成相對時間會改變 carrier 與 demod phase 的浮點計算路徑，不能用代數上的時間平移等價代替逐值驗收。

## 實測

定位試驗在 `5dbccaffa` 上用下述 `chain_freq` 的完整輸入跑一次。Materialize 呼叫執行緒 CPU 共 134.530 s，其中 event plan 5.343 s、transition CDF 17.510 s、initial CDF 1.269 s、逐事件 IQ evaluate 58.345 s。Plan 與 transition CDF 合計 16.988%，扣除觀察器成本的保守估計為 16.889%，高於 5% 的原型門檻。這趟有量測 hook，不用來比較速度。

正式配對比較 `5dbccaffa0b8f29934ef0c6106971038459e44e3` → `1d82a507d8024d1ed8961e2b3431c8d8b84ed144`。兩側依[入庫清單](../../tools/data/experiment_wall_time/case-inputs.json)與同一離線 fixture，manifest SHA-256 `c255a7fb19c9598fa4cc248227d67b07520e1c8301824e480d3e1aa5e7c0b1bc`。每場每側各一次完整 Run，31 點 × 2,000 reps × 320 rounds，每點 16 reads，FTF seed 123。每對由同一排他 CPU slot 串行執行，單 worker、單原生執行緒、各自冷 cache，沒有 profiler 或觀察器。原始收據未入庫。

| `twotone/ro_optimize` 場景 | Materialize 秒，前 → 後 | Materialize 下降 | Core 秒，前 → 後 |
| --- | ---: | ---: | ---: |
| `chain_freq` | 139.611 → 115.223 | 17.47% | 154.649 → 128.696 |
| `chain_power` | 132.851 → 111.731 | 15.90% | 145.716 → 124.472 |

使用同一對 commit 和各場完整輸入另跑逐值比對。每場的 320 個 ordered ADC round 收據位元相同，涵蓋各事件的 state、取整前後 IQ、RNG 呼叫次序與 CTMC 前後的 PCG64 state。完整 Result snapshot 也相同。這四個帶觀察器的 Run 不參與計時比較。

## 為何採用

兩場 materialize 都超過 5% 的改善門檻，且沒有改變抽樣結果或亂數消耗。不把跨事件 plan 共用算入收益。

## 留下的線索

Materialize 仍是兩場最大分項。此次完整 31 點配對和[先前 11 點 flat-jump 配對](2026-11-24-adc-flat-jump-baselines.md#實測)的輸入不同，不能串接成累計改善。

- 浮點結果接近不能保證 ADC 整數逐值相同。差異可能跨過取整界線。若要接受近似，須另訂統計層級的契約；本次不改逐值契約，也不共用跨事件 plan。
- 若 `chain_freq` 改成直接產出讀出統計量，不再逐 shot materialize，應重新估計 CDF 重用還能省多少工作。
