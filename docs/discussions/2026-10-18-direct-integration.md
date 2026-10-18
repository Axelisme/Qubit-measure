---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-18-shaped-period-and-gates.md
  - docs/discussions/2026-10-18-two-tone-floquet.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@6aad06a85：sample-spacing controls 實作"
  - "repo@b0cb487f3：strong-ramp interaction-picture study 基線"
---

# Direct integration 的 RHS、step controls 與 interaction picture

## 結論

Direct path 的收益來自減少每次 RHS 工作、移除不必要的固定 step ceiling，以及在適合的單 tone rows 轉到 H0 interaction picture。這些方法都保留 adaptive controller，不靠放寬全域精度宣稱加速。

研究中的 gain grouping 最後撤掉，因為它讓結果依 apply-call composition 與 worker count 改變。正式採用的是每 row 獨立的 interaction-picture selection。

本文記錄 main `f152eff67` 以前的證據，不描述後續整理的模組分工。[evolution README](../../lib/zcu_tools/simulate/evolution/README.md)擁有 solver 契約。

## 問題

294 MHz 工作點保留的 16 levels 有約 12.55 GHz energy span。carrier 低，lab-frame explicit RK 仍可能受高能階相位的 stability 限制。population error 很小，也不代表可以把 step 任意放大。

F5 強 ramp Rabi 的原 batching 逐 row 積分，每 row 重設 adaptive history。batch 只省 Python 呼叫，不共用 Runge–Kutta stages，所以不能說是最強 row 拖慢整批。

## 考慮過的選項與比較

### Hermitian RHS 只做一次 Hρ

Hermitian RHS 實作 `9eeb1659a` 利用 H 與 ρ 的對稱性，由 Hρ 及其 conjugate transpose 組成 commutator，不必再算第二次 ρH。這項代數重排在六個實際 rows 的單核 CPU 中省 17.9–21.3%。Engine 比較使用兩 workers，density 差小於 2.3e-13。

這些都是同 process 暖機後 A/B，Engine 使用新的空 prepare cache。計時截止於 readout start，不包含 ADC 或 noisy acquire；也不是 flux RHS 的效能結論。沒有改 tolerance。原報告未列這組 CPU 比較的逐 pair foreign fraction，不將它稱作 isolated throughput。

### Ceiling 由 active sample spacing 決定

舊 fixed 0.1 ns ceiling 改成所有 active envelopes 的最小 sample spacing。continuous-flux path 也納入 flux sample spacing。constant envelopes 不增加 envelope ceiling。每 row 仍以 0.1 ns deterministic seed 起步，並受 spacing 限制；seed 不是最大 accepted step。

研究保留原 samples 與 linear playback，row boundaries 精確停住，但不要求每個內部 knot 都停步。joint tolerance scale 選 s=0.03：

- rtol=min(1e-4, s×budget)。
- atol=min(1e-6, 0.01×s×budget)。

budget=1e-8 的既有三 tone isolated-feature assertion 決定這個係數。s=0.04、0.05、0.1 都不通過。這是 empirical local controls，不是 global error bound。

sample-spacing 改動的最後 scenario pairs 比較 `d099d4893` 的舊 core 與整合版 `2ec7ef123`。兩 workers、同 process A/B、暖機後空 prepare cache，沒有顯示超出量測波動的退步。這組摘要沒有逐 pair foreign fraction。S6 wall 中位數增加 0.31%，CPU 下降 1.05%，wall ranges 重疊。294 MHz 長段仍 stability-limited，沒有因 tolerance 放鬆就解決。

### Interaction picture 減少需要積分的 H0 相位

strong-ramp study 在 `b0cb487f3` 上用 σ=exp(+i2πH0t)ρexp(−i2πH0t)，解析處理每 level 的 H0 phases。IP 加 gain grouping 的 F5 201 點、兩 worker acquire 從 4.679 → 3.205 s，省 31.50%，foreign 5.53%／6.99%。21 點 strict population 差 2.13e-5，第一個 Rabi 峰在同一 gain 格點。

這是一組 clean study pair，不是 production admission。單獨 grouping 只省 2.12%。F15wide 只量 prototype，沒有 matched baseline，所以沒有宣稱其收益。

正式版撤掉 grouping。只有 single-tone、nonzero gain、且不是同一 apply call 中 touching chain 一部分的 rows 走 IP。每 row 保留原 tableau、tolerances、0.1 ns first step 與 active-knot ceiling。其他 rows 保留 lab runner，continuous flux path 不變。

F5 201 點正式三組 clean pairs 的版本中位數為 5.056 → 3.768 s，比 0.745。F15wide 2001 點比為 0.922。其他 DOP853 場景符合當時 ≤1.02 的不退步門檻，34 案 routing 不變。這些是新一輪 matched 比較，不能把 study 的 4.679 s 接成正式版的分母。

## 被否決的選項與理由

- gain grouping 改變一個 apply call 放哪些 rows，就可能改變 trajectory、composition 或結果。worker 切法也會影響結果，因此不保留。
- 把 sampled knot compression API 留作相容層。研究已改成保留原 samples 的 spacing control，未採用的 compression API、logic 與專用測試一起移除。
- 將 s 提高到 0.04 以上。既有三 tone assertion 不通過，不能只看多數 cases 的寬裕誤差。
- 把所有 rows 都改成 IP。touching chain、多 tone、zero gain 或 flux rows 沒有由這份研究取得同等驗證，lab 與 chain 路線保留。
- 把 H0-IP period-builder 原型當成這次 direct-IP 的收益。前者在低頻 CF4 study 中沒有成為主要收益，問題與控制不同。

## 留下的問題

正式 IP 的 warm-disk first-use ratios 都≤1.02，但 cold-cache pairs 因 foreign 差超過五個百分點而 inconclusive。新 JIT entry points 在 13 個 DOP853 場景各只有一個 signature，這是編譯機制證據，不是冷啟動 wall 已過的替代宣稱。

另一輪 production worker validation，用 294 MHz、21 點 flattop sweep 比較 workers=1／6。同 process、兩次 full-target warmup、新 Engine，evolve 中位數為 18.904／7.569 s，2.50 倍，density bitwise、paths 相同。host load 約 6.13–7.64，這是 contended-host 結果，不能當 isolated scaling curve，也不是 IP 的新增收益。

## 依據

本文整合 Hermitian RHS、sample-spacing controls、strong-ramp study、正式 interaction-picture 與 worker-validation 報告。原始工作檔未入庫。單 row CPU、Engine evolve 與完整 acquire 的邊界分別保留。

[Shaped preparation](2026-10-18-shaped-period-and-gates.md)記錄 F5 在 phase-table 方向沒有找到收益的結果；[two-tone routing](2026-10-18-two-tone-floquet.md)說明 direct solver 後來何時比 Floquet 更便宜。
