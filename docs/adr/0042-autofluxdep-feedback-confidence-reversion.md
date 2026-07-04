---
status: accepted
---

# ADR-0042 — autofluxdep feedback confidence reversion

關聯 [[0041]]（autofluxdep feedback framework）、[[0018]]（autofluxdep resolver-builder boundary）。

## 脈絡

autofluxdep 掃 flux 時常會經過一段 SNR 很差、fit 不可信或沒有 fit 的區域。
`qubit_freq` 的 prediction correction 若在這段區域永遠持有上一個 residual，會把
drive center 鎖在過時修正上；`lenrabi` 的 drive gain controller 若永遠持有上一個
proposal，會在沒有新回饋時持續沿用可能已不適合當前 flux 的 gain。相反地，硬切
stale threshold 會讓行為在 cutoff 點不連續，容易造成下一點設定跳變。

ADR-0041 已決定 generic feedback 不知道 domain composition、fallback、clamp 或
fit-quality gate。平滑 stale 行為需要保留這個邊界：generic layer 可以描述「這個
feedback sample 有多新」，但不能決定 `qubit_freq` 要退回 base predictor、`lenrabi`
要退回 open-loop `pi_product` gain，或任何實驗要如何處理 bounds/safety。

## 決策

1. **generic feedback 回傳 sample，不只回傳 scalar。** Estimator `estimate()`、
   controller `latest()` 與 `propose()` 回傳 `FeedbackSample(value, confidence,
   age_points)`；`value` 是 strategy 的 raw scalar output，`confidence` 是 0..1
   的抽象新鮮度，`age_points` 是從最近一次可信 observation/proposal 後被查詢的點數。

2. **confidence 依 query age 平滑衰減。** Slot schema 暴露
   `<prefix>_decay_points`，generic runtime 以 `exp(-age_points / decay_points)`
   產生 confidence。`decay_points` 只描述 service sample 的新鮮度半徑，不是 domain
   bound、clamp 或 fit gate。

3. **無 observation 仍是 `None`。** Estimator 沒有任何可信 observation 時回傳
   `None`；disabled slot 也回傳 `None`。generic layer 不用 confidence=0 的 sample
   假裝存在 seed/default，避免把 use-site fallback 藏進 service。

4. **node 擁有退回目標與組合方式。** `qubit_freq` 把 correction 用
   `confidence * correction` 加到 base predictor prediction 上，使 long-stale
   correction 平滑退回 0。`lenrabi` 把 controller proposal 與 open-loop
   `pi_product` gain 在 log-domain 依 confidence 混合，使 long-stale proposal 平滑
   退回 open-loop gain。bounds 與 max gain clamp 仍在 node use site。

5. **Builder 擁有每個 slot 的預設超參數。** 不同被預測/被控制的值用不同 Builder
   slot declaration 設定 `default_decay_points`，GUI 仍透過 placed-node 的
   Generation overrides 控制每個 placement 的超參數。

## 後果

- generic feedback 仍是抽象 scalar capability；它新增 confidence metadata，但不承擔
  domain fallback、bounds 或 acceptance responsibility。
- `qubit_freq` 在長時間低 SNR 區域會逐步回到 base predictor，而不是硬切或無限沿用
  過時 correction。
- `lenrabi` 在長時間沒有可信 pi-length 回饋時會逐步回到 open-loop `pi_product` gain，
  減少 stale controller proposal 造成的 drift。
- 沒有引入 PID integral state，因此不會在低 SNR / no-feedback 區域累積 windup。

## 拒絕的替代方案

- **硬 stale cutoff。** 實作簡單，但 cutoff 點會造成不連續跳變；對 flux-dependent
  掃描不如平滑退回穩定。
- **在 generic feedback layer 內建 fallback target。** 這會讓 service 知道
  `base prediction`、`open-loop gain` 等 domain concept，破壞 ADR-0041 的責任分離。
- **預設 PID integral controller。** PID 適合固定 setpoint 的 online actuator，但在
  flux-dependent、長時間 no-feedback 的掃描中，integral state 容易累積過時誤差；
  目前只保留後端接口可擴充，不作為預設。
