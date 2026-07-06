---
status: accepted
---

# ADR-0041 — autofluxdep feedback framework

關聯 [[0018]]（autofluxdep resolver-builder boundary）、[[0029]]（Fluxonium prediction engine）、[[0042]]（feedback confidence reversion）。

## 脈絡

`autofluxdep-gui` 的多個 node 需要跨 flux point 的自適應行為：`qubit_freq`
需要把 measured frequency 對 prediction 的 residual 用於下一點，`lenrabi`
需要讓 measured pi length 透過 drive gain 負回饋回到期望長度。若每個 node
各自保存 ad hoc state，GUI policy、runtime lifetime、測試與診斷會分散；若把
state 暴露成 Patch，又會把 node 私有控制細節變成 workflow dependency。

同時，ADR-0018 要求 orchestrator 只做 requirement resolution：它只看
`requires` / `provides` / `produce`，不理解 experiment-specific feedback。
Builder 是把 execution environment curry 進短生命週期 Node 的邊界；跨 flux
state 應位於 run-lived service/capability。

## 決策

1. **feedback framework 是 autofluxdep app-local deep module。** 通用 scalar
   estimator/controller 機制位於 `gui/app/autofluxdep/feedback/`。它不提升到
   shared session，也不放進 lower experiment module。

2. **feedback 是 run-lived capability，不是 Patch dependency。** Controller
   在 run start 依 enabled placed node 的 schema 建立 per-placement feedback
   slot map，放入 `Tools`，再由 orchestrator 以 provider name 取 placement-scoped
   view 放入 `RunEnv.feedback`。Node 在 `produce` / `make_cfg` 使用 capability；
   capability state 不經 Patch / Snapshot 流動。

3. **Builder 宣告 semantic slots。** Builder 以 node-owned declaration 宣告 slot
   key、kind、schema prefix 與 defaults；generic helper 預設可掛在
   `generation.feedback` group，但實際 node 會依領域語意選擇更清楚的
   generation group（例如 `generation.predictor_correction` 或
   `generation.pi_feedback`）。visible label 可只顯示 slot suffix（如
   `strategy` / `decay_points`），因為 group 已提供 semantic context；
   strategy selector 以 `ChoiceSectionSpec` 只顯示目前 strategy 需要的
   hyperparameters；persisted raw 仍是 flat logical generation keys。同一 Builder
   可被放置多次，runtime key 使用 placed-node name，因此每個 placement 有獨立
   hyperparameters 與 state。

4. **generic feedback 只提供抽象 scalar mechanics。** Estimator strategy 是
   `off` / `idw` / `last_good`；controller strategy 是 `off` / `log_step`。
   `off` 表示該 slot disabled，runtime lookup 回傳 `None`。generic layer 回傳
   `FeedbackSample(value, confidence, age_queries)`，只做 finite/positive 等自身
   數學前置檢查與 age-based confidence decay，不處理 bounds、clamp、saturation、
   max-step、stop/fail、fit-quality gate、fallback target 或 acceptance policy。

5. **disabled 與 undeclared 語意明確。** 已宣告但 `strategy=off` 的 slot lookup
   回傳 `None`，由 use site 決定 fallback；未宣告 slot lookup fast-fail，表示
   node 與 Builder declaration 不一致。

6. **no-observation 不產生 generic fallback。** estimator 沒有 observation 時回傳
   `None`，generic layer 不回傳 `0`、seed 或 default correction。use site 決定
   是用 seed、no correction、skip，或 fast-fail。

7. **node 擁有 domain composition。** feedback capability 不知道被控制/預測的
   domain 值。`qubit_freq` 決定 `base predictor prediction + correction` 如何組成
   drive center、`fit_detune` 與 result role；fixed-bias mode 保持 base predictor
   不變，hard-bias mode 才要求 predictor backend 校準。`lenrabi` 決定 pi-length
   normalized error 如何轉成 gain proposal 並在何處 clamp。

8. **predictor 不隱藏 residual correction。** `Tools.Predictor` 只提供 physical/base
   prediction 與 backend-supported calibration。`qubit_freq` 的 residual
   interpolation 是 feedback estimator slot，不再是 `SimplePredictor` /
   `FluxoniumPredictorAdapter` 的 private IDW。

## 後果

- Orchestrator 仍是純 requirement resolver，只增加 `RunEnv.feedback` pass-through，
  不知道 slot key 或 experiment semantics。
- GUI 可用同一個 generation section 控制不同 slot 的 hyperparameters，不需要
  第三個 root-level config block。
- `qubit_freq` 的 stored `predict_freq` result role 表示實際用來中心化 detune
  sweep 的 composed prediction；predictor service output 是 base prediction。
- `lenrabi` 的 max drive gain clamp 與 fit gate 留在 node；generic controller
  只提出 raw proposal。
- 後續 node 若需要 scalar estimator/controller，新增 Builder slot declaration 與
  use-site composition 即可，不必改 orchestrator。

## 拒絕的替代方案

- **把每個 feedback 值作為 Patch key 暴露。** 這會把 node 私有控制 state 變成
  workflow public dependency，增加下游認知負擔，也破壞「Patch 是供其他 node
  消費的 domain output」邊界。
- **讓 feedback layer 處理 bounds/clamp/safety。** 不同實驗對 clamp、fail、stop
  與 acceptance 的語意不同；放進 generic layer 會讓抽象層知道 domain policy。
- **繼續把 residual IDW 藏在 predictor adapter。** 這會讓 `qubit_freq` 無法清楚
  控制 base prediction 與 correction 的組合，也讓其他 scalar correction 不能
  重用同一套 policy/schema/runtime。
- **在 orchestrator 中特殊處理 feedback。** 這違反 ADR-0018，會讓 requirement
  resolver 重新理解 experiment semantics。
