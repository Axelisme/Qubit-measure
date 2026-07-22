# ADR-0056：Resonance fitting 使用 optional phase curvature 與 internal rational initializer

**狀態：** accepted

## Context

ADR-0054 讓 one-tone resonance fitting 使用乘法式 real log-amplitude
background，並保留 `edelay` 作為唯一 global linear phase slope。ADR-0055 則讓
absolute electrical-delay branch 由 route-scoped `auto` / `calibrated` /
`manual` seed 與 bounded local refinement 擁有。

實測 homophasal trace 顯示，只有 amplitude slope 時仍可能留下平滑的 edge phase
residual；同時，single-pole ABCD rational fit 在移除正確 route delay 後能提供有用的
initial trace，但 upstream automatic delay estimator 會在長 delay / nonuniform grid 上
選錯 branch。因此 phase background 與 rational initializer 需要明確邊界，不能引入第二套
delay ownership。

## Decision

Hanger 與 transmission resonance model 支援兩個彼此獨立的 optional background terms：

\[
S(f)=\exp[g(f-f_r)]\,\exp[i c(f-f_r)^2]\,S_0(f)
     \exp[-i2\pi f\tau].
\]

- `bg_amp_slope = g`，單位為 MHz⁻¹，語意沿用 [[0054]]。
- `bg_phase_curvature = c`，單位為 rad/MHz²。Quadratic phase 以 fitted
  resonance frequency `f_r` 為中心，因此在 `f_r` 的 phase value 與一階 slope 都為零。
  Constant phase 留給 `a0`，linear phase 留給 `edelay`。
- `fit_bg_amp_slope` 與 `fit_bg_phase_curvature` 是獨立 analyze options。四種組合都有效；
  未啟用的 term 固定為 `0.0`，且不加入 optimizer parameter vector。
- Core experiment 與 fake adapter 的 phase-curvature default 是 `False`。Real one-tone GUI
  維持 amplitude background default on，但 phase curvature default off。
- Optimizer 可用無因次內部尺度表示 curvature，但 public params 與 result dict 永遠回傳
  physical `bg_phase_curvature`。
- Optional background 只在 raw-complex joint refinement path 啟用。兩個 option 都關閉時，
  sequential circle/phase path 的行為維持不變。

Internal rational initializer 只重寫 degree-1 complex rational core，用於已經依 ADR-0055
移除 route-owned electrical delay 的 corrected trace：

- 不新增 `abcd_rf_fit` runtime dependency，也不 vendor upstream package。
- 不移植 upstream delay estimator、unbounded final least-squares、plot layer、container
  model 或 vector-fit path。
- Frequency 與 signal 先做 scale-safe normalization；nonuniform grid 的 derivative /
  weighting 必須使用實際 frequency spacing。
- Rational result 只作 initializer 或 denoised single-pole trace。病態、非 finite、pole /
  denominator invalid 或 residual 不可信時發出 `RuntimeWarning`，並回退既有 sequential
  initializer。
- Absolute delay branch 仍完全由 [[0055]] 的 route-scoped seed 與 bounded refinement 擁有。
  Rational initializer 不得自行擴張、重估或替換 branch。

Hanger complex fit validity 與 derived internal quality factor 分離：

- Complex fit acceptance 只由 optimizer success、finite result、active bound 與核心 physical
  parameters 決定。
- Derived inverse internal loss non-positive 時，fit 可以保留；result 回傳 `Qi=None` 與
  `qi_status="model_incompatible"`。
- Positive derived loss 回傳 finite `Qi` 與 `qi_status="physical"`。
- Plot 與 callers 必須處理 optional `Qi`，不能把 `None` 格式化成數字。

Plot contract 延續 [[0054]]：

- IQ / circle / phase 使用移除 enabled background terms 與 delay 後的 corrected domain。
- Magnitude envelope 只在 amplitude background enabled 時顯示。
- Phase curvature 只在 `fit_bg_phase_curvature=True` 時出現在 annotation；關閉時不顯示
  fixed-zero line、text 或 legend entry。

新增 analyze option 屬 public remote schema 變更，measure-gui main `WIRE_VERSION` 與
`GUI_VERSION` 同步 bump；不提供 legacy key alias 或 session migration。

## Consequences

- Phase curvature 表達平滑 nonlinear phase background，而不和 `edelay` 的 global linear
  slope 退化。
- Default behavior 保守；只有使用者或 GUI analyze params 明確打開 phase curvature 時才增加
  自由度。
- ABCD-style rational fit 提供初值品質與速度，但 final result 仍由本 repo 的 bounded
  raw-complex model、background options 與 delay contract 管理。
- Negative或不可解釋的 hanger `Qi` 成為 diagnostic，而不是把更好的 complex fit 整體丟棄的
  rejection reason。

## Rejected alternatives

- **把 curvature 寫成另一個 linear phase term：** 和 `edelay` 不可識別，違反 [[0054]]。
- **直接依賴 `abcd_rf_fit`：** 上游 package 的 delay estimator、bounds、license metadata 與
  final optimizer contract 不符合 measure GUI 的 production requirements。
- **讓 rational initializer 估 absolute delay：** 會和 [[0055]] 的 route-scoped branch
  ownership 形成第二套來源。
- **在 disabled option 仍保留 dummy optimizer parameter：** 會讓四組 option 的 model
  selection 與 bound behavior 不可解釋。
- **因 `Qi<=0` 回退整個 complex fit：** 混淆了 resonator primitive 的 fit validity 與
  internal-loss physical interpretation。
