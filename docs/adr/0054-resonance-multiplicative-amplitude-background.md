# ADR-0054：Resonance fitting 使用 multiplicative amplitude background

**狀態：** accepted

## Context

One-tone resonator fitting 需要吸收窄頻量測鏈中平滑的 magnitude baseline 變化，同時
維持 resonance circle 的物理解釋。既有 optional nuisance term 只旋轉 circle center 到
sample 的 vector，因此不是 cascade background，也會和 loaded-Q 的 circle phase 變化
競爭。`edelay` 已經完整表達 global linear phase，第二個 linear phase term 無法獨立識別。

Analyze params 會經 measure-gui remote get/update 投影，因此 fitting semantic 的替換也屬
public wire schema 變更，遵循 [[0013]] 的 remote adapter 邊界與版本握手。

## Decision

Hanger 與 transmission model 共用 real log-amplitude background：

\[
S(f)=\exp\left[g(f-f_r)\right]S_0(f)
     \exp\left[-i2\pi f\tau\right].
\]

- `g` 以 `bg_amp_slope` 暴露，單位為 MHz⁻¹。
- `a0` 是 background envelope 在 fitted `f_r` 的 complex scale。
- `edelay` 是唯一 global phase slope；`g` 不改變 phase。
- 關閉 optional fitting 時走既有 sequential circle/phase fit，並回傳
  `bg_amp_slope = 0`。
- 開啟時以 sequential result 初始化 bounded raw-complex joint refinement；Hanger
  refinement 擁有 `f_r/Q_l/|Q_c|/phi/a0/g`，transmission refinement 擁有
  `f_r/Q_l/a0/g`。caller 沒有固定 `edelay` 時，refinement 同時調整它。
- Refinement 後先從 raw samples 移除 final delay 與 amplitude envelope，再重算
  `circle_params`、circle phase 與 derived quality factors。
- IQ/circle/phase figure 使用 corrected domain；raw magnitude figure 同時顯示 total
  fit 與 `|a0| exp[g(f-f_r)]` background envelope。
- Analyze param 原子切換為 `fit_bg_amp_slope`，不提供舊 key alias 或 session migration。
  measure-gui `WIRE_VERSION` 與 `GUI_VERSION` 同步遞增，MCP code version 不變。

## Consequences

- Optional parameter 對應窄頻 scalar cascade 中的一階 magnitude response，且不再改變
  resonance phase law。
- Nonzero `g` 使 raw IQ locus 不再是 exact circle，因此 circle fit 只能作 initializer；
  final estimate 必須使用 complex samples。
- `g`、resonator parameters 與 unfixed delay 仍可能在窄 span 中相關；bounded optimizer
  failure、non-finite result 或 parameter-bound solution 會明確 warning 並回退 sequential
  result，不會靜默接受。
- Additive leakage、Fano path、phase curvature 與 complex background 不屬此模型；需要時
  另立具名 physical model，不把它們塞進 `bg_amp_slope`。

## Rejected alternatives

- **再加入 complex linear coefficient：** imaginary part 和 `edelay` 完全退化。
- **只對 circle vector 套用 slope：** 保持圓但不是 multiplicative background，且會直接
  改變 phase/Q semantics。
- **使用 additive complex line：** 可描述 leakage，但自由度與物理假設不同，不作為
  one-tone default nuisance term。
