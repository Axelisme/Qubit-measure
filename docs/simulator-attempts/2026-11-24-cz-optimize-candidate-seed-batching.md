---
date: 2026-11-24
status: paused
commit: e060dae1f
modules:
  - zcu_lab/v2/two_qubit/cz_optimize
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# cz_optimize 的候選與 seed 合批或平行

**讀者：** framework-dev、agent。記錄候選與 seed 的工作分組，以及單 worker 基準量到哪裡。

## 核心思路

[`CZOptimizeExp.run`](../../zcu_lab/v2/two_qubit/cz_optimize/core.py) 按候選、再按 seed 串行評分。每個候選只跑 interleaved RB arm，每個 seed 各建一個 Program 並執行一次 acquire。候選與 seed 都不在 point axis。CMA 每代先取出整個 population，收齊分數後才更新，因而可考慮把同一代的候選與 seed 放進同一批，或平行派送獨立 acquire。

## 實測

單次串行 Run 用 `e060dae1f`、164 ns arb CZ 的 path-timing fixture、depth 2、seeds 701／709／719、reps 256、1 代、popsize 2 與單 worker。這組輸入用來計時，不是物理校準。原始工作檔未入庫。

- 第一候選用原 pulse，三個 seed 的 acquire 實測 wall 合計 316.373 s。暖機後完整 acquire 約 66–80 s／seed，包含下一候選唯一完成的 seed（同一 `e060dae1f` 與上述輸入）。
- 第一候選三個 seed 的 evolve 占 core wall 約 96%，即 302.998／316.081 s。`dop853` 占 family CPU 約 77%，即 219.910／284.637 CPU s（同一 `e060dae1f` 與上述輸入，三次完整 acquire 的實測總和；family CPU 不是 wall）。
- 單次 Run 觸及 450 s timeout 時，規劃的 9 次 acquire 有 5 次留下 telemetry，其中 4 次完整、1 次未完成（同一 `e060dae1f` 與上述輸入）。沒有完整 Run 耗時。

## 為何採用或否決

暫停。局部計時能指出 evolve 的成本，不能當成合批或多 worker 的配對基準。合批需要 core 在同一 Program 內表達各候選的 arb 與 local Z，以及各 seed 的 Clifford 序列。目前沒有這個介面。Engine 的既有 worker 只分派同一次 evolve 內的多列工作，不會平行派送這些分開的 acquire。

## 留下的線索

先取得相同輸入的完整單 worker Run，再用同一份輸入比較合批或多 worker。分開記錄 Run wall 與 family CPU，避免把單一 family 的 CPU 比例當作整輪可平行的比例。
