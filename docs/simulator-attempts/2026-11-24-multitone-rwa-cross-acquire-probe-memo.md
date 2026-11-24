---
date: 2026-11-24
status: rejected
commit: e060dae1f
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# 跨 acquisition 重用 multitone RWA 探針

**讀者：** framework-dev、agent。記錄跨 acquisition 擴大 sampled-probe memo 的命中與收益上界。

## 核心思路

[既有 memo](./2026-11-23-ftf-rwa-probe-memo.md) 只在同一 preparation 內重用探針結果。這次想延長 memo 的生命週期，讓不同 acquisition 遇到相同的 sampled-probe exact key 時也能跳過積分。使用者於 2026-11-24 回覆「只要能加速，不論手段」，允許保留這類快取。快取仍只接受完全相同的物理輸入，容量須有上限，生命週期須寫明。

## 實測

量測以 `e060dae1f` 為基底，加上未入庫的量測用 trace hook。Hook 記下探針的 exact key、budget、route row、acquire／evolve／memo 識別碼、查找結果與 guard 判定。另用臨時計時器記每次積分的執行緒 CPU 耗時。原始工作檔未入庫。

- `two_qubit/rb` 用 const CZ、seeds 0／1／2、depths 0／1／2／4 與單 worker。每臂實測 14 次探針請求，只有 3 次跨 acquire 重複（`e060dae1f` 加 trace hook，各臂各一輪）。另一次單臂計時的 core 是 80.038 s；若跳過這 3 次積分，最多省 6.278 CPU s，推算 core 耗時的加速上界是 1.085×（相同基底與輸入，零查找與維護成本，未做快取原型）。
- 把兩臂設成相同物理輸入，假設 memo 跨臂保留，且第二臂的所有探針都命中，core 耗時的加速上界是 1.250×（`e060dae1f` 加 trace hook，沿用上述三 seed RB，以兩臂總耗時外推）。實際 reference 與 interleaved 兩臂不保證有相同的 key。
- `two_qubit/cz_amplitude` 在單次 acquire 內跨 preparation 多命中兩次時，core 耗時的加速上界是 1.108×（`e060dae1f` 加 trace hook，const CZ、11 個 scale 點、單 worker，一輪計時的零成本外推）。
- [`cz_optimize` 的 arb CZ 診斷](./2026-11-24-cz-optimize-candidate-seed-batching.md) 中，`multitone_rwa` 只占 0.004／284.637 CPU s（`e060dae1f`，164 ns arb CZ、depth 2、seeds 701／709／719、reps 256、單 worker，第一候選三次 acquire 的 family CPU 實測總和）。

## 為何採用或否決

RB 與 `cz_amplitude` 的外推上界都低於這次衝刺的落地門檻 1.3×。arb CZ 的 `cz_optimize` 幾乎不花 CPU 在該 family，無法靠這個 memo 省下可觀時間。不做原型，也沒有配對計時或 density 一致性驗證。

## 留下的線索

延長 memo 壽命前，先按 exact key 和 acquire ID 數新增命中，再單獨計時能跳過的積分。Prepare 的總成本包含不重複的探針，不能全算成快取收益。
