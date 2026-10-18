---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-18-constant-period-builders.md
  - docs/discussions/2026-10-18-direct-integration.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@4390788a4、058963f1a：ramp handoff 與 phase-adaptive 研究基線"
  - "repo@b0cb487f3：physical carrier interpolation 研究基線"
---

# Shaped periods、gate preparation 與 carrier reuse

## 結論

這輪研究沒有找到通用的 reuse 次數門檻。gate-cache、ordinary period-band table 與 shared ramp spectral family 的 build 成本不同，guard 也可能已經產生可重用的 table。有效的比較必須把這些工作算進去。

採用的方向包括 phase-adaptive gate preparation、guard-built Sambe payload handoff，以及只對 narrow carrier group 做 physical-block interpolation。wide scan 的插值原型曾退步，最後採 exact bypass。

本文描述 main `f152eff67` 以前的實驗與採用理由。[evolution README](../../lib/zcu_tools/simulate/evolution/README.md)擁有算法與失敗語意，不從本文推導後續模組設計。

## 問題

長 ramp 的低 reuse 不代表 direct integration 一定便宜。短 gate 的 phase-table preparation 也可能比 apply 貴。早期固定 reuse≥10 或 reuse≤2 filter 把這些情況混在一起，錯過數值 guard 可接受的候選。

ordinary table 的 period-band reuse、gate 的完整 key reuse、Sambe 的 spectral-family reuse 不是同一個數量。把它們都叫 reuse 再套固定閾值，無法估 build amortization。

## 考慮過的選項與比較

### 從固定 reuse 改為實際成本交叉

早期研究比較 reuse=1、2、5、9、10、20，並納入 frequency、duration 與 real/complex envelope。reuse 分別按各 family 的 key 計算，不把 ordinary band 當成 Sambe spectral family。結果支持以 gate-key 與 period-band 分別估成本，不讀演化中的 density。ordinary table 只建實際用到的 port，fractional tail 改用 compiled strict runner，gate 增加 Strang guard。

這裡的成本排序沒有取代數值 certificate。通過適用條件只表示值得試，不表示可以忽略 harmonic、amplitude 或 dissipative split 誤差。

### Guard 的 table 不要再建一次

Sambe 提供低 reuse shaped rows 的另一條路。shared ramp family 讓多個 carriers 或 envelope rows 重用 spectral preparation，guard 收斂時把 payload 交給後續 prepare/apply，不再重算。

ramp handoff 實作 `058ed0f52`／`4a41594ff` 相對 `4390788a4` 的同 process、暖機後交錯 A/B，每次新 Engine、prepare cache budget=0、兩 workers，2001 個頻率的兩個 1 µs ramps 從 41.391 → 28.359 s，density bitwise 相等。原報告未列逐 pair foreign fraction，不稱它 isolated throughput。五個 flux × 2001 frequencies 的全部 ramps 仍要 163.403 s。當時 60 s 目標的分母是每 flux 的兩 ramps，不是五 flux 的完整 acquire，更不是整個弱 pulse 的 plateau 加 readout。

後來移除 shaped reuse≤2 filter，讓 real shaped candidates 先試數值 guard。34 個場景中九案 reroute 到 shaped Sambe。F6 routing 不變，新增 guard 為 0.00078 CPU-s，只占 acquire CPU 的 0.033%。當輪 wall pairs contended，因此這案的不退步證據是工作量，不是 clean wall 中位數。subdivisions>32 先拒絕，避免為已知不合適的 row 建 spectral data。

### Gate 的 phase nodes 由驗證決定

`058963f1a` 的 prototype 把 gate 每 key 的 16 個 phase integrations 改成六 nodes 加一個獨立 probe，保留原時間積分 tolerance。201 點、兩 workers、native threads=1、完整 target warmup 後清 cache，獨立 process interleaved pairs 中，最後一組 F9 Ramsey 省 31.28%，F10 echo 省 34.49%。另一組有效 pair 省 25.43%／33.25%。

這份 study 不改 F5 強 shaped Rabi。F5 unchanged control 的 6.56% wall 差不能算收益，也沒有找到 ≥20% 的 F5 候選。後來正式 phase-adaptive gate 用兩個獨立 probes，不能把 study 的六加一原型寫成最後機制。原 16-node builder 仍是保守 fallback。

### Narrow carrier 的 physical blocks 可以共用

carrier-reuse study 在 `b0cb487f3` 上試插值兩組資料，coarse/fine physical Fourier blocks 與 amplitude-Chebyshev/Fourier coefficients。它不插值 eigenvectors，也不做 eigenvector gauge alignment。

六個 Chebyshev-Gauss nodes 建多項式，三個獨立 probes 檢查 Frobenius-sum defect≤1e-7，失敗則二分。這是 calibrated interpolation，不是區間上的嚴格誤差界。

原型 F4 2001 點 acquire 是 39.476 → 29.546 s，retry 是 38.968 → 28.702 s。雖觀測省 25.15%／26.35%，兩組 foreign fraction 都超過 20%，只能記 inconclusive。

正式版先按 model、level、port、bound、subdivisions 分組。只有 same-sign、超過九個 unique carriers、端點 magnitude ratio≤1.1 的整組才插值。每 evolve 的 workspace 個別計費，不跨 acquire 留資料。

正式 clean pair 的 F4 是 28.12 → 19.56 s，foreign 6.1%／5.9%，eigensolves 從 12006 降至 54。physical holdout 差 3.9e-13。這是另一個版本的重新量測，不能拿前述 39.476 s 作它的分母。

## 被否決的選項與理由

- 全部 carrier scans 都插值。F15wide 原型 primary 退步 15.96%，retry 退步 9.29%。寬頻需要太多 nodes 與 probes，省不下 eigensolve，還增加同步與驗證成本。
- 正式 F15wide 因整組寬度不符直接 bypass，eigensolves 保持 9438，workspace=0 B。這不是「插值後沒有收益」，而是根本沒有為它啟用 workspace。
- 把 guard build 當免費再重建 table。這會隱藏 preparation 成本，與兩 ramps 的 build-once 證據相反。
- 從 F4 或 gate-cache 的收益外推全部 shaped workloads。F5、wide carrier 及完整弱 pulse 都提供反例。

## 留下的問題

ordinary period_table 的 adaptive counts 在 34 案重掃中因零 routing 停止，詳見 [constant builders](2026-10-18-constant-period-builders.md)。這不是證明該 fallback 無用。

截至本文基線，最寬 scan 的 shaped Sambe guard 仍很貴。這些結果沒有建立跨 evolve 或跨 acquire 的 workspace 契約。phase、carrier、amplitude probes 的 sampled defect 也沒有變成連續域上的 rigorous bound。

## 依據

本文整合 shaped crossover、ramp Sambe handoff、phase-adaptive gate、carrier-reuse study 與正式整合報告。原始工作檔未入庫，沒有在整理時重跑數值或速度。

F9／F10 是 Ramsey／echo，F5 是強 ramp Rabi，F4 是 narrow spectrum，F15wide 掃 50–7000 MHz。報告中的 labels 代表這些固定場景，不代表所有同名實驗都有相同收益。[Direct integration 研究](2026-10-18-direct-integration.md)記錄 F5 後來採用的另一條路。
