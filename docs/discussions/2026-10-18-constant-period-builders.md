---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-18-dyson-input-reuse.md
  - docs/discussions/2026-10-18-shaped-period-and-gates.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@058963f1a、b0cb487f3、12f9e96a3：Padé、continuation 與 Sambe 研究基線"
  - "repo@956c86ddb：constant Sambe 正式比較基線"
---

# Constant-tone period builders 的演進與重用限制

## 結論

低 carrier 的 period preparation 先從 SciPy callback 搬到 compiled DOP853，再改成 adaptive CF4，之後採 Padé7、gain anchors 與按 key 選擇的 Sambe builder。它們改善的是不同成本，不能把各輪降幅相乘。

carrier continuation 的 1103 次嘗試全部失敗。gain continuation 有效，但 worker-dependent trajectories 與較密 anchors 曾使 CPU 或 RSS 超標。這些失敗限制了後來的 workspace 設計。

本文記錄 main `f152eff67` 以前的研究，不描述整理之後的模組切法。[evolution README](../../lib/zcu_tools/simulate/evolution/README.md)擁有數值契約。

## 問題

carrier 很低，不代表 Hamiltonian 的 spectral width 很小。50 MHz 模型保留 16 levels，H0 width 約 7.99 GHz，period builder 仍須解析高能階相位。

本文的 L7strong 是 49.991 MHz 模型的強驅動頻譜，L8 是同模型的 Rabi gain sweep。早期低頻 L1–L6 是另一份模型，其實算 transition 為 581.742664 MHz，不是刻意調成 400 MHz。完整 acquire 成功且 exceedances 為空，分類只占 0.004–0.48%。強 T6 與 shaped build 才是重點，這份 baseline 不是 A/B 加速證據。L1–L3／L5 用 `dcdc02353`，L4／L6 用 `a5ffa6135`，每場景只量一次、workers=2，原摘要沒有逐 run contention 標記。

`a5ffa6135` 上的 50 MHz coverage audit 更直接暴露問題。單次 T6 prepare 要 99.170 s，direct 要 1.257 s，population 差為 1.63e-7。另一個 weak spectrum 雖只差 2.34e-9，卻抹平譜線，peak 差 5.72 grid steps。absolute population error 小，不等於物理 feature 通過。singleton wall 是單 row 診斷，不是完整 grid 的 interleaved speed gate，也未附逐 pair foreign fraction。

## 考慮過的選項與比較

### DOP853、Strang 與 CF4

compiled DOP853 實作 `1980cca65`／`bf56fb006` 改用 native dense polynomial 並融合 stage buffers。L2 prepare thread CPU 從 SciPy 的 18.045 s 降為 5.968 s，再降為 3.960 s。這是同 process 交錯、兩 workers、native threads=1 的分階段 prepare 診斷，不是完整 acquire speedup。原報告未為這三個 CPU 值列逐 pair foreign fraction。當時低頻門檻為 0.3 GHz，低頻路徑後來又由 CF4 取代。

Strang 實作 `cda550a6c` 試過 0.03 GHz 門檻與 m≤4 的分段。L7、L8 避開 T6，但完整尺寸只外推為 165／25.5 s，仍未達 60／5 s。外推不是 PASS，也沒有另加 T6/DOP 成本分支來掩蓋結果。

adaptive CF4 實作 `8babdf75e` 把 m 擴到 64，≥1 GHz 仍保留 DOP builder。294 MHz、201 個 length points 實測 1.967 s，strong 10 µs、2001 點 spectrum 為 43.825 s。當輪 L8 中位數為 6.788 s，沒有達到 5 s 目標，不能挑首次 4.887 s 當 PASS。Strang 與 CF4 都用兩 workers、同 process 交錯、暖機後清 prepare cache。原報告未列逐 pair foreign fraction，不能當 isolated throughput，更不能跨兩輪相除。

### Padé7 比 Padé9 更划算

在 `058963f1a` 上，CF4 每 step 的兩次 Hermitian exponential 原本要做 eigendecomposition。prototype 改成 scaled [7/7] Padé，用矩陣乘法、linear solve 與 squaring，保留原 grids、refinement、九個 endpoints 比較及 dissipative split。

兩 worker、native threads=1、完整 target warmup 後清 cache 的獨立 process clean pair 中，L7strong 的 2001 點 acquire 為 31.539 → 20.670 s，L8 的 201 點為 5.851 → 3.961 s。這不是每個 workload 都保證 30% 收益。較早的 Padé9 只省 17.61%／12.65%，未勝過七階。

### Gain continuation 有效，carrier continuation 無效

`b0cb487f3` 上的 continuation prototype，在 L8 的一組最終 cold-preparation pair 中，把 acquire 從 3.087 → 2.215 s，省 28.3%。這是單組結果，不是重複中位數。H0 interaction-picture DOP853 沒有成為主要收益，因為重建 phase 與積分仍很貴。

同一原型在 L7strong 為 17.154 → 19.195 s，退步 11.9%。1103 次 carrier continuation 零命中，即使已移除 H0 phase、使用 normalized-phase nodes，仍未通過原檢查。因此研究只保留 gain 方向，沒有直接採用整份原型。

正式 gain anchors 改成依整組 unique gains 排序，每隔 24 個與端點建 anchors。非 anchor 在 interaction picture 插值後，仍跑原 coarse grid 與八相位 defect check。失敗時重用 coarse grid，回原 refinement。這是 calibrated convergence check，不是嚴格插值誤差界。

正式 L8 201 點比較用 interleaved pairs、CPU 0/1。workers=2 wall 為 3.207 → 2.193 s，workers=6 的 acquire CPU 比為 0.641，RSS 增 5.7%。W=2 RSS 增 8.7%，距當時 10% 限制只剩 1.3 個百分點。W=1/2/6、不同 chunks 的 197 張 tables 與 populations bitwise 相等，workspace 上限與 peak 都是 15.67 MB。這輪 W=6 指六個 Engine workers，計時 affinity 仍只有 CPU 0/1，不能當六核心 throughput。

### Constant Sambe 省掉大量 CF4 steps

在 `12f9e96a3` 的 study 中，L7strong 2001 點從 17.861 → 3.788 s。這組兩 worker 獨立 process pair 的 foreign fraction 為 2.58%／2.00%，完整 target warmup 後清 prepare cache。1107 keys 全用 cutoff=3，也就是 112×112 Hermitian Sambe matrix。一次 eigensolve 提供 period 與 fractional endpoints，不插值 eigenvectors。

原型用相鄰 harmonic cutoffs 在九個 endpoints 的最大 Frobenius defect 判斷收斂，門檻沿用 min(1e-4, budget×0.1)。它不是全時間 operator bound。高 gain L8 需要 cutoff=4。

正式版按 key 比 CF4 step estimate 與 Sambe matrix cost，不把所有 carriers 強制送 Sambe。相對 `956c86ddb` 的 L7strong clean pair 是 18.357 → 4.220 s，1107 builds、0 fallbacks，foreign 為 3.35%／2.53%。這與前述 prototype pair 是不同版本，不合併成一組結果。其他可用 pairs 的 foreign 差距在五個百分點內，但仍 contended。L8 比為 0.502，F4 為 1.006，不能都稱 clean gains。

Sambe candidates 與其 CF4 fallbacks 不進 gain-anchor plan，避免兩種 preparation 重複付費。當輪 36 個場景 routing 與比較基線相同。

## 被否決的選項與理由

- Padé9 比 Padé7 慢，沒有保留第二套同用途的 exponential 選擇。
- 全面開啟 carrier continuation 會付出預測與驗證成本，卻沒有通過原 guard。零命中不能用「相鄰 carrier 應該平滑」來推翻。
- gain anchor spacing=16 的 CF4 工作最少，但 RSS 超標。
- 第一版 fixed trajectories 雖 bitwise invariant，W=6 CPU 卻為基線 2.04 倍，RSS 增 33%。它只適合 W=2 的 stride，沒有採用。
- F4 的 constant Sambe 單 key 約 5 ms，比原 DOP853 約 3.1 ms 慢。加 physical-U carrier workspace 後，第一組 W=2 看似省 30.9%，但 baseline contended；對稱重跑退步 10.4%，W=6 退步 9.8%。這不是已證明的 ≥20% 候選。
- 放寬 error budget 或略過 polar、split 與 convergence checks，不屬於以上加速的依據。

## 留下的問題

ordinary period-table 的 adaptive counts 曾被提出，但在 `f4309632f` 的 34 個原尺寸場景、workers=2 重掃中，period_table routing、prepare share 與 apply share 全為零。原用 ordinary table 的四案已走 shaped Sambe，沒有場景達到繼續研究所需的 prepare CPU share≥25%。研究依停止條件結束，沒有 prototype，也沒有新舊 pair。

這只表示當時 workload 不值得優先做 adaptive counts，不能推論 ordinary family 對合法輸入不可達。workspace 的跨 evolve 生命週期也未由這些結果驗證。

## 依據

本文整合低頻 baseline、coverage audit、compiled period、Strang、CF4、Padé、continuation、constant Sambe 與正式整合報告。原始工作檔未入庫。歷史報告中的 prepare CPU、evolve wall、acquire wall 與外推分別保留，不跨版本相除。

[Dyson input reuse](2026-10-18-dyson-input-reuse.md)說明何時 period 比 distinct-input contraction 便宜；[shaped 研究](2026-10-18-shaped-period-and-gates.md)記錄另一種 carrier workspace，不能與此處失敗的 constant 原型混用。
