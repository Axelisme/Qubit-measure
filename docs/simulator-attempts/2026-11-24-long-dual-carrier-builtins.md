---
date: 2026-11-24
status: paused
commit: 32dfb53a5
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 長時強驅動雙載波的內建波形路由

**讀者：** framework-dev、agent。查閱不同載波雙線為何留在直接積分，以及何時值得再查。

## 核心思路

先依波形與選路結果分類內建脈衝格，再從直接積分的 CPU 成本挑出診斷對象。檢查長時、強驅動的不同載波雙線能否在不放寬 guard 的情況下改走較快的 family。

## 實測

脈衝普查使用 `593426c2f`、24 態 FTF、固定 flux、CPU 單 worker，每格 inspect 和 evolve 各一次，process CPU 含首次 JIT。90 個內建格有 55 格執行過 `dop853` family，包含混合路由。排除兩筆未拆分 JIT 成本的 20 ns 異常值後，只提出 `dop853` 的雙線格成本最高。其中最花時間的是 1000 ns 的 cosine、Gaussian、DRAG 和 flat-top。這四種波形各用估算 π 振幅的 1 倍和 2 倍，兩線載波約為 0.298 和 0.222 GHz，取樣包絡用 129 個線性插值節點。這 8 格的 10 個直接積分段合計 6.449438 CPU s，只算 DP5(4) apply，不含 flat-top 平台。這不是暖機重測。

在 `32dfb53a5` 用相同設定重放選路與 guard 診斷，輸入是上述 8 格與 4 個 0.1 倍 π 的弱振幅對照。預設選路只為目標的 10 個直接積分段列出 `dop853`；弱對照的 5 個直接積分段先被 `common_frame` 拒收，再選 `dop853`。另行檢查 `common_frame` 時，這 15 段的 coherent 指標都超過 0.0018 門檻，目標 10 段為 0.154–7.25。這是 guard 指標，不是 density 誤差。診斷未執行演化，也未收集計時。原始普查輸出、分類表與診斷工作檔未入庫。

## 為何暫停

改估價至多讓 `common_frame` 成為候選，不會讓這些段通過既有 guard。其他候選也未證實能取代直接積分。本次不改路由或 guard，公開行為維持不變。沒有修好的格可新增 `Engine.inspect_dispatch` 路由覆蓋契約，也沒有前後配對計時。

## 留下的線索

1. 目標段的預設選路未提出 sampled `multitone_rwa`。現有診斷未拆開省略項的 proxy 篩選與估價跳過，不能把未提出候選當成 guard 拒收。
2. 按 `32dfb53a5` 的價格模型，對 budget `1e-3`、段長至少 100 ns 的單段，`multitone_gate` 冷建 30 個 full-drive 端點節點的估價約是同段 DP5(4) 的 37.8 倍，還沒算 Fourier 或套用。這是估價比，不是實測，也不能外推到多列共用同一份 gate 的成本。
3. 這類非平台段的現行成本是 `dop853` family 內的 CPU DP5(4) 直接積分。以速度普查第 1b 塊（補測單 qubit／mux）、第 2 塊（雙 qubit 與同頻雙線）或真實工作負載的排名為準。只有當長時、強驅動的不同載波雙線排進慢場景前段時才重開。
