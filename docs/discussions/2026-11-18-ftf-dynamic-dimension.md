---
date: 2026-11-18
status: active
superseded_by: null
related:
  - 2026-11-18-ftf-subspace-reduction.md
  - 2026-11-18-remove-private-replay.md
basis: main 1ae244996 與 integration 7b45c6a19（含 TimelineBatch.select_points）。原始量測工作檔未入庫，必要數字寫在正文。
---

# FTF 動態維度：`retained_levels` 改為上限，每次 acquire 用 pilot 選維度

## 結論

`retained_levels` 從固定維度改成上限。FTF runtime 在符合條件的 acquire 自己選維度 N，`Engine` 不做約化。

- **適用條件：** 至少 8 條 timeline，只有終點讀出，沒有早期 cavity drive。其他 acquire 用設定的維度，結果不變。
- **選 k：** 以上限基底解一條驅動最強的 timeline（pilot），從它的完整 density 讀出 `drop(k)`，也就是截到前 k 態會丟掉的最大元素。取最小的 k 使 `drop(k)` ≤0.5×budget。
- **驗算：** 先估剩餘 timelines 的預估節省是否大於驗算本身的成本。划算才以 k 驗算一次，完整 density 差 ≤budget 就用 k；不划算或驗算不過，就用上限。不逐個試其他 k。
- **輸出：** pilot timeline 的輸出是它的上限解，其餘 timeline 整批以 N 求解。N 維模型取上限模型能量排序的前 N 態，結果補零回上限基底再讀出。
- **重用：** 有 pilot 的 acquire 也進跨 acquire 的 preparation 快取，並記下 N。重跑同一個 acquire 不重做 pilot。
- **切列：** pilot 與剩餘批次的切列由 `simulate.evolution.TimelineBatch.select_points` 提供，FTF 只選點索引。
- **預設啟用：** budget 用 FTF 預設 1e-3，不分精度檔。

## 問題

GUI 預設的 `retained_levels` 24，對單 qubit／單 tone 掃描偏大；CZ 這類雙 qubit 序列又需要它。先前的子空間約化討論結論是：`Engine` 內沒有便宜又嚴格的逐點 guard，維度由使用者宣告。結果使用者在不同實驗之間要來回改 `retained_levels`。

## 考慮過的選項與比較

**啟用範圍**
- 兩個精度檔都預設啟用，`retained_levels` 當上限（採用）。
- 只在 fast 檔啟用。
- 獨立開關，預設關閉。
- 維持手動改 `retained_levels`。

**精度檔位：** 曾考慮用「程式裡有沒有 flux 脈衝」這類辨認來切換寬窄預算。使用者否決，理由是 CZ 也可能從 charge line 打，模擬器也不該知道是什麼架構。精度需求改由使用者宣告，沒宣告就用預設 budget。

**怎麼選 k**
- 由小到大逐個試：k 太小、驗不過時，成本隨試的次數增加。
- 二分搜尋：最壞仍要約 log₂ 次演化。
- 只看 `drop(k)`、不驗算：`drop(k)` 只是下界，漏掉截斷對留下來的態的動力學影響，例如上層態造成的 AC Stark 相移。第一輪的相鄰 N 互比估計器只看指標，15 點中有 9 點超標。
- 從一次上限 pilot 的讀值直接算 k，再驗算一次（採用）。

**適用範圍**
- 只做終點讀出（採用）。
- 擴到無 herald 的早期 cavity drive：pilot 要包含多段 evolve，成本與收益沒量過。
- 兩類都支援：要另定 history 權重、缺失 branch 與逐 branch density 的誤差契約。

**pilot 的前置成本**
- 門檻提高到 16 條、pilot 結果不沿用：使用者否決。
- 門檻維持 8 條，pilot 的上限解直接當它的輸出（採用）。
- 把驗算併進整批，先以 k 樂觀地跑完再比：驗算不過時整批白算；要改 `Engine` 的排程邊界。
- 驗算前先估價（採用）。這是正式量測發現「k 接近上限時，驗算花了上限解的 84%，剩下只有一輪」之後加上的。

**切列由誰負責**
- FTF 自己重建 `TimelineBatch` 的 ragged flux schedule 與 waveform offsets：呼叫端在組被呼叫者的內部版面。批次日後新增 point-aligned 欄位時會漏接。
- 由批次的 owner 提供中性切列（採用）。新方法在 F1 與四個正式案例上切出的每個欄位、完整 density 與逐列 family，都和原本的切法逐位相同，所以正式計時沿用原樹的數字。

## 數字

- **精度：** F1 gain 24 點對 N32 oracle 的完整 density 最大差 2.69e-4，在 2e-3 內。四種掃描都選 N = 12。
- **速度：** 對含 CPU DP5(4) 的 main，7 workers，CPU，K=3 ABBA，演化階段 wall：
  - F1 gain 24 點：快 1.2025 倍。CPU 時間約少 2.8 倍，單次約 150 ms。
  - 雙 qubit 同時驅動 24 點：快 2.49 倍（較早一次的正式量測）。
  - 判定不降維：慢 1.157 倍。
  - 估價不回本、跳過驗算、留上限：慢 1.117 倍。同一個 fixture 在加估價之前會實際驗算，驗算不過，當輪慢 1.46 倍。
- **開銷：** F1 在兩臂可辨識的 Joblib wall 約 14 ms，其中約 12–13 ms 是最後一個 job 完成後的 completion polling。先前從 CPU 與 wall 推估的「兩臂共有約 50 ms 固定開銷」沒有被診斷證實。

## 使用者的判斷與修正

- 啟用範圍：對「兩個精度檔都預設啟用，`retained_levels` 當上限」答「A」（2026-11-18）。
- 精度檔位：「这不合理，CZ gate 可能也是从 charge line 打，而且模拟器理应不知道现在的架构是 FTF 还是只有单个 F 还是什么的。」（2026-11-18）
- pilot 門檻：「嗯，A，但你也要想办法降低前置计算的成本，真的」（2026-11-18）
- 前置不採用時的代價：「以比较快的场景稍微变慢一点为代价，让慢场景快很多是可以接受的」（2026-11-18）。使用者同意把這條套到「pilot 判定不降維」：只記錄，演化階段 wall 超過基準的 1.2 倍才回報。
- 驗收倍率：使用者把算法納入與門檻交給 designer 決定：「你们要帮我决定啊，因为你们最清楚各项参数的细节。你应该知道这个算法在什么情况下有优势，什么情况下有劣势，值不值得的，要不要纳入，要不要采取变种，是你们帮我决定的。」（2026-11-18）。designer 把 F1 門檻從 1.3 倍改為 1.1 倍，理由是 F1 每次約 150 ms，wall 被固定成本主導，而 CPU 時間少約 2.8 倍；慢的掃描（雙 qubit 同時驅動）快 2.49 倍。

## 留下的問題

- 驗算不過的分支，只有合成 density 的證據，沒有真實場景觸發。
- 非 pilot 點與後續 DC 的截斷誤差，以 pilot 點代表，屬於估計的限制，沒有發現超標。
- 約 12–13 ms 的 completion polling 沒有處理。它影響所有多 worker 的 evolve。
- 有早期 cavity drive 或 projective read 的 acquire 不做 pilot。

## 依據

- 本 repo：main `1ae244996`、integration `7b45c6a19`。
- 規則與欄位：`lib/zcu_tools/simulators/ftf/environment/README.md` 的〈dimension selection〉、〈routing telemetry〉、〈preparation reuse〉；`docs/adr/draft/simulation-environments.md`〈分段引擎與 debug capture〉。
- 原始量測工作檔未入庫。
