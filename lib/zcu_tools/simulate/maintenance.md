**讀者：** framework-dev、agent。改模擬器之前，照這份決定改哪裡、怎麼量、怎麼驗收；每條附出處與自查方式。

# 模擬器維護規則

範圍：`simulate`（主要是 `simulate.evolution`）、`simulators` 的三個環境，以及服務它們的 `tools/` 量測工具與 `tests/simulate` 精度測試。

這份文件寫做事的順序與自查方式。規則的理由與完整條文由 ADR-0078、README 與討論紀錄擁有，每條都連到它的 owner。
這裡只收已經在落地的改動中用過、結果記錄在 git 的做法；新做法照〈6.2〉加入。

## 1. 動手前

**1.1 先找有紀錄的瓶頸。** 速度優化只做使用者實際在等的場景。
瓶頸從已有的耗時紀錄找：[kernel 表](evolution/README.md#kernel-summary)的實測證據、量表摘要與討論紀錄；找不到紀錄才問使用者。
要支援、要算對的輸入範圍由物理與介面決定，不問使用者常不常用。

- 依據：ADR-0078〈[先保證正確，再優化瓶頸](../../../docs/adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸)〉〈[支援範圍不看使用頻率](../../../docs/adr/0078-evolution-module-boundary.md#支援範圍不看使用頻率)〉。
- 實例：`t6` 的 direct 做法從四個有耗時紀錄的慢場景下手；掃長度的 first-evolve wall 從 736 s 降到 6.3 s，四個場景的最終誤差都 ≤ 2e-3（[t6 的 direct 加速策略](../../../docs/discussions/2026-11-14-t6-direct-strategy.md)）。
- 自查：說得出要改的場景，以及它目前的耗時紀錄在哪個檔案。

**1.2 先試改寫既有路徑，再考慮新 family。** 要加速一條路徑，先試改寫它的內部做法。
新做法在同一批場景上不被現有路徑支配，才進 production。

- 實例：`t6` 的 direct 做法放在 `t6` 內，名字、適用域與 guard 不變，沒有新增 family（同上）。
- 自查：新增 family 時，寫得出為什麼不能改寫既有 family。

**1.3 有誤差問題的 family 先修，不先刪。** 提出收窄或刪除之前，先查清誤差或變慢的來源，以及能否修正。
修不好的範圍才收 guard；瓶頸範圍留在快速算法。

- 依據：ADR-0078〈[快速算法先修好，不輕易放棄](../../../docs/adr/0078-evolution-module-boundary.md#快速算法先修好不輕易放棄)〉。
- 實例：`floquet` 再研究一輪並設停損條件，結果在 committed 批次上比 `dop853` 快 1.65–1.81 倍，因此保留並重校價格（[Floquet 的保留與價格重校](../../../docs/discussions/2026-11-16-floquet-kept-price.md)）。
- 邊界：已經算得準、卻沒有服務任何現有瓶頸的做法可以移除，例如 `t6` 的 channel 做法（[t6 只保留 direct](../../../docs/discussions/2026-11-15-t6-channel-removal.md)）。
- 自查：收窄或刪除的提案寫明原因、試過的修法，以及修法為什麼不夠。

**1.4 修法還在研究時，先走慢但對的候選。** Guard 在整個參數空間不放行最終誤差超過 2e-3 的結果。
修法已經在做、很快落地時，過渡期維持現狀。
修法時程未定時，先收窄 admission，讓其他候選算對，修好再放回。

- 依據：ADR-0078〈[先保證正確，再優化瓶頸](../../../docs/adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸)〉。
- 實例：sampled multitone 在修法研究期間暫停 admission，改由其他候選計算；速度代價照樣用 benchmark 判斷（[Sampled multitone 的過渡期拒收](../../../docs/discussions/2026-11-14-sampled-multitone-interim.md)）。

## 2. 改哪裡

**2.1 加或改 family 只動固定的位置。** 照 [Adding a family](evolution/README.md#adding-a-family) 改：kernel 自己的 module（guard、prepare key、preparation、apply）、`kernels/__init__.py` 的實例、owning domain classifier 的具名選擇；要比成本時再加 dispatch helper 的價格式。
選路知識全在 dispatch，不加 registry、Protocol 或能力宣告。

- 依據：ADR-0078〈[內部允許實質耦合，dispatch 擁有決策知識](../../../docs/adr/0078-evolution-module-boundary.md#內部允許實質耦合dispatch-擁有決策知識)〉〈[算法內聚，共用數值有單一 owner](../../../docs/adr/0078-evolution-module-boundary.md#算法內聚共用數值有單一-owner)〉。
- 實例：`common_frame` 的步長成本改由 dispatch 計價，kernel 只留數值 guard（[common_frame 的步長成本歸 dispatch](../../../docs/discussions/2026-11-14-common-frame-step-pricing.md)）。`t6` 的定價、key inventory 與選擇由 dispatch helper `_t6_dispatch` 擁有（[t6 的 direct 加速策略](../../../docs/discussions/2026-11-14-t6-direct-strategy.md)）。
- 自查：diff 只動上面的位置、該 family 的數值 owner 與它的 dispatch helper；呼叫端沒有逐段選算法。

**2.2 選路選到較慢的合格 family 時，先查價格。** 在 dispatch helper 重校價格式；guard、放行門檻與候選順序不跟著改。
校準用的場景與適用範圍寫在 evolution README 或價格模型的 docstring。

- 實例：`floquet`／`dop853` 的 two-tone 價格重校只改價格，沒有改 guard、放行門檻或候選順序（[Floquet 的保留與價格重校](../../../docs/discussions/2026-11-16-floquet-kept-price.md)）。
- 自查：重校前後，在同一批場景上比較選路結果與耗時。

**2.3 Guard 只表示算法的精度邊界。** 使用者的實驗參數設得不好時，模擬器照常給結果，算不準的部分以警告說明是模擬器的限制。
契約外的輸入（型別、範圍、維度錯）照舊在進入點 raise。

- 依據：ADR-0078〈[模擬器不替使用者擋實驗參數](../../../docs/adr/0078-evolution-module-boundary.md#模擬器不替使用者擋實驗參數)〉。
- 實例：projective read 改成 Markov chain 時，51 點、21 能階的合法參數沒有被記憶體預算擋下（[Projective read 的輸出改成 Markov chain](../../../docs/discussions/2026-11-17-projection-markov-chain-readout.md)）。
- 自查：新加的每個 raise 擋的是契約外輸入或算法超標，不是「參數不好」。

## 3. 精度驗收

**3.1 對獨立參考解量完整序列的最終誤差。** 完整 density 最大絕對差 ≤ 2e-3。
真值是獨立的完整 Hamiltonian 參考解，不是改動前的輸出；不要求改動前後逐值一致。
單一 kernel 的誤差或每段 `budget` 不算合格證據。

- 依據：ADR-0078〈[歷史控制與精度驗收](../../../docs/adr/0078-evolution-module-boundary.md#歷史控制與精度驗收)〉〈[Family 去留：合格只比耗時](../../../docs/adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)〉。

**3.2 精度矩陣跟著 family 改。** 每個 family 在精度矩陣有代表輸入；有數值 guard 的 family 另有一個剛好在 guard 邊緣內側的案例。
新增 family 或改 guard 時，同一個 diff 補這兩類案例，並更新 kernel 表。

- 依據：[Evolution 精度矩陣](../../../tests/README.md#evolution-精度矩陣)、[Adding a family](evolution/README.md#adding-a-family)。
- 實例：邊緣案例上線時找出四個超標的 family，現以 strict xfail 保留實測誤差（ADR-0078〈歷史控制與精度驗收〉）。
- 自查：terminal summary 裡該 family 是 `PASS`；不是 `PASS` 的，是帶實測誤差的 strict xfail，且已回報使用者。

**3.3 超標不放寬門檻。** 已知超標以 strict xfail 保留實測誤差，新發現的超標回報使用者。
修正照 1.3、1.4 處理。

**3.4 選路歷史用公開參數控制。** 測試與量測用 `RoutingHistory` 與 `families=` 控制空歷史、沿用歷史與允許的 family，不 monkeypatch 內部狀態。

- 依據：ADR-0078〈[歷史控制與精度驗收](../../../docs/adr/0078-evolution-module-boundary.md#歷史控制與精度驗收)〉。

## 4. 量測

**4.1 按層量，不做 e2e。** Kernel 層量一段輸入：`Engine.evolve` 以 `families=` 固定候選；測重複就是同一段接 N 次；一次只改一個約化軸；誤差與耗時在同一組座標上量。
序列層用手寫的多段 `TimelineBatch`，量最終誤差、整體耗時與 family 去留。
選路用 `Engine.inspect_dispatch` 檢查。查詢可演化 guard 自己的探針，不跑使用者 timeline。
使用者回報的錯在各層都重現不出來時，才跑一次 e2e 定位；腳本不入庫，也不當門檻。

- 依據：ADR-0078〈[模擬器的測試與量測分層，不做 e2e](../../../docs/adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)〉〈[每個 kernel 的量測場景只有一段](../../../docs/adr/0078-evolution-module-boundary.md#每個-kernel-的量測場景只有一段)〉；[Simulator benchmark](../../../tools/README.md#simulator-benchmark)。
- 自查：量測腳本只呼叫 `Engine.evolve` 或 `Engine.inspect_dispatch`，不經 program 編譯或 acquire。

**4.2 場景照約化軸宣告。** 不為單一實驗加專屬場景。
把輸入約化成 kernel 與 guard 實際看到的物理量（耦合、失諧、時長、包絡形狀、tone 間距），事先宣告格點；每個實驗只核它的座標落在已覆蓋的區域。
約化會丟掉影響結果的物理時，加一條新軸。
序列場景避開會讓誤差互相抵消的固定相位循環。

- 依據：ADR-0078〈[場景覆蓋靠約化的參數空間](../../../docs/adr/0078-evolution-module-boundary.md#場景覆蓋靠約化的參數空間)〉。
- 實例：固定的 X、Y、−X、−Y 循環會抵消部分 RWA 誤差，量表因此加了隨機相位的 RB 型序列（[RB 型相位序列作為固定量表輸入](../../../docs/discussions/2026-11-14-random-gate-sequence-workloads.md)）。

**4.3 新場景走凍結流程。** 場景由 tracked producer 建，數值凍結進共用快取，Git 只多一行名稱與三個 hash。
Inventory 已宣告、index 還沒有的名字，用 `--update-index --scenario NEW` 明確加入。
參考解的 solver 留在 tests 側；求解前先取得運算授權。

- 依據：[Freeze numeric scenes](../../../tools/README.md#freeze-numeric-scenes)、[Evolution 精度矩陣](../../../tests/README.md#evolution-精度矩陣)；[凍結場景改為共用快取](../../../docs/discussions/2026-11-16-frozen-scene-storage.md)、[共用清單的 Pareto 選集與明確新增場景](../../../docs/discussions/2026-11-17-pareto-selection-and-scene-append.md)。

**4.4 速度比較在同一次 exclusive 內成對量。** 跨 checkout 的版本比較用 `tools.paired_benchmark run`。
存下來的跨時段耗時只當量級參考，不用來判快慢。

- 依據：[Paired Engine benchmark](../../../tools/README.md#paired-engine-benchmark)；[成對 Engine 計時的 process、身份與統計](../../../docs/discussions/2026-11-16-paired-engine-timing.md)、[Family 量表的單場景重量與摘要](../../../docs/discussions/2026-11-14-family-pareto-remeasurement.md)。

**4.5 變慢超過 10% 先回報。** 完整 benchmark 的任一工作負載、history mode 或 clock 變慢超過 10%，合併前回報使用者。

- 依據：[Sampled multitone 的過渡期拒收](../../../docs/discussions/2026-11-14-sampled-multitone-interim.md)〈問題〉段、[t6 的 direct 加速策略](../../../docs/discussions/2026-11-14-t6-direct-strategy.md)的成對比值門檻。

**4.6 Family 去留照「合格只比耗時」。** 最終誤差合格的 family 只比耗時，另一個合格者快超過 10% 才算支配。
刪除較準但較慢的 family，要有同型態長序列的證據，並由使用者看過數據點頭。

- 依據：ADR-0078〈[Family 去留：合格只比耗時](../../../docs/adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)〉；量表工具見 [Engine family Pareto table](../../../tools/README.md#engine-family-pareto-table)。

## 5. 測試成本與工具

**5.1 測試用小模型，完整量測放在測試外。** 測試用小模型、短 timeline，每個測試單核心少於 30 s。
完整精度與效能量測放在 task 的量測腳本或入庫工具。
數值測試標 `sim`，經 `tools/run_heavy.py` 執行。

- 依據：[測試成本](../../../tests/README.md#測試成本)、[Evolution 精度矩陣](../../../tests/README.md#evolution-精度矩陣)。

**5.2 重複實做的量測工具才入庫。** 入庫的工具要好用、好讀，並在模擬器領域內通用：任何 family、任何場景都能用。
一個功能要等兩個 task 實際需要才加。

- 依據：[Tools 標準化](../../../docs/discussions/2026-11-16-tools-standardization.md)、[實驗 probe 的 snapshot 保存與比較](../../../docs/discussions/2026-11-16-probe-snapshots.md)。

## 6. 落地與增修

**6.1 落地時更新 owner 文件。** Kernel 表寫每個 family 的適用情況、近似與 guard 門檻、精度案例，以及它不被支配的場景。
改了選路或價格，同步改 kernel 表或價格模型的說明。
依 repo 規則寫討論紀錄。

**6.2 加一條規則的條件。** 新規則要同時滿足兩件事：有使用者裁定、ADR 或原則；至少一次落地的改動照它做過，結果記錄在 git（討論紀錄、README 或量表摘要）。
規則本身寫進它的 owner；這份文件只加做法、連結與自查。
