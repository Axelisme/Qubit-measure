---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0072-module-boundaries.md
  - ../adr/0073-measure-form-derivation.md
  - ../../tools/README.md
  - ../../lib/zcu_tools/simulators/README.md
  - ../../lib/zcu_tools/program/v2/simulation/README.md
  - 2026-11-21-experiment-run-wall-time.md
  - 2026-11-16-frozen-scene-storage.md
basis:
  - "本 repo 調查時的 main 32c7db0b57a71d503d2eca6ef29430416e9654cb"
  - "第一張耗時表的 commit fce233b5ce8d593d73897869bd9fcb1da550952a"
  - "本 repo 落地候選 f4c11395b1c5ec565be08c7d19f22ae1e37de029"
---

# 模擬器核心耗時：從完整 Run 讀，不建凍結場景

## 結論

3 分鐘目標的指標是每個實驗的模擬器核心耗時。`tools.experiment_wall_time` 照舊以現行表單預設跑完整 Run。核心是該 Run 各次 acquire 的編譯、演化、讀出與 raw materialize 四項之和。Controller、表單、Qt、繪圖、Analyze、Save 與 `ShotReducer` 等 host 處理不計。工具不建凍結的核心場景：改了實驗就用 `--select` 重跑那一列，比較模擬器改動時前後兩次在同一版實驗上跑。使用者對這項建議答「那就采用建议吧」。規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)。

落地時另有三項細化，都是 designer 代裁，使用者未覆核：

- Acquire 之後由 getter 才產生的 raw shots 用同一個計時。Getter 成功時，telemetry 補寫一筆帶原 acquire 身分的紀錄；已寫出的主紀錄與寫出順序不變。
- Run 在期限內到終態、每筆主紀錄分項齊全、每筆補充紀錄都配到主紀錄時，報表才填完整核心。其餘列只填下限，只加總分項齊全的主紀錄與配到它們的補充紀錄。完整核心或下限嚴格大於 180 秒才判超標，Run 秒數與期限不參與判定。缺新欄位的舊 journal 由新工具拒讀，改在原批次記錄的 commit 重建。
- 「分項齊全」只就現存紀錄判定。Telemetry writer 採盡力寫入，整筆遺失無法察覺，寫成工具的已知限制。

欄位與格式由 [Tools README](../../tools/README.md#實驗-run-牆鐘)、[acquire telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 與 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md) 擁有。

## 問題

[實驗 Run 牆鐘工具](2026-11-21-experiment-run-wall-time.md)記的指標是整個 Run 的牆鐘。使用者 2026-11-21 更正：「大头是模拟器本身，所以量模拟器速度就好了」。同一天使用者說：「其实我没有很想要跑实验的1-to-1，我觉得只要从参考场景提取出这个实验模拟的主回圈与主acquire之类的核心，然后测量它的时间就好了，其他的都不太影响。而且这样我改实验也不会需要重量。」在凍結核心場景、照舊跑完整 Run 改讀 telemetry、維持現狀三者中，使用者先選凍結，並說：「我甚至觉得可能不需要经过 Quick 的 program， 只要能够测量前端跟后端就好了，但可能实作上会比较麻烦，经过 Quick 还是行啊。那就选 A 吧。」其中「Quick」即 QICK，「A」即凍結核心場景。

調查找到的成因：工具從 acquire telemetry 讀編譯、演化、讀出三項，其餘記成 remainder。Raw shots 的物理訊號、雜訊與取整由 `simulate.adc` 產生，屬模擬器的工作。這段 materialize 在 acquire 之後才做，落在 remainder，和 host 的 `ShotReducer` 混在一起。第一張表（commit `fce233b5c`）三列完成量測的 acquire 外時間是 0.714、54.285、100.358 秒，分不出各屬哪一種工作。只讀三項時，`mux/onetone_flux_local`（Run 181.664 秒、核心 91.152 秒）與 `twotone/ro_optimize/auto`（Run 267.004 秒、核心 48.204 秒）都落在 180 秒內，超過 180 秒的列由 10 列變 8 列。先前估計的「acquire 外只有 0.1–0.2 秒」只在 heralded 的 live 路徑成立。

## 考慮過的選項與比較

- **照舊（K）。** 工具跑完整 Run，核心讀 telemetry 三項。覆蓋全部 adapter，不需改動。Raw materialize 不在核心內，核心少算。
- **錄製重播（F）。** 工具在明確更新時跑每個 adapter 一次，經既有 `SimulationCallbacks(on_end)` 與 `AcquireOutcome.rounds` 錄下每次 acquire 的 `CompiledStream` 與環境。計時器載入凍結資料，重播 lowering、FTF 後端與 raw materialize。凍結格式要保存 routes 來源、`soccfg` clocks、board profile、有效 FTF 設定、每輪 DC 快照與抽樣身分。`AcquireRequest` 的 `frozen=True` 只是淺層，巢狀陣列要深拷貝。估計要四張票：codec、錄製器、計時 CLI 與文件。依前次結果決定下一步的 3 個 adapter、13 個 `stop_condition` 呼叫點與 4 個 workflow，只能重播錄下的一條軌跡。調查後 designer 原本建議 F，全部 adapter 都錄。
- **手寫代表場景（H）。** 沿用 `family_pareto` 由工具持有數值場景的做法，補上 ADC 讀出，約 6 到 10 個場景。表變成每種形狀一列，不再逐實驗。
- **照舊加 raw materialize（B'，採用）。** 使用者問：「你觉得哪一种比较简单？是直接跑完整的1:1比较简单，还是复制场景比较简单？请你从维护性跟那个扩充性来考虑。」designer 答 1:1 比較簡單。現有工具已落地，自持 FTF 設定與 board profile，可用 `--select` 重跑單列，新增的實驗自動入表。F 的凍結格式綁住 `CompiledStream`、routes 與 FTF 設定 schema，這些型別在模擬器優化期間常改；每改一次就要重錄，而重錄本身就要跑 1:1。H 要人工跟著 program API 改。代價是改了實驗要重跑那一列，比較模擬器改動要在同一版實驗上跑，raw materialize 要補進 telemetry。使用者答「那就采用建议吧」。

是否經過 QICK program 不影響這個選擇。合成樣本的 QICK build 是 8.91 ms，lowering 是 2.74 ms；樣本來自 dirty tree，只作量級。

落地時的三個子問題：

- **延後的 materialize。** `run_acquire_shots` 在 acquire 紀錄寫出後才呼叫 `get_shot_result()`。`singleshot/ge`、`check`、`amp_rabi`、`len_rabi`、`qutrit_population` 走這條路。選項是只記 acquire 內的 materialize，或讓延後的 materialize 走同一個計時並補寫紀錄。採後者。Getter 失敗時不重試、不補寫；硬體路徑沒有模擬器，也沒有紀錄。
- **下限與判定。** 下限可以只加總分項齊全的主紀錄，也可以逐項加入不完整主紀錄的已知秒數。新 writer 每筆主紀錄都寫齊分項；不完整只來自逾時時尚未寫出的 acquire、尚未寫出的補充紀錄或寫入失敗，所以採前者。不完整列可以一律未判定，也可以在下限嚴格大於 180 秒時標「已知至少超標」；採後者，只報實測秒數證明得了的那一側。舊 journal 可以拒讀，也可以加相容格式；採拒讀。
- **整筆遺失。** Writer 在保留上限不足時刪最舊檔，寫入失敗只發警告。中間某筆紀錄整筆消失時，剩下的紀錄可能都齊全且都能配對，工具會報完整核心。選項是保留判定規則並寫明限制，或要求獨立證據確認所有紀錄都已寫入。採前者，Run 層完備性登記為後續候選。

## 被否決的選項與理由

- 凍結核心場景與 F：要多一套版本化凍結格式與重播器，綁住常改的型別，重錄仍要跑 1:1。
- H：手寫序列會和真實實驗漂移，表變成每種形狀一列。
- K 原樣：raw materialize 不在核心內，核心少算。
- 只凍結 `Engine.evolve` 的輸入：量不到讀出。`chain_length` 在 2.2 µs 時核心約 358.5 秒，演化只有約 4 秒。
- 只收 request、不跑後端的錄製 SoC：沒有結果，依前次結果決定下一步的 adapter 與 workflow 走不下去。
- 凍結 `AcquireRequest`：重播量不到 lowering。
- 只記 acquire 內的 materialize：常用的 single-shot 路徑被排除，核心會少算，還會把 0 當成完整分項。
- 把 acquire 紀錄改到 `get_shot_result()` 之後才寫：改動 Schedule 的寫出順序，影響其他讀取方。
- 逐項加入不完整主紀錄的已知秒數：新 writer 不產生這種紀錄，規則沒有真實輸入，還會混加涵蓋範圍不同的分項。
- 不完整列一律未判定：丟掉實測下限已證明的超標。
- 以 Run ≤ 180 秒推定核心達標：需要 acquire 不重疊、核心只在 Run 內的前提，沒有調查。
- 舊 journal 的相容格式：舊 remainder 含 materialize，新定義不含，要維護兩套語意。舊批次在原 commit 就能重建，repo 預設也不加 legacy 邏輯。
- 確認所有紀錄都已寫入的 Run 完備性協定：要為未實測的遺失率新增跨 producer 與 consumer 的契約。這支工具是開發用的診斷量表，誤判一列會在下一次全表或單列重跑時發現。

## 先例與立場

- [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則)：工具自持核心 pulse 序列的凍結場景只適用固定 optimizer benchmark。遵循，本題不擴充它的範圍。
- [凍結場景的存放](2026-11-16-frozen-scene-storage.md)：區別。那是 optimizer benchmark 場景的存放規則，本題不建場景。
- [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 的分層：遵循。牆鐘工具仍是唯一的整實驗例外，kernel 與 family 比較仍只量 `Engine.evolve`。
- [ADR-0078](../adr/0078-evolution-module-boundary.md#算法內聚共用數值有單一-owner) 的共用數值單一 owner：遵循。計時放在呼叫 materialize 的模擬 binding，工具只彙總。
- [ADR-0073](../adr/0073-measure-form-derivation.md#自動推導點到為止) 的使用者慣例「一般实验用简单的做法就好了，有需求再做多余的设计」（2026-11-21）：遵循。不建凍結場景，不加 Run 完備性協定，有具體需求再設計。
- 先前工具把所有 `timeout` 列進「Run >180 s」清單，依據是 Run 大於期限、期限至少 180 秒：修改。指標改成核心後，Run 期限證明不了核心超過 180 秒。只有下限嚴格大於 180 秒的列標「已知至少超標」，其餘未判定。
- [程式碼品質](../code-quality.md#在適當位置建立資料與錯誤契約)「量測失敗不能偽裝成成功的空資料」：遵循。缺 materialize 不補 0，沒有齊全主紀錄時下限寫 `null`。
- [實驗 Run 牆鐘工具](2026-11-21-experiment-run-wall-time.md)：指標部分由使用者更正；工具入庫、跑完整 Run、不當 gate 仍成立。

## 留下的問題

- 不完整列的分支在落地時只由工具的 synthetic `main(argv)` 測試走過。第一個真實 `timeout` 列要等下一次全表。
- 第一張表沒有 materialize 欄，按新定義的完整核心要等下一次全表。
- 下限不超過 180 秒的逾時列，要另外量到完整 Run 才知道核心是否超標。
- 完整 Run 無法比較實驗已改過的兩個時間點。出現具體需求再設計凍結場景。
- Run 層的 telemetry 完備性等出現實測遺失或誤判達標的案例再處理。

## 依據

使用者原話與日期見上文。調查的各角度報告、dossier 與 challenge，以及三項代裁的原始工作檔都未入庫，結論所需的事實與數字寫在本文。第一張表的數字來自 commit `fce233b5c` 的批次輸出；`chain_length` 的 358.5 秒與 4 秒來自後續一次重測；兩份輸出都未入庫。

候選 `4dd7e11ab` 上，單列 `singleshot/ge` 以非排他 runner 執行，exit 0。兩筆延後補充紀錄都配到主紀錄：`materialize_s` 3.154541253 秒、`core_s` 14.695911851 秒、`remainder_s` 0.130404188 秒。非排他秒數只作診斷，收據未入庫。

## 後記

2026-11-23，[每點速度目標](2026-11-23-simulator-speed-per-point.md)取代掃描與逐次評估的整次 180 秒判定。核心定義、部分量測只報下限、以及無軸 shots 的 180 秒目標不變。
