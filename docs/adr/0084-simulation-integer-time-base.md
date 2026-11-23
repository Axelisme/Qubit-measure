---
status: accepted
---

# 0084 — 模擬 request 的公共整數時基

## 脈絡

Compiled simulation 的 request 目前以浮點微秒傳遞時間。Frontend 與 consumer 各自加總時間、建立 ADC 取樣網格。同一時間以不同運算順序組合，結果會差一個 ulp。固定 seed 不能消除這個差異。

各時鐘的週期不一定是 tProc 週期的整數倍。例如 mock 的 tProc、generator fabric、ADC output、envelope sample clocks 分別為 430.08、384、307.2、6144 MHz。只改用 tProc tick 會遺失其他時鐘的格點。

## 決策與落實狀態

2026-10-31，使用者提出「我傾向根除的解決方案」，並選擇「2b」。採用第一階段的公共整數時基 request。第二階段的物理建模留待後議。

`simulate.boundary.TimeBase` 擁有公共有理數頻率與換算規則。Program 從 soccfg 擷取本次 tProc、已宣告 generator 的 fabric／envelope sample clocks 與已宣告 ADC 的 output clocks，再交給 TimeBase。未宣告 channel 不參與，TimeBase 屬於單一 request。TimeBase 不讀 soccfg，也不 import driver。額外宣告不影響既有事件的異質 clock channel 可以改變時基，但不能改變原 channel 的物理時間與模擬輸出。Frontend 的 ns 網格與 physical IQ 不變性已納入驗證。測試額外宣告 512 MHz fabric generator，時基由 43008 變為 172032 MHz，原 generator／ADC 的事件與 IQ bits 不變。

TimeBase、request schema、compiled frontend 與 physical consumer 使用整數時間。Planner 以必填 time_base keyword 接收 request 的時基。TimelineTemplate 不重複保存時基。Plan 與 preparation cache key 納入時基。

## 時間契約

TimeBase 的頻率為約分後的有理數 `F = num / den`，單位為 ticks/μs，也就是 MHz。一個 tick 為 `den / num` μs。

- 每個時鐘以 `Fraction(repr(clock))` 建立有理數，並驗證轉回 float 不變。公共頻率取各有理數時鐘的最小公倍數。空時鐘集合、非正數、非有限數或失真轉換報錯。
- Request 攜帶 TimeBase。Start、duration、trigger、phase reference、envelope 取樣週期及 ADC 取樣週期都用整數 tick 表達。Frontend 全程做整數運算，不先轉成浮點微秒再組合。
- 手建 fixture 也必須落在格點上。非格點時間報 `ValueError`，不四捨五入回格點。需要非十進位的精確時間時，caller 可提供 Fraction。
- `TimeBase.ns` 是唯一的 tick→ns 換算入口。它支援 scalar 與 int64 ndarray，array shape 不變。每個結果都等於有理數 `ticks * 1000 * den / num` 正確捨入一次所得的 float64。
- 既有 2 ns support 量化由 `TimeBase.ns(..., quantum_ns=2)` 對精確有理數時間做整數 ties-even 捨入。未傳 quantum_ns 時保留單次正確捨入的原換算。
- ADC 網格先在整數域組成 `trigger + k * period`，再交給 ns。Readout plan 與環境 template 呼叫這個入口，不各自展開換算迴圈或浮點乘除。
- Num 必須能在 float64 精確表示。Request 的時間輸入與中間結果須在 int64 範圍內，另限制 `abs(ticks * 1000 * den) <= 2**53`。超出範圍報 `OverflowError`，不截斷、不溢位回繞。
- Relax delay 沿用 QICK 真正編譯時的 tProc-cycle 捨入，再映射到公共整數時基。它不以 raw 作者參數套用手建 fixture 的格點限制。

Mock 時鐘的公共頻率為 43008 MHz。對應的 tProc、fabric、ADC、envelope 週期為 100、112、140、7 ticks。ADC output 為 76.8 MHz 的 FTF 配置，其 ADC 週期為 560 ticks。

### 模擬 round 時長

**規則。** 整數 request 的單輪時長歸 `program.v2.simulation.round_time.request_round_time_s`，只由模擬 binding 使用。`zcu_tools.program` 不再匯出這個函式。公式不變：以 Python 整數加總所有掃描點的 duration ticks，每點計入一次 relax delay，再乘以 reps。最後由 `TimeBase.seconds` 將有理數時間換成秒。

**例子。** 單點時長 2 µs、無 relax delay、100 reps，單輪時長是 200 µs。Binding 用這個物理時長做 [round pacing 及時鐘補時](../../lib/zcu_tools/program/v2/simulation/README.md#round-pacing)。

**邊界。** 單一 request 的 int64 與 ns 換算界限不套用到整輪加總；總時長不會因此讓長 acquisition 報錯。`shot_time_s` 仍在 Program，沿用 µs 輸入供一般估時。硬體取數路徑與公共時基不變。

**依據。** 2026-10-31，advisor 依代理授權裁定原先的 owner 與計算契約。Designer 依 2026-11-23 的授權代裁，只改 owner 與 Program 的公開匯出，保留計算契約。使用者於 2026-11-23 追認代裁：「0242採用建議」。依據是使用者 2026-11-22 的界線：「我认为这两处可能有违背了我的设计理念。对我来说，只要把这个 mixin 拿掉，我的 program 就不会有模拟的逻辑存在了。」見 [ADR-0072〈真實 acquire 與模擬邊界〉](0072-module-boundaries.md#真實-acquire-與模擬邊界)與 [Program 的模擬邏輯只在 mixin 內](../discussions/2026-11-23-program-simulation-mixin.md)。

## 取捨與範圍

保留浮點時間並調整加總順序，只能固定某一條計算路徑。讓每個 consumer 自行量化，則會把格點規則散在多個模組。本決策改變 request 格式，接受重寫 fixture 與拒絕舊 replay 的成本，讓時間的組合順序不再影響換算結果。

本階段不改變物理模型。平移不變、整數 DDS 相位、ADC 相對動力學，以及第一個樣本對齊 ADC clock 的建模，都不在本次選擇內。物理模擬仍使用 [[0071]] 的 ns 與頻率單位。

例外：ADC 解調頻率要有具名來源。具名 ADC config 的解調頻率是作者明示的物理 image，照原樣使用；同一來源的 WMEM／`r_wave` 修改與 sweep 也算具名。Window 起點生效的 ADC 頻率字沒有具名來源時，frontend 在編譯時拒絕，訊息寫明 ADC 與 trigger 時間，並請作者改用具名 ADC config，因為模擬器不知道 raw 字代表哪個 DDS image。Window 內與起點設定等值的更新不另解 image。絕對相位約定不變。2026-11-21，使用者在「保留 DDS 同餘類」與「raw 字編譯時拒絕」之間改選後者，答「S」；先前選保留的理由經回歸測試證實 table sweep 保留具名來源。比較見[raw ADC 頻率字改成編譯時拒絕](../discussions/2026-11-21-raw-adc-word-reject.md)。

## 依賴與驗證

沿用 [[0072]] 的分層與 `.importlinter` C8、C9、C12、C19。Program 的模擬子包與 experiment 使用 `simulate.boundary` 的資料契約；`simulate-closed` contract 禁止 simulate import 其他 zcu_tools package。TimeBase 只使用標準庫與 NumPy。Compiled frontend 的入口與 snapshot owner 維持 [[0083]]。

公開測試驗證兩組時鐘的整數週期、時間組合順序、scalar／array 的逐位正確捨入、非整數時基、格點拒絕及精確範圍邊界。文件由直接審閱確認，不以靜態文字斷言測試。
