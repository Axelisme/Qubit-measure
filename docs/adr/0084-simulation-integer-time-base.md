---
status: accepted
---

# ADR-0084 — 模擬 request 的公共整數時基

## 脈絡

Compiled simulation 的 request 目前以浮點微秒傳遞時間。Frontend 與 consumer 各自加總時間、建立 ADC 取樣網格。同一時間以不同運算順序組合，結果會差一個 ulp。固定 seed 不能消除這個差異。

各時鐘的週期不一定是 tProc 週期的整數倍。例如 mock 的 tProc、generator fabric、ADC output、envelope sample clocks 分別為 430.08、384、307.2、6144 MHz。只改用 tProc tick 會遺失其他時鐘的格點。

## 決策與落實狀態

2026-10-31，使用者提出「我傾向根除的解決方案」，並選擇「2b」。採用第一階段的公共整數時基 request。第二階段的物理建模留待後議。

`core.simulation.TimeBase` 擁有公共有理數頻率與換算規則。Program 從 soccfg 擷取本次 tProc、已宣告 generator 的 fabric／envelope sample clocks 與已宣告 ADC 的 output clocks，再交給 TimeBase。未宣告 channel 不參與，TimeBase 屬於單一 request。Core 不讀 soccfg，也不 import driver。額外宣告不影響既有事件的異質 clock channel 可以改變時基，但不能改變原 channel 的物理時間與模擬輸出。Frontend 的 ns 網格不變性已納入驗證，物理 IQ 不變性待 consumer 遷移後驗收。

TimeBase 已提供建立時基、格點檢查與 scalar／array 換算。Request schema 與 compiled frontend 已使用整數時間。Consumer 與 replay 仍待遷移，本篇不表示整條遷移已完成。

## 時間契約

TimeBase 的頻率為約分後的有理數 `F = num / den`，單位為 ticks/μs，也就是 MHz。一個 tick 為 `den / num` μs。

- 每個時鐘以 `Fraction(repr(clock))` 建立有理數，並驗證轉回 float 不變。公共頻率取各有理數時鐘的最小公倍數。空時鐘集合、非正數、非有限數或失真轉換報錯。
- Request 攜帶 TimeBase。Start、duration、trigger、phase reference、envelope 取樣週期及 ADC 取樣週期都用整數 tick 表達。Frontend 全程做整數運算，不先轉成浮點微秒再組合。
- 手建 fixture 也必須落在格點上。非格點時間報 `ValueError`，不四捨五入回格點。需要非十進位的精確時間時，caller 可提供 Fraction。
- `TimeBase.ns` 是唯一的 tick→ns 換算入口。它支援 scalar 與 int64 ndarray，array shape 不變。每個結果都等於有理數 `ticks * 1000 * den / num` 正確捨入一次所得的 float64。
- ADC 網格先在整數域組成 `trigger + k * period`，再交給 ns。Readout plan 與環境 template 呼叫這個入口，不各自展開換算迴圈或浮點乘除。
- Num 必須能在 float64 精確表示。Request 的時間輸入與中間結果須在 int64 範圍內，另限制 `abs(ticks * 1000 * den) <= 2**53`。超出範圍報 `OverflowError`，不截斷、不溢位回繞。
- Relax delay 沿用 QICK 真正編譯時的 tProc-cycle 捨入，再映射到公共整數時基。它不以 raw 作者參數套用手建 fixture 的格點限制。
- `program.shot_time.request_round_time_s` 擁有整數 request 的 round 時長公式，binding 只呼叫這個入口。它用 Python 整數加總各點、relax delay 與 reps，再由 `TimeBase.seconds` 以有理數換成秒。整輪總和不套用單一 request 的 int64 或 ns 精確界限，不因總時長而新拒絕長 acquisition。2026-10-31，advisor 依代理授權裁定此 owner 與範圍。既有 us API 與硬體路徑不變。
- Replay 明確保存時基的 num／den 與整數時間，並升級格式版本。舊版本明確拒絕，不做相容轉換。

Mock 時鐘的公共頻率為 43008 MHz。對應的 tProc、fabric、ADC、envelope 週期為 100、112、140、7 ticks。ADC output 為 76.8 MHz 的 FTF 配置，其 ADC 週期為 560 ticks。

## 取捨與範圍

保留浮點時間並調整加總順序，只能固定某一條計算路徑。讓每個 consumer 自行量化，則會把格點規則散在多個模組。本決策改變 request 格式，接受重寫 fixture 與拒絕舊 replay 的成本，讓時間的組合順序不再影響換算結果。

本階段不改變物理模型。平移不變、整數 DDS 相位、ADC 相對動力學，以及第一個樣本對齊 ADC clock 的建模，都不在本次選擇內。物理模擬仍使用 [[0071]] 的 ns 與頻率單位。

## 依賴與驗證

沿用 [[0072]] 的分層與 `.importlinter` C8、C9、C12、C19。Core 僅依賴標準庫、外部函式庫與 utils，program 與 experiment 使用 core 的資料契約。本決策不新增跨 package import 邊，也不新增 import-linter contract。Compiled frontend 的入口與 snapshot owner 維持 [[0083]]。

公開測試驗證兩組時鐘的整數週期、時間組合順序、scalar／array 的逐位正確捨入、非整數時基、格點拒絕及精確範圍邊界。文件與 replay 版本政策由直接審閱確認，不以靜態文字斷言測試。
