---
date: 2026-11-24
status: active
superseded_by: null
related:
  - 2026-11-22-adc-raw-plan-acquire-owner.md
  - 2026-11-22-simulator-speed-first-verification.md
basis:
  repo: 4a1d3228d0d875f70999254d87fc674a5d227ecd
  external: null
---

# ADC raw 停線只放行一次 flat 跳轉試驗

## 結論

准試一次：多態 ADC 的 integrated raw 在不回傳 trace、所有載波皆為 flat 且實際跳轉時，只計算來源與目的兩態的基線場。原公式、載波疊加順序與腔記憶更新不變。這是 raw 內優化停線的單一例外，不重開其他 kernel 或統計量直算。

Designer 依 2026-11-24 夜間授權代裁：「方向問題也授權你代裁，land也可以自己決定，優化任務也是有利就做，都不要等我，這是浪費時間」。這是代裁授權，不是使用者對本原型的回答。例外仍待使用者覆核。

試驗通過。[嘗試紀錄](../simulator-attempts/2026-11-24-adc-flat-jump-baselines.md)記下 `chain_freq` 與 `chain_power` 的 core 分別下降 18.36% 和 18.39%，Result 與 shot moments 最大絕對差為 0。兩側為 `addd6c8fb` 與 `5e43af2c7`，同離線 fixture、seed 123、同 CPU slot、單 worker、各冷 cache，每個場景每側一次完整 11 格 Run。另以同一對 commit 從公開 ADC 接縫測合成多態雙載波跳轉，固定 sampling identity 的 256 個 int64 值逐個相同。

## 問題

2026-11-22 使用者對 raw 內優化停線、統計量在本階段驗證的選項答「B」。後來精確四階矩試驗超過核准的成本上限，依當時裁決停止。這次 `chain_freq` 與 `chain_power` 仍以 materialize 為最大分項。兩場的 profile 都指到跳轉時的 baseline 求值；每個跳轉解所有狀態，腔記憶的更新卻只讀兩態。調查沒有找到 ADC 以外能省掉同一重工的路徑。

## 比較過的選項

- **維持停線。** 不改程式，有定位結果但不減少等待。一次窄範圍試驗若未證收益，就回到這條，不留下旗標或備選 kernel。
- **只試 flat 跳轉的兩態基線。** 每個跳時使用原公式求兩態，不複製公式、不加快取。比較時先驗同 identity 的 raw int64 逐值相同，再量兩次完整前後配對。兩場各自須 core 降至少 5%、materialize 同向下降、shot count 不變、最大 Result／shot moments 差不超過 `1e-2`。任一對不等價、不可比或不足收益就捨棄。5% 是本次採用門檻，不是通用速度規則。
- **先算 CDF 再擇門檻。** 有機會省掉逐 shot 累積，但狀態選擇由 raw 和 trace 共用。要另驗 draw 順序與稀疏 rows 的準備代價。這次不做，也不當第二輪試驗。
- **Sums-only caller 直接取得統計量。** 仍須保留單發 IQ 的 caller。改走統計量會改 Program 與環境交接，而且沒有更快的精確算法，不能用只保留事件邊際的 compact 結果代替逐發配對。維持先前停止條件。

## 留下的問題

本次只證明兩個指定場景受惠。兩場仍由 materialize 主導，core 仍約 5 s／格，遠高於 0.1 s／格的目標。其他 raw caller 未量，不因共用 ADC 而推定相同收益。

## 依據

[ADR-0078](../adr/0078-evolution-module-boundary.md) 與 [ADC raw plan 的 owner 討論](2026-11-22-adc-raw-plan-acquire-owner.md) 的限制維持不變：raw plan 壽命、RNG、Stop 與發布不動。相同物理的 projected prefix 是技術先例，省的是格點間的傳播，這次省的是跳時場。它不是這次越過停線的授權。審查也要求把 raw 的條件寫全，因為 trace 和 window 同樣需要跳時腔記憶。本案保留它們的原路，並以公開接縫對照。原調查、挑戰與計時原檔未入庫；數字與輸入範圍保存在前述嘗試紀錄。

## 後記

使用者在 2026-11-24 覆核這項裁決時說：「我没有要求不再做啊，我的要求真的就只有可维护性，还有执行的效率而已。那其他都只是当下环境做的暂时性裁决而已，那不随时是可以推翻的。」

使用者追認本次 flat 跳轉試驗。先前暫停 raw 內優化與統計量直算是當時的裁決，不是使用者的要求。本試驗不再是待覆核的局部例外。
