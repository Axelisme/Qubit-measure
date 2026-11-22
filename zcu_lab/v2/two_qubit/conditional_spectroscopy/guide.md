**讀者：** operator、agent。從四種名目準備的終端頻譜為後續 CZ 選窗。

# Conditional spectroscopy

每發做 **可選 Reset → 名目 g,g／g,e／e,g／e,e 準備 → 同頻雙線弱 probe → 可選雙 Xπ 對比 → mux 終端讀出**。標籤按 Qubit1、Qubit2；g 用 I、e 用 Xπ，雙翻轉準備同時開始。沒有 reset 時仍有熱布居。

## 設定表單

- 核對 [qubit 位置與 mux tone 次序](../guide.md#綁定位置與讀出-tones)；不需綁 coupler。兩顆 **Xπ** 從各自 `x180` slot 複製，缺 slot 留空。用同讀出條件下的 GE circles 與 GE Post 半徑，保留完整 **Mux readout**。
- **Two-qubit spectroscopy probe** 不等於準備用 Xπ。若有 coupler 的 `cz_qubit1`、`cz_qubit2` slots，表單以它們為起點；否則給 cosine 封包與各 qubit 的 channel，另填 gain、長度及 NQZ。兩線與各自的 Xπ 同 channel，包絡長度和 pre-delay 相同，帶號 gain 與 phase 可不同。調為弱 probe，不照搬 CZ 功率。
- 填 **Common probe RF frequency** 起訖 (MHz)；每個頻點同時覆蓋兩路 RF，其他 pulse 設定不變。圖的頻率是掃描要求值，不是另量的 RF。**Double Xπ contrast** 預設開，probe 後依序打 Qubit1、Qubit2 Xπ；關閉則直接讀出。
- **Coupler DC source** 留空不動 DC。若掃 bias，用 **DC source unit** 選 A、V 或 native，並以該單位填 **Coupler DC bias sweep**。不補償其他 loop，結束或 Stop 不復原。Reset 留空不發射；若選 mux herald，見[讀出與等待](../guide.md#herald-reset-與獨立-heralded-入口)。

## 讀 Run 與 Analyze 圖

- **Terminal P(e)**：各 bias 的四個準備 panel 各有兩顆 qubit 的終端 P(e) 對 RF 曲線。分母含保留 shots 的 g/e/other，不回推雙 Xπ 之前的布居。用峰谷位置選窗。Run 與 Analyze 的一維頻譜以折線連接實測點，誤差棒用同頻點保留 shots 算出 P(e) 的 shot SEM，表示該點機率估計的抽樣不確定度，不是 fit 誤差或輪間散布。少於兩發不畫棒。Run 顯示已取樣的準備／頻率點及保留數。
- 多 bias 的 **Terminal P(e) map**：每顆 qubit 各有 RF × DC map，四個分支共用 0–1 色階，用亮暗帶追蹤頻線；不畫逐格誤差棒。
- **Retained shots**：比較各準備的保留計數與 RF；0 代表取得資料但沒有 shot 通過，缺點代表未取數。不要把空白當成零激發。

## 選下一次掃描

比較同一 qubit 在各準備下的峰谷。以候選採樣頻率為窄掃中心，保留峰谷兩側基線；若變化還在邊界，先延窗。可固定其他設定對照有無 Double Xπ contrast，不把圖形反轉當成新躍遷。這是分類後的熱 ensemble 光譜，不單獨辨識高能階、leakage、Hamiltonian 參數或 CZ frequency；沒有峰擬合和寫回。

舊檔見[家族載入說明](../guide.md#opendefaults-與人工-apply)。共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。
