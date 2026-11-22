**讀者：** operator、agent。用兩個 spectator Ramsey 分支估計 idle ZZ 的頻差。

# Conditional Ramsey 與 idle ZZ

每發做 **可選 Reset → spectator I 或 Xπ → target π/2 → 自由演化 → target π/2 相位投影 → mux 終端讀出**。未 reset 時，spectator 是兩個熱 ensemble 上的名目分支，不是純 g／e 態。

## 設定表單

- 核對 [qubit 位置與 mux tone 次序](../guide.md#綁定位置與讀出-tones)；不需綁 coupler。**Ramsey target** 初選 Qubit1。兩顆的 Xπ 與 Xπ/2 從各自的 `x180`、`x90` slot 複製，缺 slot 留空。同一顆的脈衝須在同一 drive channel、兩顆分用不同 channel。用校準脈衝與完整 **Mux readout**，兩個位置須有同讀出設定下的 GE circles 與 GE Post 半徑。
- **Programmed Ramsey detuning** 留白待填。第二個 π/2 的相位是第一個的相位加 `360° × detuning (MHz) × actual delay (µs)`，不改 RF。取絕對值大於近期 Ramsey 頻率誤差與預估 |ζ| 之和的非零 detuning，確保兩分支同頻率符號側。
- **Ramsey free-evolution delay** 從 0–10 µs、101 點起填，需至少六個不同且等距的實際延遲。最快 fringe 每週期至少 10 點（2 MHz 時步距不超過 0.05 µs）；若要看衰減，窗口取約 3×T2*。太短只看到部分振盪或衰減，fit 得外推；太疏會把快速條紋看成低頻。
- **Coupler DC source** 留空不動 DC；若掃 bias，用 **DC source unit** 選 A、V 或 native，並以同單位填 **Coupler DC bias sweep**。不補償其他 loop，結束或 Stop 不復原偏壓。**Reset** 留空不發射；若選 mux herald，核對[讀出與等待](../guide.md#herald-reset-與獨立-heralded-入口)。

## 讀 Run 與 Analyze 圖

- **Ramsey contrast**：各 bias 比較 spectator I／Xπ 兩色的 `(N_e−N_g)/N_total` 對實際延遲。Other 單發記 0 並留在分母。棒是保留 shots 的樣本標準差除以 √N，少於兩發不畫；不是 fit 誤差或輪間散布。Run 的進度與保留數可區分無資料和零對比。
- **Analyze** 疊兩條衰減餘弦，任一分支缺必要延遲或無保留 shots 不給頻率。頻率旁的 ± 是 fit SE；R² 是解釋的起伏比例，normalized RMS 是殘差 RMS／實測峰谷差，相對頻率誤差是頻率 SE／|f|。不以這些量設自動門檻。
- 多 bias 的 **Conditional fringe shift** 比較 ζ；棒為兩分支頻率 SE 平方和開根，不是 shot SEM。

## 決定是否採用頻差

兩分支同側時，`f = sign(detuning) × |f|`，`ζ = f(Xπ) − f(I)`，共同人工相位斜率相消。若 |f| 到 0 或 2|detuning| 的距離 ≤ `max(1/實際時間跨度, 頻率 SE)`，Analyze 不報 signed f 或 ζ。先依圖上原因補窗口、保留取樣密度，或在同側前提不明時重校 target RF／調 detuning。熱混合可能讓單頻模型失配；這是名目分支頻差，不等於純 Hamiltonian ζ。掃 bias 時複測低 |ζ| 且誤差可辨的候選；沒有 writeback。

舊檔見[家族載入說明](../guide.md#opendefaults-與人工-apply)。共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。
