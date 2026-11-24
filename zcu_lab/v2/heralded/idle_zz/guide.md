**讀者：** operator、agent。用 herald 分組的 Ramsey 條紋估計 idle ZZ 頻差。

# Heralded conditional Ramsey 與 idle ZZ

每發做 **可選 Reset → mux herald → spectator I 或 Xπ → target π/2 → 自由演化 → target π/2 相位投影 → mux 終端讀出**。Herald 在 spectator 脈衝之前分類。Native 存檔保留 herald 與終端的所有聯合計數，不在取數時做 herald 篩選。

## 設定表單

- 核對 [qubit 位置與 mux tone 次序](../../two_qubit/guide.md#綁定位置與讀出-tones)，不需綁 coupler。**Ramsey target** 選做 Ramsey 的 qubit。兩顆的 Xπ 與 Xπ/2 引用各自 slot；同一顆用同一 drive channel，兩顆分用不同 channel。
- **Mux readout** 的兩個位置都需要 GE Post 的 circles 與半徑。Herald 與終端共用這份設定。用 `post_delay` 留出 herald 後的等待。**Reset** 留白時不發 reset，但仍做 herald。
- **Programmed Ramsey detuning** 要填非零值，絕對值大於預估 |ζ| 加 target 頻率誤差。第二個 π/2 的相位加上 `360° × detuning (MHz) × actual delay (µs)`，不改 RF。**Ramsey free-evolution delay** 從 0–10 µs、101 點起填，至少要六個不同且等距的實際延遲。最快條紋每週期留至少十點。
- **Coupler DC source** 留白時不動 DC。若掃 bias，將 **Coupler DC bias sweep** 用 **DC source unit** 所選的單位填寫。Run 結束或 Stop 不復原偏壓，也不補償其他 loop。

## 判讀頻差

- gg 與 target 初態 e、spectator 初態 g 的一類進主估計。後者先反轉 target 對比：target 是 Qubit1 時用 eg，是 Qubit2 時用 ge。
- Spectator 初態 e 的兩類各自對照，不進主估計。依 spectator 實際 g／e 態交換其 I／Xπ 分支。Herald 的 Other 類留在資料，不進擬合；終端 Other 在對比分子計零並留在分母。
- 每個 bias 的 ζ 是實際 spectator e 的 signed Ramsey 頻率減去 g 的頻率。沿用普通 [Idle ZZ 的同側條件](../../two_qubit/idle_zz/guide.md#決定是否採用頻差)。缺必要分類或延遲時，主估計留空，不用 gg-only 頻差補值。
- 量測點的誤差棒是 shot SEM；頻率的 ± 是 fit SE。ζ 的 SE 由兩個實際 spectator 態的頻率 SE 平方和開根。掃 bias 時，優先複測 |ζ| 低且誤差可辨的點；沒有 writeback。

共用操作見 [量測流程](../../operator-workflow.md)。
