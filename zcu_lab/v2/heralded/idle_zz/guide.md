**讀者：** operator、agent。用 herald 分組的 Ramsey 條紋估計 idle ZZ 頻差。

# Heralded conditional Ramsey 與 idle ZZ

每發做 **可選 Reset → mux herald → spectator I 或 Xπ → target π/2 → 自由演化 → target π/2 相位投影 → mux 終端讀出**。Herald 在 spectator 脈衝之前分類。Native 存檔保留 herald 與終端的所有聯合計數，不在取數時做 herald 篩選。

## 設定表單

- 核對 [qubit 位置與 mux tone 次序](../../two_qubit/guide.md#綁定位置與讀出-tones)，不需綁 coupler。**Ramsey target** 選做 Ramsey 的 qubit。兩顆的 Xπ 與 Xπ/2 引用各自 slot；同一顆用同一 drive channel，兩顆分用不同 channel。
- **Mux readout** 的兩個位置都需要 GE Post 的 circles 與半徑。Herald 與終端共用這份設定。用 `post_delay` 留出 herald 後的等待。**Reset** 留白時不發 reset，但仍做 herald。
- **Programmed Ramsey detuning** 要填非零值，絕對值大於預估 |ζ| 加 target 頻率誤差。第二個 π/2 的相位加上 `360° × detuning (MHz) × actual delay (µs)`，不改 RF。**Ramsey free-evolution delay** 從 0–10 µs、101 點起填，至少要六個不同且等距的實際延遲。最快條紋每週期留至少十點。
- **Coupler DC source** 留白時不動 DC。若掃 bias，將 **Coupler DC bias sweep** 用 **DC source unit** 所選的單位填寫。Run 結束或 Stop 不復原偏壓，也不補償其他 loop。

## 判讀頻差

按 **Analyze**。主圖把每個 bias 的實際 spectator g／e 分在兩欄。每欄用 gg 和 target herald e、spectator herald g 的一類擬合。後者在 Qubit1 target 是 eg，在 Qubit2 target 是 ge。先反轉該類的 target 對比。兩條共用頻率，分別擬合振幅、偏移、相位和衰減時間。

獨立診斷圖顯示 gg-only，以及 spectator herald e 的兩類。後兩類的實際 g 對應 Xπ，實際 e 對應 I。這三種擬合都不進主估計。Herald Other 留在資料，不進擬合。終端 Other 在對比分子計零，仍留在分母。未量測點與該類零發點都不畫成零。

ζ 是實際 spectator e 的有號頻率減去 g 的有號頻率。兩個頻率都需符合 [Idle ZZ 的同側條件](../../two_qubit/idle_zz/guide.md#決定是否採用頻差)。若頻率距離 0 或兩倍 detuning 絕對值不超過 `max(1 / 延遲窗長, 頻率 SE)`，該支不定符號。缺必要分類、缺延遲或頻率不能定符號時，保留可得的點與診斷，不用 gg-only 補 ζ。只缺診斷類不會擋住主估計。

量測點的誤差棒是逐發抽樣的 shot SEM。頻率的 ± 是 fit SE，ζ 的 SE 由兩支頻率 SE 平方和開根。用圖上的 R² 和正規化殘差 RMS 檢查各條曲線是否貼近量測點，這兩項不作為擬合門檻。掃 bias 時，優先複測 |ζ| 低且誤差可辨的點。這個估計不代表各 herald 類的製備污染相同，也不寫回設定。

共用操作見 [量測流程](../../operator-workflow.md)。
