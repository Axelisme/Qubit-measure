**讀者：** operator、agent。按 probe 前實測的雙 qubit 類別找 CZ 候選頻率。

# Heralded conditional spectroscopy

**序列：** 可選一般 Reset → 名目 I／Xπ 準備 → mux herald → 等待 → 同頻雙線 probe → 可選雙 Xπ 對比 → mux 終端讀出。兩顆同時做名目準備，依 Qubit1、Qubit2 的次序各取 I 或 Xπ。Herald 在準備之後，因此 gg、ge、eg、ee 代表實測類別，不是名目準備。四種準備讓熱布居重新分配，目標是讓四類取得相近的 shot 數；實際比例取決於準備與讀出。

## 設定表單

- 核對 [qubit 位置與 mux tone 次序](../guide.md#綁定位置與讀出-tones)，兩顆都要有校準的 Xπ 和同一讀出條件下的 GE Post 圓形分類器。Herald 與終端共用完整的 **Mux readout** 與 `post_delay`。從 `5/(2π·κ)` µs 開始，κ 取讀出 tones 中最小的 linewidth，詳見[等待的校準方法](../guide.md#herald-reset-與獨立-heralded-入口)。**Reset** 留空就略過，只接受一般 reset。不要再加一層 herald reset。
- **Two-qubit spectroscopy probe** 不是準備用的 Xπ。兩線用各自的 qubit generator，包絡長度與 pre-delay 相同。若用 coupler 的 CZ slots 當初值，先降到探譜所需的 gain，不照搬 CZ 功率。**Common probe RF frequency** 同時覆蓋兩路 RF；若打開 **Double Xπ contrast**，probe 後依序翻轉 Qubit1、Qubit2，終端 P(e) 仍是讀出時的值。
- **Coupler DC source** 留空不動 bias，不要求 `coupler` alias。掃 bias 時填裝置、單位和掃描範圍。掃描只改選定來源，完成或 Stop 都不復原。

## 讀圖與選下一步

- **Terminal P(e)** 每個 herald 分支各有兩顆 qubit 的曲線。每個 bias、頻點、分支先合計四種名目準備的 shots，再用終端 e 數除以終端 g、e、other 的總數。棒是這批保留 shots 的逐點 shot SEM；少於兩發不畫棒。Live 另顯示已取樣的準備／頻率格數。
- **Retained shots by herald** 顯示每個頻點各分支的保留數。0 是有取數但沒落在該 herald 類別，缺點是四種準備都未取數。任一 tone 判為 other 的 herald 不進四支曲線，但仍保存。
- **Herald class fractions** 每列是一種名目準備。合計該 bias 已取樣頻點的 shots，再算九種 herald 類別的比例。用這張圖看名目準備和實測分類如何對應；它不是純度、gate fidelity 或 leakage 的量測。
- 多 bias 時，用每顆 qubit 的 **Terminal P(e) map** 追蹤 RF × DC 上的峰谷。四個 panel 共用該圖的色階。

用離開掃描邊緣的峰谷當窄掃中心，保留兩側基線。比較 Double Xπ 開關時，對比變號不代表多一條線。此入口不擬合峰位、不寫回。Save 後用同一入口 Load；Analyze 只讀該次 cfg 和分類計數，不讀目前表單。通用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。
