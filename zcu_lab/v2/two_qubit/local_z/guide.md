# Local Z

**讀者：** operator、agent。量兩顆 qubit 每次 CZ 的 local phase，取得反號 correction pair。

同一 Run 分別用兩顆作 target。每發做 **可選 Reset → target Xπ/2 → n 次雙線 CZ → X 或 Y projection → mux 終端讀出**；另一顆不打準備脈衝。X、Y 使用獨立 shots，投影相位相對準備為 0° 和 90°。

## Run 前

- 核對 [位置與完整 mux 次序](../guide.md#綁定位置與讀出-tones)、校準的 Xπ/2 與目標 tones 的 GE circles、GE Post 半徑。選定 CZ amplitude、frequency 後再量 local Z。**Dual-line drive** 取自 coupler 的 `cz_qubit1` 和 `cz_qubit2` slots。兩路 CZ 用不同 generator，保持共用載波、包絡長度與 pre-delay；各 Xπ/2 用自己的 CZ generator。
- **CZ gates per shot** 填至少三個連續遞增非負整數，預設 [0,1,2,3,4]；n=0 不發射 CZ，用作基線。預期相鄰相位增量須小於 180°，每 gate 加上整圈無法辨識。
- **Reset** 留空不發射。若選 mux herald，見[讀出與等待](../guide.md#herald-reset-與獨立-heralded-入口)；只用通過篩選的終端 shots。
- Recipe `two_qubit_local_z`：省略 `gate_count`、`reps`、`rounds` 保留表單值，明填 null 交給表單。若 overrides 後仍缺必填值，在 Run 前的 `check_ready` 失敗。[共用輸入](../../../recipes/two-qubit-guide.md#共用輸入)另說明綁定、tab 重用和 overrides。

## Phase 與 slope

- **Run 與 quadratures**：按 target 看 X/Y 的 `σ=(N_e−N_g)/N_total` 對 n；other 記零並留在分母。棒是保留 shots 的樣本標準差除以 √N，不是 fit SE 或輪間散布。Run 顯示已完成點；沒有有效投影時不產生 phase。
- **Phase points** 比較向量長度（contrast）與 `atan2(σ_Y,σ_X)` 的 wrapped phase。兩投影都須完成必要取數，保留數與 contrast 都不可為零。微弱但非零的 contrast 仍可能給 fit，先對照 X/Y 棒。
- **Phase** 顯示相鄰 unwrap 後的角度與計數加權直線 `intercept + slope × n`。Slope 是每 CZ 的 logical Bloch phase；缺點、半圈歧義或無正值 phase variance 不給該 correction。± 為 slope fit SE；R² 為解釋的相位變異，normalized RMS 是殘差 RMS／實測相位範圍，relative SE 是參數 SE／參數絕對值，沒有額外選取門檻。GUI 有 Run／Load capture 時可列元件名，位置仍是 Qubit1、Qubit2。

## Preview 與 Apply

`correction = −slope`（度，不再 wrap）。只有兩分量都有效才提案寫一筆 `$coupler.ext.cz_phase`，含有序元件名及兩角；不改脈衝或 slot。後續 virtual Z 更新 logical frame，並核對配對名稱與順序。修改 CZ gain、frequency、duration、waveform、relative phase 或 DC 工作點後重測。熱 ensemble 的 slope 不是純態 phase 或 fidelity。

分析用來源 cfg；舊檔見[家族載入說明](../guide.md#opendefaults-與人工-apply)。共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。
