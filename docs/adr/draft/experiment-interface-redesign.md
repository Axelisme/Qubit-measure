# 實驗核心、前端包裝與具名圖形產物

**狀態：** 設計方向已核准，尚未實作。本文是 ADR 草案，不取代現行 [實驗 workflow](../0062-experiment-workflow.md)、[保存](../0063-persistence-ownership.md)、[cfg](../0065-cfg-editing.md)、[operation](../0066-operation-lifecycle.md) 與 [GUI](../0067-gui-application.md) 契約。未定細節列於末節。

## 問題

目前 v2 實驗 class 同時提供 Notebook 的 last_result 便利行為、量測、分析與保存。GUI adapter 再補表單、參數轉換與統一介面。以 Notebook 呼叫習慣作為核心契約，造成 GUI 特別傳遞 cfg 外參數，也讓 Figure、數值分析與前端生命週期混在一起。

只改用函式、拆 package 或增加宣告式 definition，不能解決這些責任問題。目標是先分開核心與前端，保留使用者直接撰寫 Python 和原生 Matplotlib 的能力。

## 核准目標

### 核心與前端

核心保留無跨次可變狀態的實驗 class。Notebook 使用每個實驗自己的便利 class，GUI adapter 直接使用核心。兩前端不互相包裝，不新增共用 runner 或通用 Notebook adapter。

核心操作形狀為：

```python
run(config, *, context) -> Result
analyze(result, options, *, plots) -> Analysis
save(result, destination) -> None
load(source) -> Result
```

QICK context 先只有 soc、soccfg、plots，每次 run 由 caller 建立。單次 buffer、tracker 與 cache 歸 run。中斷、pbar、硬體互斥、重試與 GUI operation 沿現有機制，不以新 outcome 或 runner 取代。

影響 acquisition 的實驗選項全部進 typed config 與 cfg snapshot。核心擁有設定驗證及環境無關預設；GUI 擁有編輯表示、標籤、expression 與 md／module library seed。核心分析不讀 live GUI 狀態。

核心 options 與 Analysis 都是實驗專屬 typed 資料。Analysis 不含 Figure 或 writeback。Caller 記錄來源 Result 與實際 options。Notebook 扁平 keyword 引用核心預設並組成 options，不反射產生簽名、不共享可變預設。Notebook 不新增 writeback。

### 圖形產物與呈現

`plots` 是本次操作的繪圖能力，不是全域服務查詢器。

```python
fig, ax = plots.subplots("fit", ...)
viewer = plots.liveplot_1d("measurement", ...)
plots.adopt("diagnostic", external_fig)
```

以上入口建立或接收 Figure 時即具名登記。所有具名圖都是本次操作的圖形產物，不做第二次 register_result。實驗直接操作原生 Figure／Axes，factory 不包裝整套 Matplotlib。

不呈現環境仍建立與更新圖並可保存，只是不開視窗或安排互動展示。Liveplot 在執行期間呈現。第三方自行建圖須經 adopt 才保證接入，不能撤回其已發生的顯示副作用。

新入口不依賴 ambient plotting scope，也不要求實驗作者寫相關 with。Factory 必須明確接合 host；不能只把舊 plt.subplots 包一層，仍暗中依賴 routing 或每次切換全域 backend。

Adapter 掌握操作／session 使用期，共用 plots 實作處理圖形機制。停止 producer、關閉呈現與保留可保存的 Figure 是不同責任。圖已登記不表示操作成功，失敗診斷圖不得混入上一筆成功分析。

GUI application 持有結果與具名圖集合，保存不反向依賴 Qt widget。前端持有 canvas 與選圖狀態。截圖取目前選圖，分析圖保存涵蓋整個集合；不新增多圖排版編輯器。

### 三種分析生命週期

- **T1**：同步 analyze 正常返回後，caller 成組發布 typed Analysis、來源、options 與圖集合。
- **singleshot/ge**：核心另提供 post_analyze。Post 消費對應 primary 與其來源資料，不重新 primary fit，不讀未重新分析的 initial_state 表單；primary／post 各自持有圖集合與 GUI writeback proposal。
- **onetone/flux_dep**：核心提供領域分析操作，不持有 widget 或 GUI session。Notebook analyze 回傳實驗專屬互動控制物件，GUI 使用既有 INTERACTIVE plugin／session／frontend。啟動返回不終止繪圖；Done 驗證並發布最終結果，Cancel 不取代上一筆成功分析。兩前端共用領域計算與 typed 結果，不強制共用互動框架。

Notebook run／load 正常返回後更新 last_result 並清空目前分析；失敗保留舊紀錄。同步 analysis 成功才成組替換分析與圖，明確分析舊 Result 不改 last_result。清空引用不銷毀使用者另行持有的圖或結果，不新增完整歷史管理。

### 保存、發現與範圍

實驗擁有 Result 到 canonical 資料的映射，caller 指定路徑；save 不依 last_result。分析圖保存另由 application／Notebook 管理，不改資料格式或 fitting 演算法。

保留 explicit catalog、v2／v2_gui 分離，以及單檔／package 並存。框架不要求固定內部檔名，不靠掃描或 import 副作用發現實驗。

本次遷移涵蓋一般 T1、singleshot/ge、onetone/flux_dep 與必要的 GUI 多圖接線。其他實驗核心、executor、動畫不因這項決策全面遷移。未遷移 adapter 的既有單圖輸出需在明確接縫轉換，不能反射猜測新舊格式。

## 取捨

- 明確 factory 使建圖依賴可讀，但任意第三方 pyplot 操作不再自動路由；adopt 處理返回的 Figure，不提供任意 pyplot 全域狀態隔離。
- 圖獨立於 Analysis，避免數值型別依前端改變，代價是 caller 必須把資料、來源與圖一起發布。
- Notebook 專屬類保留少量扁平參數映射，換取熟悉入口與型別提示；不以通用 adapter 或動態簽名消除此映射。
- 不呈現仍建圖，保留實作一致性與保存能力，接受建圖及 artist 更新成本。
- 不以單一同步方法統一互動 session；通用繪圖不保證自動提供互動輸入。

## 七項待落實接縫

| 編號 | 缺口 | 方案需回答 |
| --- | --- | --- |
| G1 | 明確 GUI factory | 如何接入 canvas，如何序列化 worker artist 修改與 GUI draw，而不依賴 ambient scope |
| G2 | 圖形生命週期 | 最後 refresh、Notebook 自動顯示、失敗清理及返回後保存 |
| G3 | 名稱與接管 | 同名、同圖多名、跨活躍 owner adopt 的拒絕／轉移規則 |
| G4 | 多圖保存 | 安全命名、集合版本、部分失敗與重試，不假稱跨檔原子提交 |
| G5 | 晚到結果 | 來源 generation／request identity 如何防止舊 session 或 post 覆寫目前分析 |
| G6 | 無互動前端 | 無呈現時如何明確拒絕互動啟動，或接受明確 selection，不以 seed 偽造完成 |
| G7 | 未遷移 adapters | 單圖與舊核心簽名如何在明確接縫轉換，不引入探測 fallback |

### 調查方案，尚待細節確認

G1 建議將一般圖與 liveplot 分開接線。一般 Figure 由 worker 獨占建立與修改，先具名登記，完成後才在 GUI owner attach。Liveplot 由 GUI owner 持有 artists，worker 的 typed update 傳遞獨立資料，由 owner 執行共用 segment 更新與 draw。這不要求實驗寫前端分支，但 active liveplot 不保證 worker 任意直接修改原生 artists 安全。若必須支援任意 mutation，需評估 worker-owned Agg frame 等不同呈現方案，不能只加 draw lock 就宣稱解決。

G2 建議停止 producer、處理已接受的最後 refresh、封存圖集合、釋放呈現資源分開。取消請求不立即封存；既有 partial 正常返回仍可呈現最後資料。封存後保留的 Figure 不由下一次操作清空。Notebook 以明確 canvas／display 避開 pyplot 的自動顯示清單；不呈現使用非互動 canvas。close 是否影響 widget 與保存依 backend 而定，不全面禁止 close，也不使用全域 close("all") 清理別人的圖。外部 Figure 的 adopt 需處理原 manager，不能承諾撤回已顯示內容。

G3 建議一名一圖：同名同物件重複 adopt 等冪，同名異物件及同物件異名拒絕；跨 owner 不偷移 canvas。所有檢查在 attach 前完成。第一版不新增 transfer 或 alias 入口；舊圖可由原集合保存，跨操作呈現可由資料重畫。這些是待凍結的名稱／所有權政策。

G4 建議以 stage 與圖名形成 ArtifactKey，逐項記錄 captured generation／path 的保存狀態。SaveService 固定本次集合、Figure 參照與目的地，沿 owner-thread export 接縫保存；前項成功、後項失敗時保留成功項，不宣稱整組已保存。重試應能只選未完成項，避免重送已成功 DATA 而重新產生路徑。命名建議一律包含 stage 與圖名，並在 I/O 前驗證安全路徑與名稱碰撞；確切格式及是否將一般 Save All 改為 dirty-only 仍待確認。不承諾跨檔原子保存，不把單圖 Save 推定為覆蓋原始資料的授權。

G5 的候選方案沿用 GUI 的 token／版本及 owner-loop；Notebook 以局部 generation 記錄接受資格。舊控制物件可保存自己的最終結果，但不回寫新的目前分析。是否禁止重疊操作仍需凍結。

G6 的候選方案由前端在互動啟動前檢查能力，無前端則拒絕；純領域 finalization 可以接受明確 selection。這不新增無頭互動 workflow，也不把 display=False 當作自動 Done。

G7 建議 GUI application 統一使用純資料結果與 request／session 所持有的 plots。未遷移 FIT adapters 在自身分析方法中明確 adopt 原核心返回的圖；兩個 interactive adapters 的共同 plugin／frontend 接線一併轉成 session-owned 圖集合。三個新核心的 concrete GUI adapters 明確轉接 run／save／load，不探測 BaseAdapter 的新舊簽名，也不遷移其他核心。不要新增 extract_legacy_figures 作為永久第二套輸出協議。

目前 catalog 為 46 項，其中 41 FIT、2 INTERACTIVE、3 NONE。三個指定核心對應 3 項 adapter；剩餘 39 FIT 與 1 INTERACTIVE 仍需必要的 GUI 圖形接線，3 NONE 需核對共用型別。這是共用 GUI 多圖的影響面，不把 twotone/flux_dep 誤列為 onetone 的第二個核心。

上述細節為調查方案，不是已生效保證。現有 Notebook close 與 GUI bridge 限制仍見 [liveplot](../../../lib/zcu_tools/plotting/liveplot/README.md) 和 [GUI plotting](../../../lib/zcu_tools/gui/plotting/README.md)。

## 轉正為現況的條件

完成三個實驗的 Notebook／GUI 接線，以及 G1–G7 的具體契約與驗證。以公開 seam 驗證 cfg snapshot、typed analysis、primary／post 來源、互動 Done／Cancel、具名多圖與保存失敗；以直接審閱確認責任、依賴與匯出。

不得用一般 T1 通過推定 interactive 或 post-analysis 已驗收。Notebook inline／widget 顯示與 Qt thread 接合需有對應觀察，不以數值測試代替。驗證完成後才將已落實部分寫入現行 ADR 與 module README。
