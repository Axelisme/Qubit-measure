# 實驗核心、前端包裝與具名圖形產物

**狀態：** 設計方向已核准，端到端遷移尚未完成。本文是 ADR 草案，不取代現行 [實驗 workflow](../0062-experiment-workflow.md)、[保存](../0063-persistence-ownership.md)、[cfg](../0065-cfg-editing.md)、[operation](../0066-operation-lifecycle.md) 與 [GUI](../0067-gui-application.md) 契約。未定細節列於末節。

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

核心 options 與 Analysis 都是實驗專屬 typed 資料。Analysis 不含 Figure 或 writeback。Notebook caller 記錄來源 Result 與實際 options。Notebook 扁平 keyword 引用核心預設並組成 options，不反射產生簽名、不共享可變預設。Notebook 不新增 writeback。

GUI 插件自行定義 typed 成功輸出的欄位。插件決定是否輸出選項、隨機 seed、時間或其他重現資訊。Framework 保存來源、插件輸出與圖，不追查插件的隱式依賴，也不保證完整 options 或可重現性。`params` 保持表單輸入，不在 Done 時替換成終態 options；不新增通用 committed-options owner。GUI 與 remote 讀同一份已提交輸出。

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

- **T1**：同步 analyze 正常返回後，Notebook caller 成組發布 typed Analysis、來源、options 與圖集合；GUI 發布 typed 分析輸出、來源與圖集合。
- **singleshot/ge**：核心另提供 post_analyze。Post 消費對應 primary 與其來源資料，不重新 primary fit，不讀未重新分析的 initial_state 表單；primary／post 各自持有圖集合與 GUI writeback proposal。
- **onetone/flux_dep**：核心提供領域分析操作，不持有 widget 或 GUI session。Notebook analyze 回傳實驗專屬互動控制物件，GUI 使用既有 INTERACTIVE plugin／session／frontend。啟動返回不終止繪圖；Done 驗證並發布最終結果，Cancel 不取代上一筆成功分析。Notebook 保留實際 options 紀錄；GUI 插件決定成功輸出的欄位，不要求完整終態 options。兩前端共用領域計算與 typed 結果，不強制共用互動框架。

Notebook run／load 正常返回後更新 last_result 並清空目前分析；失敗保留舊紀錄。同步 analysis 成功才成組替換分析與圖，明確分析舊 Result 不改 last_result。清空引用不銷毀使用者另行持有的圖或結果，不新增完整歷史管理。

### 保存、發現與範圍

實驗擁有 Result 到 canonical 資料的映射，caller 指定路徑；save 不依 last_result。分析圖保存另由 application／Notebook 管理，不改資料格式或 fitting 演算法。

保留 explicit catalog、v2／v2_gui 分離，以及單檔／package 並存。框架不要求固定內部檔名，不靠掃描或 import 副作用發現實驗。

最終範圍是所有實驗及其 Notebook／GUI caller，不以現有 GUI catalog 為上限。先以一般 T1、singleshot/ge、onetone/flux_dep 驗證同步、post 與互動分析，再分批遷移其餘實驗及必要的共用 helper。

過渡期間允許尚未遷移的實驗因舊契約而報錯，不為了保持它們可用而加入 pyplot 相容層。報錯須對應到未遷移項目，不能忽略已遷移路徑的 regression，也不能把過渡狀態當成最終交付。這不授權重寫 executor 或動畫框架，既有 acquisition、排程與硬體鎖機制保持。

## 取捨

- 明確 factory 使建圖依賴可讀，但任意第三方 pyplot 操作不再自動路由；adopt 處理返回的 Figure，不提供任意 pyplot 全域狀態隔離。
- 圖獨立於 Analysis，避免數值型別依前端改變，代價是 caller 必須把資料、來源與圖一起發布。
- Notebook 專屬類保留少量扁平參數映射，換取熟悉入口與型別提示；不以通用 adapter 或動態簽名消除此映射。
- 不呈現仍建圖，保留實作一致性與保存能力，接受建圖及 artist 更新成本。
- 不以單一同步方法統一互動 session；通用繪圖不保證自動提供互動輸入。
- 接受未遷移實驗在中間階段報錯，避免維護第二套繪圖或分析協議；代價是必須逐項追蹤遷移與驗證，不能只用三個標準實驗通過推定整批完成。

## 七項核准政策與待落實接縫

下列政策已核准，部分共用繪圖與保存能力已落實，但完整 caller 遷移尚未完成。末節列出接線與驗證義務，不能將政策核准或底層能力通過視為端到端保證。

### G1：一般圖與 liveplot

一般 subplots 立即建立並具名登記，GUI 於函數完成後呈現。Liveplot 立即呈現，worker 透過 typed update 傳遞資料，GUI owner 更新 artists。Worker 不任意直接修改 active live Figure／Axes。實驗不需寫前端分支，也不能以 draw lock 取代這項執行緒契約。

### G2：圖形生命週期

停止 producer、最後有效 refresh、保留 Figure 與釋放呈現資源分開處理。取消請求不立即銷毀圖，既有 partial 正常返回仍可呈現最後資料。新操作不關閉使用者持有的舊圖。

Notebook backend adapter 處理顯示與 close，避免 cell 結束重複顯示，並提供明確釋放呈現資源的方式。close 對 widget 與保存的影響需依 backend 核對，不使用全域 close("all") 清理別人的圖。外部 Figure 的 adopt 不能撤回已發生的顯示副作用。

### G3：名稱與接管

操作內一名一圖。同名同 Figure 重複 adopt 等冪，同名異圖及同圖異名拒絕。已由其他操作持有的 Figure 拒絕接管，不偷移 canvas。第一版不提供 transfer 或 alias。舊圖由原集合保存，跨操作重畫使用資料。

### G4：多圖保存

逐圖記錄是否曾成功保存，以 stage 與圖名命名，單圖也遵守相同規則。保存前固定本次集合、產物 identity 與目的地，在 I/O 前檢查安全路徑與名稱碰撞。Identity 用來將完成紀錄對應回原產物，不是內容 dirty 版本。

同一份圖一旦保存成功，修改 artist、視圖或目的地不清除成功紀錄。不新增 dirty 通知或自動內容追蹤。新產物從未保存開始。

部分失敗保留成功項，不宣稱整組已保存，也不承諾跨檔原子保存。Save All 只保存未保存項，Retry 只重試未完成項。明確再次匯出另行表達，retry 不構成覆寫原始資料的授權。

### G5：晚到結果

Notebook 不新增 generation／request identity 的發布限制。較早啟動的互動或分析較晚完成時，允許覆蓋目前分析，操作順序由使用者負責。該次數值、來源 Result、實際 options 與圖集合仍須成組發布，不能混用不同操作的資料。

既有 GUI busy、operation token 與已取消 session 的生命週期規則不變。這項裁決不重新定義 run 中斷，也不刪除 GUI 的既有保護。

### G6：Notebook 互動前端

Notebook 互動分析直接使用 Notebook widget，不增加前端可用性 preflight 或 backend 偵測。建立或使用 widget 的實際錯誤正常傳遞，不自動降級到 inline、無頭分析或自動 Done。

同步分析的不呈現繪圖模式維持不變。它不是無頭互動工作流。

### G7：未遷移 adapters

GUI application 統一使用插件定義的 typed 分析輸出與 plots，不另要求通用終態 options。三個標準實驗的 concrete adapters 先接入新核心，其他實驗與 adapters 隨後分期遷移，不要求每個中間階段都維持舊 caller 可用。Interactive plugin／frontend 使用 session-owned 圖集合，不探測新舊簽名，不新增反射或雙協議 fallback。

完整清單須包含未出現在 GUI catalog 的實驗、組合量測 caller 與必要 helper。共用基底、fake 和支援框架需另外分類，不以 class 數量代替公開實驗清單。每項都需對應遷移範圍與驗證證據。

### 舊自訂 backend 退場

三個標準實驗與其餘實驗／GUI caller 接入 explicit factory／host 後，移除專案自訂的 pyplot routing backend，以及只服務它的 routing／scope 和設定。不以離屏 pyplot scope 延長舊核心的使用期。

退場包含共享 backend 的相關 GUI caller，例如 measure 與 fluxdep search，不能只刪除 backend 檔案而留下必要路徑未接線。這不是移除 Matplotlib rendering：原生 Figure／Axes、Qt canvas、Agg 與 ipympl 仍負責實際繪製。

共享 runtime 的 backend 選擇、host 初始化、shutdown handling 與 mathtext lock／prewarm 要分別核對。移除舊路由職責，保留或整理仍必要的容器、owner scheduling、attach／detach 及 rendering 初始化。

## 實作前仍需細化

- GUI 多圖檔名的精確格式、State／SaveService／截圖接線，以及各批 adapter／核心遷移的責任與驗證範圍。
- 完成後取圖、保留參照與釋放呈現的具體介面，以及名稱或接管衝突拒絕後的 owner 完整性。
- Notebook inline／widget 的顯示與 close、GUI worker／canvas 更新、最後 refresh 及失敗收尾。
- GE primary 替換後的 post 關係，以及 Notebook 互動完成後取得結果的方法名。細化不得新增 G5 已排除的 Notebook 晚到發布限制。

現有 Notebook close 與 GUI bridge 限制仍見 [liveplot](../../../lib/zcu_tools/plotting/liveplot/README.md) 和 [GUI plotting](../../../lib/zcu_tools/gui/plotting/README.md)。本草案不聲稱已完成執行期驗證。

## 轉正為現況的條件

先完成三個標準實驗的 Notebook／GUI 接線與 G1–G7 契約驗證，再以完整清單逐項確認其他實驗及必要 caller 已遷移。以公開 seam 驗證 cfg snapshot、typed analysis、primary／post 來源、互動 Done／Cancel、具名多圖與保存失敗；以直接審閱確認責任、依賴與匯出。

舊自訂 backend 退場須有適用 GUI caller 的行為證據。刪除路由與專用測試後，仍須驗證新 host 的呈現、刷新、失敗與釋放行為；不新增「舊檔案不存在」的測試，也不以刪檔結果作為替代成功的證明。

不得用一般 T1 通過推定 interactive 或 post-analysis 已驗收。Notebook inline／widget 顯示與 Qt thread 接合需有對應觀察，不以數值測試代替。驗證完成後才將已落實部分寫入現行 ADR 與 module README。
