# 0017 — worker 線程畫圖：依「畫圖在哪呼叫」分流 marshal vs 純通知

**狀態：** accepted（探索後定性）。
**關聯：** 同屬「worker 不能直接碰主線程 UI」家族（[[0002]] off-main handler）；承 plot substrate 收斂（`zcu_tools.gui.plotting`）。

## 背景

多個 Qt GUI app 都有「worker 線程要把 matplotlib 圖顯示到 Qt widget」的需求。matplotlib 官方互動指南的鐵律：UI 只能在主線程建立、且**只能從主線程畫到螢幕**；worker 直接 `pyplot` 會不渲染或 crash。問題分成兩種執行模型，採**相反的具體解**——判準是**「畫圖在哪呼叫」**。

## 決策

### Case A — worker body 直接呼 pyplot → 經 signal/slot marshal 回主線程

適用：**measure-gui**（`RunWorker` liveplot / `AnalyzeWorker`，adapter 內部 `plt.subplots()`）、**fluxdep**（`_SearchWorker`，`search_in_database` 內部 `plt.figure()`）——繪圖邏輯藏在 worker 跑的既有 adapter/notebook API 內、中間量不回傳、改不動（越 gui scope）。

機制（`zcu_tools.gui.plotting`）：自訂 `module://` backend 攔截 `plt.figure/show/draw`，經**主線程單一 host QObject 的 signal/slot** 把畫圖操作 marshal 回主線程，內嵌進 `FigureContainer`。worker 在 body 裡 `with routing_scope(container)` 宣告路由目標。

這是**官方推薦標準解，非 over-engineering**：

- 互動指南逐字：「background threads … via Qt's Signal/Slot mechanism … **This is the recommended approach for updating Matplotlib figures from worker threads while keeping the GUI loop on the main thread.**」
- matplotlib core maintainer 的 `mpl-qtthread` 與本機制逐點同構（主線程 QObject 持 signals、自訂 `module://` backend emit signal、callback 在主線程跑）。要正確支援 worker 畫圖，唯一辦法就是裝它或寫等價物——本實作是等價物，沒多做。
- measure analysis 與 fluxdep search 走**同一條** routing 路徑、同一個 `configure_matplotlib_backend()` / `BACKEND_NAME`，是收斂兩 app 的結果，非兩套。
- `routing_scope` 在 worker body 進場是**固有一環**：ContextVar 不被 Qt pool/thread worker 從 enqueuing 線程 snapshot，所以**任何** worker 都必須在自己 body 裡宣告路由目標（兩 app 共有，非缺陷）。

### Case B — 畫圖留主線程、worker 只發資料通知 → 一般 Qt queued signal

適用：**autofluxdep** run worker。執行模型不同：Plotter 由 UI 在主線程建（`make_plotter(figure)`，生命週期 = 整個 sweep）；worker 的 `run_body` **從不碰 matplotlib**，只把當前 raw 原地填進 sweep Result 的 flux-idx 行（numpy），emit 一個**純通知 signal**（只帶 flux index）；主線程 slot 收到才 `plotter.update(result, idx)` 畫。

worker 根本不畫圖，沒有「worker 線程的畫圖操作」需要 marshal——只有「資料更新了」需要通知，一般 Qt queued signal 即足，不需 backend 攔截 / host routing。

**執行緒安全**：worker 原地寫 Result 第 idx 行、主線程收 queued signal **之後**才讀那行畫圖；Qt queued signal 提供 happens-before，故共享 numpy Result 安全。

### 判準與分流

| | Case A（marshal） | Case B（純通知） |
|---|---|---|
| 畫圖在哪呼叫 | worker body 內 `plt.subplots()`（adapter/notebook 內部畫，中間量不回傳） | 主線程 Plotter，worker 不碰 matplotlib |
| worker→主線程送什麼 | 被攔截的畫圖操作（backend → host signal/slot） | 純資料通知（flux index） |
| 為何能這樣 | 繪圖藏在既有 API 內、改不動（越 gui scope） | 新寫的 app、自定義 Plotter，能把「怎麼畫」收進主線程 |

兩者**不互斥**：某 NodeType 若不得不複用「只在 worker 跑、內部 `plt` 畫」的既有分析，那個 NodeType 退回 Case A。

### Case A 的 thread-safety 缺口：off-main mathtext parsing（BUG-1）

Case A 把 worker body 內**經 backend 攔截的 `plt.figure/draw`** marshal 回主線程，但 worker 跑的 domain `analyze()` 仍在 worker 線程做**非繪圖**的 matplotlib 計算——`set_title("$...$")` + `tight_layout()` 會在 worker 線程 parse mathtext。matplotlib 的 mathtext parser 是 class-level singleton（`MathTextParser._parser`，pyparsing，非 thread-safe）；兩個 worker、或一個 worker 與主線程 draw 並行 parse 時踩髒 singleton，丟非決定性 `ParseException`。backend 的 figure-op marshal 不涵蓋這條：parse 發生在「畫圖操作被攔截」之前的 layout 計算裡，仍在 worker 線程。

**目前生效的 contained 修法**：process-wide mathtext lock + 主線程 prewarm（`zcu_tools.gui.plotting.mathtext_lock`）。`install_mathtext_lock()` 用單一 process-wide `threading.Lock` 包住 `MathTextParser.parse`（public 進入點，覆蓋所有 thread/path），idempotent；`prewarm_mathtext()` 在主線程先 parse 一次 `$...$`，使 parser 的 lazy init 不首度發生在 worker 競態下。三個 GUI 啟動點（measure `app/main/app.py`、共用 `run_qt_app`/`run_app.py` 涵蓋 fluxdep+dispersive、autofluxdep `app/autofluxdep/app.py`）在 QApplication 建立後於主線程各裝一次（idempotent，防退化）。不改任何繪圖程式碼，也不改 domain 的 `$...$` title 字串。

**Follow-up（尚未做）**：根治法是把 domain `analyze()` 的 compute 與 plot 拆分，讓所有 matplotlib 操作（含 mathtext parsing）都 marshal 回主線程跑，使 Case A 不再有任何 matplotlib 工作留在 worker 線程。此舉需改 adapter/notebook 內部繪圖契約（越 gui scope，與 Case A「繪圖藏在既有 API 內、改不動」的前提衝突），故現階段以 lock + prewarm 涵蓋缺口。

## 拒絕的替代方案

- **worker 改畫 detached `Figure()`、emit Figure 回主線塞 canvas**（想削掉 Case A 的 `routing_scope`/ContextVar）：否決。① 削不掉 pattern（measure analysis 仍走 routing，backend marshal 層照樣存在）；② 製造分岔（fluxdep 變特例，與收斂方向相反）；③ 侵入非 gui scope（fluxdep 繪圖中間量藏在 `notebook/analysis/fluxdep/fitting.py` 的 `search_in_database`，要改其內部繪圖/回傳契約，越出 gui scope 且波及 notebook 直接使用者）。
- **把整個 search 搬回主線程跑**（放棄 off-main）：HDF5 load + 進度回報會凍 UI 數秒。
- **每個畫圖來源各自一個 backend / 多 client 註冊**：matplotlib 一進程只認一個 backend；維持單一 client = 單一 backend、host 持 registry 解析 figure→container。

## 後果

- Case A 共用 `zcu_tools.gui.plotting` 機制定性為**必需且正確**，不再被「能不能簡化掉 worker marshal」的念頭繞回來重啟。
- Case B（autofluxdep）figure 嵌入用**樸素 matplotlib-Qt**（主線程 `Figure()` + `FigureCanvasQTAgg` + `canvas.draw_idle()`），不走 `gui/plotting` 的 FigureContainer/attach。與 fluxdep/dispersive 分岔是**必然**——那套 backend marshal 是為 worker 畫圖設計的，autofluxdep worker 不畫圖；`make_plotter(figure)` 收純 matplotlib `Figure`。
- 若未來 matplotlib/Qt 提供 first-class 跨線程繪圖 API，重啟時本 ADR 的查證結論與 `mpl-qtthread` 對照可直接取用。
