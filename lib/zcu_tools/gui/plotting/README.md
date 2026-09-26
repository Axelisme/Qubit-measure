# `zcu_tools.gui.plotting` — Qt 繪圖接入

**Last updated:** 2026-09-27 — plotting 家族定位

此目錄提供共用的 matplotlib/Qt 接入機制，不決定各 app 的 figure 接受、保留或清理政策。`setup.py` 的 `configure_matplotlib_backend()` 必須在匯入 `pyplot` 前由入口程式呼叫，選擇 process-wide 的 `module://zcu_tools.gui.plotting.backend`；若 `pyplot` 已匯入會報錯。根模組透過 `__getattr__` 延遲載入 Qt／matplotlib 相關匯出，避免單純匯入 package 就觸發重型依賴。

`backend.py` 的 `GuiFigureManager` 在 pyplot 建圖時將 canvas 交給 host，`show()` 只啟用已附著的 figure。`container.py` 的 `FigureContainer` 是包住 `QStackedWidget` 的被動容器，提供 attach、detach 與清理 canvas 的方法；建立容器時先在 GUI 主線程初始化 host bridge。`routing.py` 以 `ContextVar` 保存當前容器，供新 figure attach 使用；`host.py` 持有主線程 QObject bridge 與 weak-key figure-to-container registry。已建立 figure 的後續操作依 registry 找容器，不重新讀取 routing context。

Host 透過 Qt signal 把 worker 發出的 attach、activate、refresh 等請求交給主線程；attach 等待主線程回傳 canvas。`GuiFigureCanvas.draw_idle()` 將 worker 的重繪請求送到主線程，主線程呼叫則直接使用 canvas 的 `draw_idle()`。這不是任意 matplotlib 操作皆可跨線程執行的保證。`mathtext_lock.py` 另以 process-wide lock 序列化 mathtext parsing，並提供主線程 prewarm；它不取代 figure lifecycle 的主線程要求。

App 負責選定 figure 容器、設定 routing scope，以及決定 Run／Analyze figure 的生命週期。Measure app 的 `main/driven/qt_liveplot_backend.py` 仍是 app-local adapter，`main/services/scopes.py` 在 run worker 的 ambient scope 註冊它；不是此目錄的共用 backend。GUI 入口與 runtime 負責 process 啟動及 backend setup，這個目錄只提供機制，不接管 app 的啟停流程。
