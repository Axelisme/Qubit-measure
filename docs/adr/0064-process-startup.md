# 0064 — GUI process startup 與 app composition

**狀態：** accepted。

## 脈絡

獨立 GUI app 需要共用的程序啟動順序，但每個 app 的 controller、window 與 domain services 不同。這裡的 runtime 是 GUI process runtime，不是 experiment acquisition runtime，也不是 Python 執行環境。

## 決策

Launcher 擁有 CLI 解析與程序入口，把每次啟動的選項交給 shared runtime。`gui.launcher` 提供共用的 CLI 轉換；app 不透過 CLI parser 組裝 domain services。

App 宣告固定的 process requirements，並提供 app-specific composition。`gui.runtime` 協調共用的 logging、Qt application、rendering 初始化、remote adapter 啟停與程序返回碼；app 組裝 controller、window、domain services 與自己的 remote adapter。Runtime 不決定 remote method、guard 或 app state。

Runtime 提供 app lifecycle 接點並控制啟動順序，app 在接點內處理自己的 startup workflow，包括需要時顯示 setup dialog。Composition 決定建立與注入哪些模組；有界的 startup coordination 決定啟動情境的觸發、順序與分支，透過既有 owner interface 執行，不接管 State、service 或 registry 的寫入權威。Dialog 與 control port 不接收 app phase：app 的接點只決定何時呼叫既有入口，啟動時開啟的 Setup 與使用者手動開啟的是同一個 dialog 身分與同一個實例，還原的偏好只預填。這不要求每個流程另建 Coordinator class。Mock setup 屬 session 能力的流程，不限定由 startup 觸發；此處不規定 mock ready、rollback 或 retry 政策。

Runtime 在 Qt application 建立後初始化 explicit plot host、shutdown callback 與繪圖鎖，不切換 process-wide Matplotlib backend。Plotting owner 維護 host 與 rendering 機制。Transport 擁有連線機制，app 擁有 remote capabilities 與 policy；runtime 只協調 adapter 的啟停。停止 socket 或結束 Qt event loop 不代表 domain operation 已完成清理；operation 的取消與資源釋放由其 owner 負責。

## 邊界與核實

目前四個 standalone launcher（measure、autofluxdep、fluxdep、dispersive）透過共用 runtime 啟動；`gui/runtime.py` 建立或取得 `QApplication`，呼叫 app assembly 與 lifecycle hooks，並在正常退出路徑停止 remote adapter。這只證明這些 standalone 入口的流程，不承諾嵌入既有 Qt host 時的 event loop 所有權，也不承諾 assembly、plotting 或 adapter 啟動途中失敗時完整清理。Runtime 不管理外部 agent session、Python profiles 或板端服務。

`GuiRuntimeSpec`、behavior hook、launch options、rendering 初始化、exit code 與精確初始化順序見 [`gui/README.md`](../../lib/zcu_tools/gui/README.md) 及對應程式；各 app 的 startup 接線見各 app README。這些局部選擇不是跨模組不變式。
