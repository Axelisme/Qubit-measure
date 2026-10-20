# 0064 — GUI process startup 與 app composition

**狀態：** accepted。

## 脈絡

獨立 GUI app 需要共用的程序啟動順序，但每個 app 的 controller、window 與 domain services 不同。這裡的 runtime 是 GUI process runtime，不是 experiment acquisition runtime，也不是 Python 執行環境。

## 決策

Launcher 擁有 CLI 解析與程序入口，把每次啟動的選項交給 shared runtime。`gui.launcher` 提供共用的 CLI 轉換；app 不透過 CLI parser 組裝 domain services。

App 宣告固定的 process requirements，並提供 app-specific composition。`gui.runtime` 協調共用的 logging、Qt application、rendering 初始化、remote adapter 啟停與程序返回碼。App 組裝 controller、window、domain services 與自己的 remote adapter。Runtime 不決定 remote method、guard 或 app state。

Runtime 提供 app lifecycle 接點並控制啟動順序。App 在接點內觸發自己的 startup workflow，透過既有 owner interface 執行。Composition 決定建立與注入哪些模組，startup 不接管 State、service 或 registry 的寫入權威。Dialog 與 control port 不接收 app phase，啟動與手動開啟使用同一個 Setup dialog。這不要求每個流程另建 Coordinator class。

Runtime 在 Qt application 建立後初始化 explicit plot host、shutdown callback 與繪圖鎖，不切換 process-wide Matplotlib backend。Plotting owner 維護 host 與 rendering 機制。Transport 擁有連線機制，app 擁有 remote capabilities 與 policy。停止 socket 或結束 Qt event loop 不代表 domain operation 已完成清理；operation 的取消與資源釋放由其 owner 負責。

## Measure 啟動腳本

Measure launcher 選擇一份可信 Python 檔，檔案定義 `startup(ctx)`。選擇規則如下：

- `--clean` 不執行任何腳本，包括預設腳本。它與 `--script` 互斥，同時給時報錯。
- 沒給這兩個選項時，執行 `zcu_lab/startup.py`。
- 給 `--script <path.py>` 時，只執行該腳本。自訂腳本若要預設行為，自己 import 並呼叫預設 `startup`，框架不自動串接。

Composition 將選好的 `Path` 注入 `MeasureGuiBehavior`。Lib 不 import zcu_lab，沿用 [[0072]] 的 C16 `framework-no-user` contract。框架沒有 simulation environment loader 或 `--simulate` 相容入口。腳本可以安裝模擬資源，也可以連線硬體、建立元件或開啟 tabs。`--clean` 也略過已保存的 Qt 外觀；無腳本時 app 開啟一般 Setup dialog。

視窗顯示且 Qt 事件迴圈啟動後，app 以既有 operation runner 在背景 worker 執行腳本。狀態列顯示進度，STARTUP gate 暫時禁止 Run，但不禁止腳本經 owner 操作 SoC 或 device。未捕捉錯誤保留已完成的命令與已登記的 callbacks，解除啟動鎖，並以含腳本路徑及可定位行號的對話框報錯。GUI 保持可用，不回滾成空 session，也不回退另一個腳本或模擬環境。

`StartupContext` 提供型別明確的狀態讀取與建立能力。Facet 將同步命令送回 owner thread，沿用原 owner 的驗證、gates 與例外。讀取只投影目前狀態，不讀硬體或重新 capture。阻塞的 SoC／device connect 僅限 script worker，owner callback 呼叫時在建立 operation 前拒絕。Tab 由腳本在來源就緒後建立，沒有 framework `restore_session` 或 pending 還原。

使用者於 2026-10-19 說明：「具體來說，是一個啟動腳本的定位，主題是一個函數，接受controller或是特定投影的facat，在app建立時自動完成一些操作，而simulate只是其中一種應用，也可以是自動載入preset以及建立component」。同日要求：「gui本身應該設計成對simulation無知，然模擬的認知集中在啟動腳本中」。因此框架提供一般操作能力，不依資源是模擬還是真實來決定啟動流程。

## 保存與關閉

Framework 只保存 `measure_layout_v1.json` 的視窗尺寸與左面板寬度。SoC、devices、entry／point、tabs 與輸入的重開政策由腳本決定，框架不保存、還原或遷移領域 session。RAM 的 `WorkspaceTab`／`WorkspaceSnapshot` 供 workspace 重建與實驗 reload 使用，不是磁碟 schema。

預設 `zcu_lab/startup.py` 擁有 user cache 根目錄下的 `measure-session.json`。它依序嘗試上次 SoC、各 device、entry／point，只有 entry／point 成功後才重開 tabs 並還原 input。Device 連線失敗時嘗試 remember 為未連線列。每項失敗記 log 並繼續，最後狀態列顯示一行摘要；略過的 tab 不會在稍後選來源時自動重試。腳本以 `on_close` 讀取當時 live state 並保存自己的檔案，不保存硬體設定或量測結果。

Alias 屬 entry。開 entry 載入 `aliases.yaml`，Inspect Apply 寫回。預設腳本不另存、不恢復 alias，也不建立第二份 alias 權威。模板佔位符只由呼叫端在 `Template.bindings` 明填，session 與 startup 不提供預設值，也不保存或恢復模板綁定，見 [[0073]]。

使用者於 2026-10-20 裁定：「你搞錯了吧，模板/alias是不同的東西，後者才是startup填的」。

使用者於 2026-10-20 說明：「框架提供restore session的功能太強勢了，等於是哪些要存哪些不要的認知，與我目前精簡框架與解偶的方向不合」。對 alias 裁定：「A，那腳本就不回復alias了」。同日明定：「那--clean的行為應該就是連預設腳本都不執行」「clean與script都給就抱錯」。

腳本用 `ctx.on_close(fn)` 登記無參數收尾。正常關窗與 MCP `app.shutdown` 都先停止 admission、取消及 drain 工作，再在 owner thread 反序執行 callbacks 一次，最後關閉 GUI 管理的 drivers 並卸載 SoC。Drain 逾時不消耗 callbacks、不提前關硬體，可重試關閉。單一 callback 失敗記 log 並繼續後續清理。Callback 可使用同步 facets，不可等候 owner delivery 或提交新背景工作。強制 kill 不保證執行。

SoC 由 caller 擁有，不假設通用 close 方法。Device 只有成功 register 後才移轉 GUI close 責任，拒絕的 driver 仍由 caller 清理。逐台 close 與 adapter 邊界見 [[0067]]，operation drain 見 [[0066]]，檔案保存權威見 [[0063]]。預設腳本的逐項失敗呈現與一般未捕捉 startup 錯誤不同，EventOrigin 責任界線見 [[0068]]。

## 邊界與核實

目前四個 standalone launcher（measure、workflows、fluxdep、dispersive）透過共用 runtime 啟動。`gui/runtime.py` 建立或取得 `QApplication`，呼叫 app assembly 與 lifecycle hooks，並在正常退出路徑停止 remote adapter。這只證明 standalone 入口的流程，不承諾嵌入既有 Qt host 時的 event loop 所有權，也不承諾 assembly、plotting 或 adapter 啟動途中失敗時完整清理。Runtime 不管理外部 agent session、Python profiles 或板端服務。

`GuiRuntimeSpec`、behavior hook、launch options 與 rendering 接線見 [`gui/README.md`](../../lib/zcu_tools/gui/README.md)。Measure 的公開 facets 與局部契約見 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)、[`startup_context.py`](../../lib/zcu_tools/gui/app/measure/startup_context.py) 及 [預設腳本](../../zcu_lab/startup.py)。
