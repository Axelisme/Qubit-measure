# ADR-0068 — 遠端前端、傳輸與 agent 介面

**狀態：** accepted。

## 問題與選擇

Qt 視窗、socket client 與 MCP agent 都能接觸同一個 GUI application。若每個入口另存可提交的狀態、複製 guard，或由共用 socket 層決定實驗方法，使用者與 agent 會依入口不同而看到不同結果。Remote 是 application 的 driving adapter；它呼叫同一批 owner 的查詢與命令，不建立第二份業務真相。Measure 的 `RemoteControlAdapter` 接在共用 endpoint 上，承接 Controller 的診斷並將請求送到 application owner。GUI 的草稿與互動分析 committed session 仍由其原 owner 持有；GUI-local pointer preview 不是遠端可提交的狀態。Qt 的 screenshot、dialog 和焦點是 frontend 能力。Agent 寫入後切換 tab／pane 的行為由 measure app 的 view 命令決定，不由 socket 或通用 MCP bridge 暗中決定。共用 application 狀態與前端邊界見 [[0067]]。

## 分層與連線

`gui.remote` 提供 NDJSON framing、request／reply、握手、per-client queue、backpressure 和 lazy push endpoint。`mcp.core.bridge` 持有 agent 程序的 GUI socket、連線與可選的 GUI subprocess；`mcp.core.stdio_server` 提供 stdio MCP 機制。四個 app 使用共用 GUI remote 機制，但各自註冊 wire method、驗證與 event serializer。Measure 還擁有版本 guard、operation tracking、診斷和 editor 清理；fluxdep、dispersive 與 autofluxdep 的 remote 接口只讀。共用機制接收 app 注入的 metadata 和 owner scheduler，並不持有實驗或資源政策。精確封套、上限、版本與 method 說明見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

同機跨程序的 MCP 使用 socket；同機並不等於同程序呼叫。GUI 可停用 remote socket，GUI core 不依賴 MCP。MCP `connect` 可接上現有 GUI，或經明確選項啟動 GUI；MCP server 結束只斷開連線，不關閉 GUI。外部 CLI／MCP workflow 擁有 agent 的啟動。GUI 不啟動 agent terminal、不管理可恢復的 agent session，也不注入 bootstrap prompt。Connection authentication 與 method exposure 是兩道不同的邊界：有 token 時 GUI 在 method 前驗證；沒有 token 的 loopback 允許本機可連線程序控制 GUI。隱藏 method 不等於授權機制，現有連線方式不承諾任意跨機部署。

## Method 與一致性

Measure GUI 的 `RemoteMethodEntry` 同時宣告 method schema、agent exposure、guard dependencies、成功讀取所揭露的資源及 operation key。GUI 的 `rpc.catalog` 只投影非 internal method 呼叫所需的名稱、描述、參數 schema、timeout、exposure、tool 路由與 operation key，不把 guard／reveal policy 複製到 MCP。MCP 每次連線重新讀 catalog。固定的 40 個特化 tool 涵蓋常用判斷點，另有 `rpc_list`、`rpc_describe` 和 `rpc_call` 供低頻 method 使用。固定工具不從 catalog 動態生成。`tool` exposure 經通用 call 回 `use_tool`，`internal` 不列入 catalog；標成 `rpc` 的 method 仍可同時被特化工具使用。每個入口最終仍經 GUI 驗證，不提供任意程式碼執行。工具的個別輸入、畫面跟隨與 operation 結果見 [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)。

GUI owner 在自己的序列中比較每條連線的 seen map 與目前資源版本，再接受 guarded command。成功的完整讀取才建立宣告的觀察；部分讀取、失敗與回覆編碼失敗不建立新觀察，版本零也不能替代未曾觀察。成功寫入只推進先前看過且版本相符的資源。MCP 不存第二份 seen、不送 wire expected versions、不用隱藏預讀解鎖。Stale 須重讀對應資源，再由呼叫者決定是否重試；event origin 不能授權寫入。Guard 不替代長時間操作的 snapshot、資源互斥與後續提交檢查，operation 的生命週期歸 [[0066]]。Timeout、斷線或無法編碼的回覆也不能證明命令沒有副作用；沒有明確冪等契約時不自動重送。

## 事件與錯誤

Application state 和明確讀取是資料來源。EventBus 產生同步的 domain facts 與 `EventMeta(seq, origin)`；dispatch 宣告 origin，operation 跨執行緒顯式攜帶原發起者與 operation id。Seq 表示同一 process 的順序，不是可靠交付證明；過濾或重啟後的跳號不能直接判定遺失。Wire 的 EventBus push 加上 seq 與不含 client id 的 origin，保留原 event payload；subscriber 可自行收合呈現，不能改變必要的 domain commit 順序。Endpoint 只有找到 matching subscriber 才建立、編碼 payload，並在 enqueue 前重驗原 recipients 的訂閱和連線狀態。完成 unsubscribe／disconnect 後不會收到遲到的 push。診斷由 Controller 直接 fan-out，不經 EventBus 訂閱；這不表示 socket 故障時仍能交付。Measure MCP 不訂閱業務 push，而以讀取與 operation wait 接手進度；重連重新讀 snapshot，沒有 replay 承諾。高頻 plot／progress 不預設推送給每個 client。

可由 caller 修正的失敗由 producer 用 remote-independent `ExpectedError` 類別顯式分類。共用 dispatch 只翻譯已標記的 invalid input 與 failed precondition；handler 直接提出的 `RemoteError` 保留原本的結構化資料。它不從 `RuntimeError`、訊息字串或 reason 前綴猜測分類。Unexpected failure 留下診斷。`wait` 呼叫本身失敗與 operation outcome 為 failed 是兩件事；timeout 亦不是安全重試的許可。精確 error code 與 reason 見 [measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。

## 代價與相關文件

共用 transport 讓四個 app 不必維護四套 socket 與 framing，但 app 仍要各自維護 method 和 domain policy。固定工具與 live catalog 分開維護呈現與可達 method；這比只生成一份動態工具表多一層入口，換來常用判斷點的明確工具與低頻能力的即時發現。Wire 支援 push 不強迫 LLM client 訂閱，也不承諾 replay。Cfg 使用邊界見 [[0065]]，operation、cancel 與 shutdown 見 [[0066]]，GUI 程序組裝見 [[0064]]。
