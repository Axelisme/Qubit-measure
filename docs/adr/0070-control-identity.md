---
status: accepted
---

# ADR-0070 — GUI 與 agent 的控制身份

## 問題與選擇

GUI 使用者與 agent 寫入同一份 application state。若每條連線維護 seen map、宣告 resource dependencies、比對 cfg revision，caller 就得在每次寫入前建立完整觀察。這仍沒有告訴使用者現在誰在控制。

Measure 與 fluxdep 使用單一 holder。非 holder 不能送業務命令，holder 不必再以版本比對證明資料沒有被另一個控制身份修改。GUI 指示燈顯示控制身份，使用者明確收回或解鎖。Dispersive 與 autofluxdep 的 remote 仍只讀，不接入這套控制身份。

## Owner 與身份

各 app 的 `Controller.control_identity` 持有共用、Qt-free 的 `ControlIdentity`。Qt frontend 與 remote 讀同一投影。狀態轉移與命令入口在 application owner loop 序列化；RPC 排入 queue 不代表獲准，執行時才查 holder。

`holder` 初始為 `gui`，Remote 為 `rpc:<label>`。`locked` 表示 Local 禁止 agent 取得；`online` 表示 holder 在線，Local 永遠在線。任一相同 label 的已初始化連線在線，Remote 就在線。`me` 只在呼叫 socket 是 owner mode 且 label 與 holder 相同時為 true。

Label 是信任 LAN 上的控制身份，不是 authentication credential。同名 owner 連線共享控制權，不區分重連與同名競爭。Dispatch 從 socket 判定身份，不信任業務 params 自稱的 label。Token authentication、method exposure、輸入驗證與硬體授權照常適用。Event origin 只供歸因，不授權命令。

既有 import contracts 約束依賴方向。`mcp-remote-only`（C13）限制 MCP 使用 GUI remote 契約，`layers`（C8）與 `top-level-acyclic`（C9）約束跨 package 分層及無循環。`session-no-app`（C2）與 `interactive-below-apps`（C17）禁止共用機制反向依賴 app policy。Qt indicator／edit gate 只投影 authority，不另維護 holder 規則。

## 連線與轉移

認證與 wire 檢查之後，caller 送 `control.connect(label="agent", mode="owner")`。兩個參數的預設如示例。初始化前 socket 是 observer，readonly methods、`rpc.catalog` 與 `control.status` 可讀；業務命令回 `not_holder`，acquire／release 回 `readonly_connection`。`control.*` 永遠可進 dispatch，再按各命令契約判定。

初始化後 label／mode 固定。相同初始化是冪等的，不重新 acquire；不同 label 或 mode 回 `invalid_params`，原身份不變。換身份須重連。Owner connect 只在未鎖定 Local 自動取得，GUI 鎖定或其他 holder 不阻止連線。Readonly 永不取得，即使 label 相同也不能寫入或 acquire。

| 命令或事件 | 結果 |
| --- | --- |
| Owner `control.acquire`，未鎖定 Local | 取得，holder 為該 label。 |
| Owner acquire，Local 鎖定 | `control_locked`，holder 不變。 |
| Owner acquire，其他 label 在線 | `control_held`，holder 不變。 |
| Owner acquire，其他 label 離線 | 取得，holder 為該 label。 |
| 同名 owner acquire | 成功，狀態不變。 |
| Readonly 或未初始化 socket acquire／release | `readonly_connection`。 |
| RPC holder `control.release` | 回未鎖定 Local。非 holder 回 `not_holder`。 |
| Holder 斷線 | 保留 holder，不取消已接受工作。 |
| GUI 點燈，Remote | 收回成 Local 鎖定。 |
| GUI 點燈，Local | 切換 Local 鎖定／未鎖定。 |

`control.connect/status/acquire/release` 回 `{holder, locked, online, me}`。控制拒絕保留原生 reason 與 `{holder, locked, online}`，沒有業務副作用。MCP 只投影錯誤，不複製 holder 規則。不設 agent 直接交棒 API；release 不預留下一個 holder。

## RPC 與 GUI 寫入

RPC 預設要求 holder。Method 明確宣告 `readonly=True` 才允許 observer 呼叫；`control.*` 與 `rpc.catalog` 是管理例外。讀寫混合的 `tab.interact` 包含純讀在內仍要求 holder。通過 authority 後，命令才進入既有 domain、busy、capability 與輸入檢查。

Remote 時，GUI 停用欄位與寫入容器，不改變子元件原本的 busy／valid／input-open gate。純檢視、縮放、捲動、截圖、Log Browser 及 Inspect 只讀照常可用。Fluxdep spectrum 選取與 Filter 切換會改 domain，所以停用；search cancel 與 interactive cancel 也須先收回。

可寫 dialog 維持 modal。Remote 只停用內容與寫入操作，Cancel／Close／✕ 可關閉。使用者關閉後再按主窗燈，未提交輸入不保留。Static chooser 與 confirmation 不改呈現，返回後、真正寫入前重查 holder；權限已轉移就不寫入，不自動收回。

View owner 在 teardown 前釋放 edit gate 的 watches 與 Qt destruction callbacks。Measure 的單次 RunLoadDialog 在 finished 同期 dispose，主窗 teardown 先 reject 它，晚回 file chooser 不能重新 accept。這不是所有 dialog 都在 finished 退訂的規則；persistent hide／reuse dialog 保留自身 lifetime。

Measure 量測 Stop 與 device operation Stop 在 Remote 仍可按。按下先收回成 Local 鎖定，再請求取消。Stop 是僅有的自動收回入口，取消不代表硬體已停止。Remote 關主窗先確認，拒絕不開始 teardown，接受沿既有 shutdown／cancel／cleanup 契約，不另加 reclaim。Fluxdep 沒有硬體緊急 Stop。

## Cfg 與 operation

Cfg edit、Custom 選擇、reset、refresh、accept 與 preview 作用於目前 inputs，不收 expected revision。Publication 提供 status、tree、source basis 與 diagnostics。Watch 推送最新 observation，tab close revoke 舊 handle；view detach 只解除 watch。

普通 Run acceptance 只接受當下 Valid publication。Run owner 可提供固定的一次性 resolution，以目前 inputs 準備 Valid 候選，不改原 publication。接受後深層隔離 values 與 source basis，後續編輯或 source publication 不改已接受的執行資料。Active Run 的同 tab 編輯禁令、通知重入防護、hardware guards 與 lease 仍有效，見 [[0065]]、[[0066]]。

SourceRevision 是解析 provenance。TunableValues 的 expected_revision／RevisionConflict 是另一個 owner 的並行修改契約。它們都不是 cfg 編輯或 Run 的版本門檻。

Operation 不綁啟動者身份。控制權轉移或斷線不取消已接受 Run，也不撤銷其正常完成提交；新 holder 可取消，任何 observer 可 wait。Partial result、terminal outcome、不可取消錯誤與資源清理由 operation owner 決定。Timeout、斷線或 reply encoding failure 不能證明命令沒有副作用，不自動 retry 或 rollback。

## 取捨與相鄰責任

單一 holder 省去 per-connection seen guard、guard dependencies、observation rollback 及 cfg ref／revision admission。它不保證 holder 已讀最新資料，也不防止同名 owner 連線互相覆蓋。Source 自動 publication 屬於 holder 操作的結果，不再加 observation 保護。

直接編輯就接管會讓滾輪或誤點搶走控制權，所以除了兩種 Stop，只接受明確按燈收回。離線 holder 不自動 release，使用者可按燈收回，其他 owner 可明確 acquire。GUI 不啟動 agent，agent prompt／notify 接口不提供；MCP 的 recipe `answer` 仍只回答已捕捉的提問。

GUI owner 邊界見 [[0067]]，wire、authentication 與錯誤投影見 [[0068]]。本篇不新增硬體操作授權、不改 lease 或 shutdown 安全承諾。
