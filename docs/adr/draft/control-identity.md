# 控制身份（draft）

**狀態：** 使用者已核准方向，待實作。本 draft 不代表現行契約，也不授權直接移除現有 guard。
**關聯：** [[0068]] 的 remote dispatch 與 authentication、[[0067]] 的 GUI frontend、[[0066]] 的 operation lifecycle、[[0065]] 的 cfg publication。

## 問題與選擇

GUI 使用者與 agent 操作同一份 application state。現行 remote 以每條連線的 seen map、resource version、guard dependencies 與 observation tracking 防止覆蓋。這套機制要求 agent 在寫入前建立完整觀察，也要求每個 method 維護讀寫依賴。

控制身份將人與 agent 的寫入權限改成單一 holder。讀取不需要持有控制權。GUI 用指示燈顯示 holder，使用者按指示燈收回控制權。Agent 在未鎖定的 Local 狀態自行取得，不需要詢問使用者。Log 只供 debug，不作為取得、收回或寫入的前置條件。

本篇只規劃 measure-gui 的控制協作。它不規劃 agent launch UI、不改其他 app，也不改硬體操作授權。

## 身份與狀態

GUI application 擁有控制狀態，remote 與 Qt frontend 讀取同一份投影。

- `holder` 是 `gui` 或 `rpc:<label>`，初始值為 `gui`。
- `label` 由 agent 在 connect 時提供，例如 herdr pane 名稱。它是可重連的控制身份，不是 authentication credential。
- 每條連線沿用既有 `client_id`。Dispatch 以呼叫連線判定其控制身份，不能信任 method params 自稱的身份。
- Local 鎖定表示使用者禁止 agent 取得控制權。它與 holder 的連線是否在線是不同的事實。
- `rpc:<label>` 的在線狀態由連線生命週期提供。斷線不改 holder，指示燈顯示離線。

`EventOrigin(kind="agent", client_id=...)` 可供事件歸因與 debug。事件來源不授權寫入，也不觸發控制權轉移。

| 指示燈狀態 | Holder 與限制 | 點擊指示燈 |
| --- | --- | --- |
| Remote | `rpc:<label>` 持有控制權。GUI 不能編輯。顯示 label 及在線或離線。 | 使用者收回，切到 Local 鎖定。 |
| Local | `gui` 持有控制權。Agent 可以自行取得。 | 切到 Local 鎖定。 |
| Local 鎖定 | `gui` 持有控制權。拒絕 agent 的取得請求。 | 解除鎖定，切到 Local。 |

收回後進入 Local 鎖定，避免 agent 下一步又取得控制權。指示燈本身保持可操作，不隨內容區停用。

## 控制權轉移

| 情境 | 結果 |
| --- | --- |
| Agent 在 Local 取得 | 直接成功，切到該 agent 的 Remote。 |
| Agent 在 Local 鎖定取得 | 拒絕，回報使用者已鎖定。 |
| `rpc:B` 要取得 `rpc:A` 的控制權，A 在線 | 拒絕，holder 不變。 |
| `rpc:B` 要取得 `rpc:A` 的控制權，A 離線 | 允許，holder 改為 `rpc:B`。 |
| Holder 斷線 | 保留 holder，Remote 指示燈顯示離線。不自動釋放。 |
| 同一 label 重連 | 繼續持有控制權，連線身份使用新的 `client_id`。 |
| RPC holder 呼叫 `control.release()` | 回到未鎖定的 Local。 |
| 使用者在 Remote 點指示燈 | 立即收回，切到 Local 鎖定。 |

不設 agent 之間的直接交棒 API。A 呼叫 `control.release()`，B 再取得。如果第三個 agent 先取得，B 收到拒絕；release 不保留下一位 holder。

取得、釋放、收回與 lock 變更屬於控制管理命令。它們依上表檢查，不能套用「只有目前 RPC holder 才能取得」的循環限制。一般業務寫入則一律檢查 holder。

控制狀態轉移與寫入入口在同一 application owner loop 序列化。RPC 排入 owner queue 不表示已獲准寫入，dispatch 執行時才檢查當下 holder。收回生效後，尚未接受的舊 holder 寫入請求必須拒絕。

## 讀寫入口與 GUI 停用

任何身份都可以讀取，但連線仍須通過現有 authentication 與 method exposure 檢查。`label` 不取代 token，控制身份也不使 internal method 對外可達。既有 MCP 到 GUI remote 的依賴仍由 C13 約束，本 draft 不新增 package import 方向。

RPC 業務 method 以 `writes=True` 標示寫入，dispatch 在呼叫 owner 前統一拒絕非 holder。拒絕不能產生業務副作用。只有 holder 通過控制權檢查後，才進入既有的輸入與 domain 驗證。

Remote 時，Qt frontend 對主視窗的內容容器呼叫 `setEnabled(False)`。這會停用子元件的編輯入口，避免在每個 GUI 寫入路徑加入接管判斷。回到 Local 或 Local 鎖定後，frontend 解除這層停用；各 widget 仍遵守 domain 的 busy 與可編輯性限制。

Device、editor、writeback 等獨立對話框不在主視窗容器內。Remote 必須同時停用已開啟的可寫對話框，並在開啟可寫對話框時檢查 holder。只停用主視窗不構成完整保護。可讀畫面與控制指示燈不應提供旁路寫入。

## Operation 與保留的契約

執行中的 operation 屬於控制權，不屬於啟動它的身份。轉移控制權或 holder 斷線不取消已接受的 run，也不使既有 operation 的正常完成提交失去資格。新 holder 可以取消可取消的 operation；舊 holder 不能再送新的取消或互動命令。`wait` 是讀取，任何身份都可等待。

取消仍是請求，不是停止完成。`not_cancellable`、terminal outcome、partial result、硬體 exclusion、lease 與資源清理由 [[0066]] 的 owner 決定。收回控制權不等於儀器已停止，也不撤銷已發出的硬體命令。

控制身份取代的是 D1 的人與 agent 覆蓋保護：per-connection resource-version guards、`guard_deps`、seen/observation tracking，以及回覆失敗時的 observation rollback。它不回滾業務效果，也不增加自動 retry。

Cfg publication 的 identity/revision、Valid acceptance、result provenance 與 operation binding 有各自的資料與生命週期意義。它們不能只因名稱含 version 就刪除。本 draft 不改 [[0065]] 的 `CfgRef` 契約，也不以 holder 取代輸入驗證、硬體限制或 domain 的 busy 檢查。

## 不採用的方案與代價

直接編輯就自動接管會讓滑鼠滾輪或誤點奪走 agent 的控制權。GUI 寫入直接呼叫 facet/service，沒有 RPC dispatch 這樣的單一檢查點。用 event origin 判定接管也不可靠，背景完成事件可能使用預設 user origin。因此只接受明確點擊指示燈的收回。

控制身份讓單一 holder 負責寫入，代價是 Remote 時使用者須先收回才能編輯。它不保證 holder 已讀過最新內容，也不阻止同一 holder 用過時資料作出決定。離線 holder 保留身份，代價是 GUI 仍需按指示燈收回；其他 agent 則可按轉移規則接手。

## 落地前待確認

下列項目不由本 draft 補猜，實作前須定出 wire contract。

- 取得與查詢控制權的 RPC 名稱、connect label 的傳遞位置、回覆欄位，以及鎖定／他人持有時的 error reason。
- 兩條在線連線使用相同 label 時的規則。必須能區分重連與同名競爭，不能讓任意同名連線冒充在線 holder。
- 兼具查詢與命令的 method 如何宣告 `writes`。例如 `tab.interact` 的純讀與帶 command payload 必須分開判定，不能讓 command 旁路檢查，也不能禁止非 holder 的純讀。
- 非編輯型 GUI 控制的停用範圍。需要明確盤點 Stop、prompt 回答與關窗等入口，確保不繞過 holder，也不使使用者失去收回入口。

## 轉正條件

透過公開 GUI 與 RPC 接縫驗證以下行為，不直接測私有 helper 或新增文件內容測試。

1. 初始 Local、Local 鎖定拒絕取得、解鎖後取得，以及 Remote 收回後進入 Local 鎖定。
2. A 在線時拒絕 B；A 離線時 B 可接手；同 label 重連按已確認的規則續持；release 後的第三方競爭回明確拒絕。
3. 非 holder 可讀但不可寫。Owner queue 中的請求按執行時 holder 判定，拒絕前不呼叫業務命令。
4. Remote 停用主視窗及既有、新開的可寫對話框。收回後恢復 local 編輯，不因背景事件自動接管。
5. 控制權轉移與斷線後 run 繼續，新 holder 可取消，其他身份可 wait。原 operation 仍按 domain 規則提交與回報終局。
6. 移除只服務 D1 observation guard 的邏輯與測試，保留 cfg acceptance、provenance、busy、authentication 與硬體安全契約。

落地前先完成待確認項目。實作與接縫驗證通過後，再更新 [[0068]] 及適用的 module README；有實際 owner 邊界變更時同步核對 [[0067]]、[[0066]]。本次只新增 draft，不修改 ADR 索引與程式碼。
