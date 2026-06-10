---
status: accepted
---

# Service 互依的「三問規則」—— 取代分層表，治「service 互呼卡循環」

**狀態：** accepted。是一條判定準則（無程式碼產出），供未來新增 / 連接 service 時遵循。

## 脈絡

domain service 越來越多（14+），反覆撞到的痛是：**想讓 A 呼叫 B，卻擔心成環**（B 也需要回頭找
A）。曾考慮的解法是「**全部接 `ctrl`**」——用一個萬能入口避開判斷。否決：那會把每個 service 很窄的依賴
（`state`/`bus`/`gate`）變寬成「能碰 Controller 上任何東西」，製造 service↔Controller 雙向環、殺揉
責任邊界，正是當初拆 god-class 的反方向（見 [[feedback_converge_to_existing_abstraction]]）。

也考慮過「**service 分層表**（tier 1/2/3，邊只能高→低）」。否決：分層是**關於全局結構的問題**，需求一變
（某 service 突然需要呼叫本來在它上層的東西）整張表就要重排——脆。

## 決策

**不維護分層表。每條「A 要用到 B」的邊落地時，只問一個局部問題：A 需要的是 B 的「答案」、要 B「做一件
事」、還是想在 B「變化後反應」？** 三種意圖對應三種連法，判定**不依賴任何全局層級**：

| A 對 B 的意圖 | 連法 | 為何不成環 |
| --- | --- | --- |
| **Query**「B 現在的值是多少」 | 兩者都讀 **State**，A 根本不碰 B | 誰都不持有誰；State 是被動容器，不反向呼叫 |
| **Command**「叫 B 去做某事」 | A 構造注入 B，直接呼叫 | 只在 B 不需回頭找 A 時用 —— 天然單向 |
| **Reaction**「B 變了我要跟著動」 | A 訂閱 **EventBus**，**不持有** B | 依賴反轉：B 不知道 A 存在，環被 bus 截斷 |

**關鍵洞察：撞到「會成環」時，那幾乎一定是把一個 Reaction 誤當 Command 在做。** 你想讓 A 直接呼叫 B，
但 A 其實只要「B 變了之後跟著算」—— 那是 Reaction，改走 EventBus，環當場消失。

### 為何比分層表抗需求變化

- 分層表問「B 在哪一層」= 全局結構問題 → 需求變就重排。
- 三問問「**這一次** A 對 B 的意圖」= 單次調用的局部問題 → 與全局結構無關。需求怎麼變，都只是對新的那條
  邊問同樣三個問題。環不靠一張表預先擋住，而是在判定「這其實是 reaction」的瞬間自然不成立。

### 落地操作流程（每次想讓一 service 用到另一個）

1. **能不能不呼叫？** A 只要數據 → 兩者都讀 State，邊根本不存在。（最優先，State-as-SSOT 架構天生支持）
2. **必須呼叫，且 B 不需回頭找 A？** → 構造注入，直接呼叫（單向 command，安全）。
3. **發現 B 也需要回頭找 A（要成環）？** → 把「反應對方變化」的那一向改成 EventBus 訂閱，環立即斷。

### 與既有事實一致

- **Command 範例**：`startup → context/device`（呼叫 register/apply）、`workspace → tab`、
  `tab_view → tab/writeback/context`、`device → operation_gate`。皆單向、無環。
- **Reaction 範例**：Phase 98 `startup` 想知道 device 變了 —— 若用 Command（startup 持 device、device
  回頭通知 startup）會成環；改訂閱 `DEVICE_CHANGED` 環就沒了。已實證。
- **Query 範例**：絕大多數 service 只依賴 `state` —— 它們要的是「現在的值」，讀 State 即可，彼此不連邊。
- **narrow Protocol（query 環境的特例）**：`ControllerProtocol` / `_EditorCtrl` / `_ViewQueryTarget` 是
  「A 需要呼叫 Controller/View 的能力，但對方是 owner（會造成環或尚未建好）」時的解 —— 用窄 Protocol 表達
  「我需要的能力」，與「誰實作它」解耦。這是 Command 的一種受控形式（依賴介面非具體類），不是第四種模式。

## 推論：狀態放哪（State vs service 私有）—— 兩軸正交，勿坍縮

三問規則決定「service 之間怎麼連」；它的 Query 那欄直接推出「一個狀態放 State 還是 service 私有」。
**關鍵：「進不進 State」與「persist 投不投影」是兩條獨立的軸，不可坍縮成一條**（早期把「不可序列化」
誤當成「不能進 State」，是錯的）：

| 軸 | 問題 | 答案決定 |
| --- | --- | --- |
| **軸 1：進不進 State** | 除了 owner service 自己，還有別人（View / agent / 另一 service）需要**讀**它嗎？ | 有 → 進 State（領域真相）；只有 owner 用 → 留 service |
| **軸 2：persist 投不投影** | 它重啟後還有意義嗎？ | 有 → persist 選擇性投影；無 → persist 跳過 |

兩軸正交，四格都合法。**「不可序列化」只影響軸 2，完全不影響軸 1** —— State 可以安心持有不可序列化的
共享活物件。實證：

- **`ExpContext.soc`（不可序列化 + 多 service 讀 → 進 State、persist 跳過）**：live SoC handle 在
  `State.exp_context.soc`，`guard` / `connection.has_soc()` / `run`(經 `RunRequest`) 都讀它；`connection`
  service 自己**不另持**一份。重啟後連線沒了 → persist 視而不見。這格自洽，已跑很久。
- `DeviceState`（可序列化 + 多人讀 → 進 State、persist 投影 remembered 子集）。
- worker / lease / driver / `save._pending_image`（只 owner 用 → 留 service；順帶不可序列化，但那不是留下的理由，
  「沒別人要讀」才是）。

**配套原則**：

1. **State 存成品、初始化邏輯留 owner service。** State 是被動容器，只提供「接收已造好對象」的 mutator
   （`add_tab(tab)` / `put_device(dev)`），**不懂怎麼算初值**。`TabService.new_tab` / `DeviceService.
   _on_connect_succeeded` 才是造初值的地方。「把初始化搬進 State」是反模式 —— 你搬的應是**成品**，不是邏輯。
2. **persist = 選擇性投影，非全量序列化。** persist 只挑 State 裡它關心的可序列化欄位投影（Phase 98 只挑
   device 的 type/name/address），對 `ExpContext.soc`、figure 等視而不見。故 State 裡可序列化真相與不可序列化
   live handle 並存無礙。見 [[project_gui_device_state_ssot]] 的「persistence 是 State 投影」。

## 未納入

- 不立 service 分層 / tier。
- 不引入 DI 容器：`build_app_services` 的顯式構造注入（按拓樸序）已足夠表達單向 command 邊。
