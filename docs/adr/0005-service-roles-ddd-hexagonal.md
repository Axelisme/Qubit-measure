---
status: accepted
---

# Service 角色規範（DDD + Hexagonal）

**狀態：** accepted（規範）。對應重構（M1–M6）已完成，本檔以現在式描述角色規範與最終結構。
**關聯：** 與 [[0004]]（三問規則）互補——三問是「app service 之間怎麼連」的戰術，本檔是「每個 service 是哪種角色」的戰略；M6 正名由 [[0013]] 落地。

## 脈絡

`gui/services/` 曾平鋪 14 個 "service"，但 "service" 一詞無責任約束 → 按**話題**聚合（「跟 X 有關的都丟 XService」）而非按**角色**聚合，導致單一類揉多種責任、依賴複雜。經查證，反覆撞到的三處不適（cfg_editor 退化 handle、startup 該不該存在、service 太鬆）對應到 DDD/Hexagonal 的三條明確規範違反。本檔把四角色直覺對齊標準詞彙並逐字引用權威來源。

**權威來源：** `/sairyss/domain-driven-hexagon`（Context7，High reputation）。以下「原文」均為逐字引用。

## 角色定義（直覺 ↔ 標準詞，附逐字原文）

### 1. Driving / Primary Adapter（「像 viewer」）

> 「Interface adapters (also called **driving/primary** adapters) are **user-facing** interfaces … **User can be either a person using an application or another server.**」
> 「Controller … **One controller per use case** is good practice.」

→ **`MainWindow`（Qt，人）與 `RemoteControlAdapter`（NDJSON，agent=another server）是兩個 driving adapter**，平級、共用同一 application 入口（[[0013]]）。

### 2. Application Service（「被動、介面易懂、穩定」）

> 「orchestrate how the outside world interacts with your application … **Contain no domain-specific business logic** … **Uses ports** to declare dependencies … **Should not depend on other application services since it may cause problems (like cyclic dependencies).**」

→ 三條硬約束：**(a) 不含 domain 邏輯、(b) 經 port 依賴基礎設施、(c) 不依賴其他 application service**。

### 3. Aggregate Root（「建立物件→拿 handle，handle 本身提供行為保證」）

> 「Aggregate root is an entity that contains other entities/value objects **and all logic to operate them** … is a **gateway** … Any references from outside the aggregate should **only** go to the aggregate root.」
> 「reference other Aggregate roots via their globally unique identifier (id). **Avoid holding a direct object reference.**」

→ 一等公民帶**自己的行為**；外界只透過它進出；跨 aggregate 用 **id** 引用。**反模式（Anemic）= entity 只是資料袋、邏輯全在 service。**

### 4. Repository（「拿到 handle/id」的發放方）

→ 只管 aggregate 的**生命週期**（造/查/存/毀），不含編輯行為（那在 aggregate 上）。

### 5. Driven / Secondary Adapter（基礎設施）

> 「Infrastructure adapters (also called **driven/secondary** adapters) … interact with external systems … **not supposed to be called directly … only through ports (interfaces).**」

→ persistence / device driver / socket I/O。**必須經 port 被呼叫，不可被直接 import。**

## 三大系統性違規（三處不適的根）

1. **貧血 Aggregate**：`TabState` / `DeviceState` / `CfgEditorSession` 曾是哑 dataclass，行為全在 service。
2. **App service 互依**：`tab_view→{tab,writeback,context}`、`workspace→tab`、`startup→{context,device}`（違反約束 c）。
3. **基礎設施未經 port**：service 直接 `import StartupPersistenceService` / 直接碰 driver / 持 `_EditorCtrl`（違反約束 b）。

## 最終結構（M1–M6 重構結果，現在式）

順序原則：**先立 port 邊界 → 再富化 aggregate → 再斷 app-service 互依 → 目錄最後**。

- **port 邊界（解違規 3）**：`gui/app/main/services/ports.py` 持 `PersistOriginatorPort` / `ProjectIOPort` / `DriverFactoryPort` / `ContextReadPort` / `ContextWritePort`（皆 `Protocol`，interface segregation）。infra 經 port 注入；`build_app_services` 顯式構造注入（按拓樸序）已足夠表達單向 command 邊（不引入 DI 容器）。
  - 邊界判斷的誠實邊角：`Runner`/`AnalyzeRunner`/`SaveDataRunner` 是 Qt-signal QObject bridge（非外部系統），不包 port（會洩漏 Qt、收益低）；`GlobalDeviceManager` 是已隔離的硬體 I/O 邊界，未為它包 `DeviceRegistryPort`（`DriverFactoryPort` 已涵蓋真正有測試價值的 seam）。
- **富化 aggregate（解違規 1）**：`CfgEditorSession` 升為 aggregate root（`set_field`/`get`/`lower`/`commit` 行為上身，service 收斂為 Repository，見 [[0008]]）；`DeviceState` 加 `is_memory_only()`/`is_connected()`/`is_live()`、`TabState` 加 `is_busy()`/`has_run_result()`/…、`ExpContext` 加 `has_context()`/`is_active()`/`has_soc()`——entity 回答關於自己的問題。State 容器的 mutator + version bump 仍是 State 職責（[[0002]] 主線寫入不變式），謂詞只是 query。
- **斷 app-service 互依（解違規 2）**：pure query（tab_view→context / tab_view→tabs）改讀 State aggregate 謂詞，edge 消失；真 orchestration（tab_view→writeback / workspace→tab / startup→context+device）改依賴**窄 port**（非具體 sibling service）。import-discipline gate（AST 掃描）證 `gui/app/main/services` 內無 app-service→app-service concrete import。
- **目錄（M5）決定不做**：三大違規已由前述全數解決；vertical-slice 是導覽性重排（cosmetic），且本框架的 service 是**橫切 domain service**（guard / operation_gate / run / device …），不像範例的 feature-per-folder app 能乾淨切片。保持 `gui/app/main/services/` 平鋪。
- **M6（RemoteControlAdapter 正名）由 [[0013]] 落地**：`RemoteControlService` 擺正成與 `MainWindow` 平級的 driving adapter、持 typed `Controller`（消 `getattr` 靜默降級）。

## 已做決策 / 邊界

- 四角色直覺 ≈ DDD+Hexagonal 標準詞，**不發明私有黑話**。
- 健康（名實相符）：`guard`（發 Permit 憑證，純讀 State）/ `view_query`。
- 不立 service 分層 / tier（[[0004]]）；不引入 DI 容器、CQRS / command bus（wire 層 + Controller façade 已承擔）。
