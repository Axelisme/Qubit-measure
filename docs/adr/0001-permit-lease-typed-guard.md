# 用型別強制的 Permit 做 domain guard，與 OperationLease 分離

## 脈絡

GUI 有兩個 client（Qt View 與 remote RPC）驅動同一批受保護操作（run / save / analyze / writeback）。先前 guard 散在 UI handler 與 dispatch handler 兩處各寫一份，行為漂移（例：UI 擋無效 cfg、remote 不擋）。

## 決策

受保護 service 方法不接受裸 `tab_id`，而是接受一個 **Permit** 憑證；Permit 只能向統一的 `GuardService` 取得，取得時做完整 domain guard（context readiness、committed cfg validity、capability 需求）。拿不到 permit 即無法呼叫 —— guard 失敗在 acquire 階段 fast-fail，且 pyright 在編譯期擋住「忘記檢查」。兩個 client 都必經 GuardService，邏輯天然一致。

Permit 與既有 `OperationLease`（`OperationGate`）**分離為兩個物件**：
- **Permit** = 呼叫前可靜態證明的前置條件，純憑證、無需釋放。
- **Lease** = 操作當下才能判定、隨生命週期變化的動態資源（hardware 互斥、`is_tab_busy`），有生命週期、terminal path 釋放。

`is_tab_busy` 刻意歸 Lease 不進 Permit —— 它隨 run 生命週期變化，permit 證明它「曾經不 busy」毫無意義。

## 考慮過的替代方案

- **service 內部各自檢查（紀律）**：只解決邏輯重複、沒解決「忘記呼叫」；型別系統無從強制。
- **集中 GuardService 但 runtime 檢查（不發 permit）**：漏洞從兩處變一處，仍非型別強制。
- **lease 即 permit（擴充 OperationGate 收 domain 檢查）**：把靜態 domain readiness 塞進純 hardware exclusion authority，破壞 `OperationGate` 已凍結的職責契約。

選型別強制 + 分離，因為它根治「容易漏檢」且與既有 `OperationLease` proof-token 先例同構（最小驚訝）。

## 範圍

只涵蓋 run / save / analyze / writeback。device mutation 不套 permit —— 其前置幾乎全是動態 hardware 互斥（歸 Lease），強套只會製造空殼型別。
