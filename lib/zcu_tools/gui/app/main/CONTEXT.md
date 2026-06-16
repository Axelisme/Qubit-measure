# GUI Framework

`zcu_tools/gui/` 是雙 client 的 PyQt 框架：View（Qt UI）與 RemoteControlAdapter（NDJSON RPC，給 MCP agent）。兩個 client 操作必須一對一對應、共用同一帶 guard 的邏輯路徑。

## Language

### 前置條件與資源

**Permit**:
證明一個受保護操作「呼叫前可靜態成立」的憑證。涵蓋**靜態前置條件**：context readiness、committed cfg validity、capability（如 SoC）需求。由 `GuardService` 統一發放；拿不到即 fail-fast。是純憑證，**無需釋放**。型別按 guard 組合分立（Run / Save / Analyze / Writeback），讓型別系統擋住「拿錯 permit」。
_Avoid_: guard token, ticket, voucher

**Lease**（`OperationLease`）:
證明一個受保護操作「此刻動態資源可用」的租約。涵蓋**動態互斥**：hardware 操作互斥（RUN / SOC_CONNECT / DEVICE_*）、tab busy。由 `OperationGate` 在操作當下 acquire，有生命週期、**必須在 terminal path 釋放**（`release(lease, outcome)`）。

**Async operation handle**（`operation_id` = lease token）:
一個進行中 async 操作（device.setup / run.start）的 async-task 式 handle（connect 已改同步 `soc.connect`，不再是 handle）。`OperationGate` 是統合 façade，內部**責任分離**：`_OperationExclusion`（互斥，release 即移、讓出 hardware）+ `_OperationRegistry`（**所有 async task 工具的統一入口** —— per-token outcome/Event + cancel/poll/await）。生命週期綁死（一個 operation 同時是互斥的+可 await 的，同一 token 串起兩者，不拆成可選疊加），但三個 async 動詞統一在 Registry：
- **await**（`await_outcome(token, timeout)`）：阻塞取 outcome，**off-main only**（不卡 Qt 主線，靠 off-main thread + threading.Event 模擬，**非 asyncio**）。
- **poll**（`poll(token)`）：非阻塞查 outcome（pending → None）。**主線唯一能用的等待原語**（主線不能阻塞 await，用 QTimer 週期 poll 替代）。
- **cancel**（`cancel(token)` / `cancel_all() → list[token]`）：**異步通知式** —— 只 `set` 該 token 的 stop_event 即返回，**不等待**。「停了沒」由之後 `poll` 查。register 時把 worker 的 **stop_event（純數據 handle，非 callback）** 一起交給 Registry，故 Registry 是 cancel 唯一入口（不再散在各 worker by-name）。**worker 自己負責把 stop 信號翻成 cancelled outcome**（`stop_event.is_set()` → emit cancelled，run/device worker 對齊）—— cancel 端純信號傳遞、無副作用、無 callback IoC（符合「用傳遞不用共享」）。cancel 是 handle 的天然能力（≈ Go context：Registry 持 sender=event.set，worker 持 receiver=event+自判），**非**互斥職責，不違背下方 _Avoid_。

**outcome = `OperationOutcome`**（中性 status finished/failed/cancelled + error，**不帶 result**；result 走原有 snapshot/query）。啟動回 `operation_id`（= token），`operation.await(operation_id)` 阻塞取 outcome；查無 token=視為 finished（不 hang）。**三層分工**同版本號：operation_id 是 RPC↔mcp 簿記，agent **從不見**裸 id —— mcp 持「語義 key→最新 operation_id」對照（device→name / run→tab_id），agent 用語義名（如 `gui_device_wait_operation(name)`，涵蓋 connect/disconnect/setup）await。

**cancel 不保證即停**：cancel 是「請求」，能否停是 operation 自己的事。run/device 有 stop_checkers 輪詢點會停；**connect 是阻塞網絡調用、無 stop 點 → 收到 cancel 仍跑到自然完成/超時**。關閉流程（closeEvent / app.shutdown）`cancel_all` 後用 QTimer poll 等所有 token settle，**超時則強關**（kill 沒停的 connect）。
_Avoid_: 讓 outcome 帶 result payload、把 operation_id 暴露給 agent、把**互斥邏輯**混進 Registry（cancel 是 handle 能力、不是互斥；_Avoid_ 防的是把 exclusion 塞進 Registry）、期望 await 真協程讓出（是 off-main thread 模擬，見 qasync spike 備案）、期望 cancel 同步等待（會死鎖主線 / 卡死在不可中斷的 connect）
_Avoid_: guard, lock token

**關閉協調 / `ShutdownCoordinator`**（ADR-0003）:
GUI 關閉（user closeEvent / agent app.shutdown）時「**中斷所有 in-flight operation → 等它們停 → 全停或超時才真關**」的編排。**Qt-free 純邏輯**：`begin()` = `gate.cancel_all()`（拿全部 token）；`tick() → state`（WAITING/SETTLED/TIMED_OUT，每 tick 對所有 token `gate.poll` + 比 deadline）。**主線不能阻塞 await，故用「週期 tick + 非阻塞 poll」取代**（項目第一個週期計時器）。分層 = Progress 重構同款（ADR-0005 Hexagonal）：coordinator 純邏輯可單測無 Qt；**QTimer 包在 driven adapter** `QtShutdownDriver`（`adapters/qt_shutdown_driver.py`）驅動 `tick()`；`Controller`（Qt-free façade）暴露 `begin_shutdown(on_closed)`（懶建 driver）+ `active_operation_count()`；`MainWindow` closeEvent/request_shutdown 調它、傳 `_perform_close` 當 on_closed（user close 保留確認框，`_closing` guard 放行 `_perform_close` 觸發的二次 closeEvent）。
_Avoid_: 在 coordinator 裡 import qtpy / 監聽 Qt signal（破壞 Qt-free + 回到「訂閱事件」；統一用 poll）、把輪詢狀態機塞進 Controller（Qt-free façade）或 MainWindow（UI）、用屬性 flag 在 closeEvent/回調間傳「在等誰」（用 coordinator 自己的局部 token 列表 = 執行上下文，非跨對象共享）

**靜態 vs 動態邊界**（Permit 與 Lease 的分工基石）:

- Permit = 呼叫前可靜態證明的前置條件（不隨操作生命週期變化）。
- Lease = 操作當下才能判定、會隨生命週期變化的資源可用性。
- `is_tab_busy` 歸**動態**（隨 run 生命週期變化）→ 留在 service 內部 acquire lease 前後檢查，**不進 Permit**。

### Client 與守門

**Client**:
驅動 GUI 操作的入口。恰有兩個：**View**（Qt UI 點擊）與 **RemoteControlAdapter**（agent 的 RPC）。兩者地位平級，必經同一 guard 路徑。
_Avoid_: caller, frontend

**GuardService**:
集中所有 domain guard 邏輯、統一發放 Permit 的 query service。讀 `State` 與 `ExpContext.readiness`，無副作用。是 guard 邏輯的單一所有者（避免散在各 client 漂移）。

**Controller**:
View 專用的便利 façade —— 事件回調、error dialog 呈現。**不再是 guard 的擁有者**（guard 下放至受保護 service 方法，憑 Permit 型別強制）。

**View 渠道**（ADR-0013，取代舊 ViewQueryService）:
Controller 對 View 的下行拆成三個接口，按 cardinality + 機制分：**DiagnosticSink**（多個，fan-out，`notify_diagnostic(severity∈error/info, title, message)`，**不經 EventBus**）、**RenderHost**（單一可選，pbar/container，run/analyze 用，headless 為 None）、**RenderView**（snapshot/screenshot/dialog 純讀，由 `RemoteControlAdapter` 持 `render_view` 直接拉、不經 Controller）。`MainWindow` 實作全三者；`RemoteControlAdapter` 是 DiagnosticSink + 持 RenderView。cfg 欄位編輯走 tab 的 `CfgEditorService` session（`editor.set_field`，ADR-0013 F11），與 form attach 同一棵 model；`tab.update_cfg`（codec-based、replace committed）是另一條語義。
_Avoid_: 把診斷架在 EventBus 上、讓 Controller 當 render 查詢的二傳手、把「agent 該被動知道什麼」當 wire 機制（那是實驗語義，歸 mcp 端 default-subscribe）

**LiveModel**（`SectionLiveField` 樹）:
cfg 編輯的 runtime draft SSOT，**本身 Qt-free**（`on_change` 是純 `CallbackList`，非 Qt signal）。由 `SectionLiveField(spec, env, initial_val=...)` 從一份 spec 建立；`env`（`LiveModelEnv`）經 `ControllerProtocol` 取 md/ml/device。**model 永遠由 `CfgEditorService` 持有（ADR-0008）**；`CfgFormWidget` 是可插拔 viewer，`attach(model)` 顯示+反映、`detach()` 走但不 teardown。LiveModel 脫離 View 獨立存在；agent 與 user 都經這棵 session model 編輯（無第二條繞過 View-model 的路徑——ADR-0013 F11 移除了它）。
_Avoid_: 把 LiveModel 當成 View 的一部分、讓 widget own/teardown model

**CfgEditor session**（`CfgEditorService`，見 ADR-0008；其「演化」段記述並取代舊 delegated 設計）:
一個由 `CfgEditorService` 持有、用 `editor_id` 索引的**有狀態 LiveModel 編輯 session**。所有 model 都 service-owned；widget 與 agent 都只持 `editor_id`、都 attach/編輯同一棵（WYSIWYG）。model 可在沒有任何 widget attach 時存在（agent 可先於/不需 widget 編輯）。

- **ml-entry 編輯（`open`，預設 gc=True）**:agent 跨多次 RPC 漸進編輯 ModuleLibrary entry，`commit` 時 lowering（EvalValue 解析 against md）後落地。受 LRU + 斷線回收（孤兒保護）。
- **UI-owned 種子 session（`open_seeded`，gc=False）**:tab cfg（種子 = `State.cfg_schema`）、inspect、writeback 草稿（種子 = item 的 edit_schema）。無 item_kind → teardown-only（拒絕 commit）。**只由 owner 顯式 `teardown(editor_id)`**，不受 LRU、不受斷線回收。owner widget 先 `detach` 再讓 service teardown。

**draft vs committed 關係**:session 是 **draft**，`State.cfg_schema` 是 **committed**（run/save/session-persist 讀的 SSOT）。tab session 的改動經 auto-commit **即時同步**進 `State.cfg_schema`；run/save 前另有一道**強制 commit = valid 驗證閘**。

_Avoid_: 把「強制 commit」當成第二層 committed state、讓 widget 自建 model

- **為何有狀態（不可用瞬時取代）**：含 ref 切換（readout→pulse readout、waveform→const…），切換**動態改變後續可填欄位**（partial re-binding）。client 必須「切 ref → 看新欄位 → 再填」漸進進行。
- **生命週期按 `gc` 分流**（取代舊的 headless/delegated 兩 kind）：`gc=True` 受 LRU + 斷線回收；`gc=False` owner 顯式 teardown。`open_seeded` 同 owner_key 再開會先 teardown 前一棵。
- **external refresh 歸 service（ADR-0004 Reaction）**：service 訂 `MD/ML/CONTEXT/DEVICE_CHANGED`，對每棵 owned model 呼 `refresh_external` 刷 EvalValue；attached widget 經 model `on_change` 免費重畫。widget **不**碰 EventBus。
- **變更通知**：任一 client 改動 → session `on_change` 廣播。widget 經既有 `_updating` flag 斷回授；agent 經 **editor 專屬變更流**（只推給訂閱該 editor_id 的 client，**不走全域 EventBus**）。
- **失效訪問**：任何原因消失的 editor_id（LRU / tab close / commit / discard / teardown / 斷線）一律回 `unknown editor session`（INVALID_PARAMS），**不區分原因**。
_Avoid_: 為失效原因加 reason、讓 cfg 欄位變更走全域 EventBus、把 gc=False session 也納入 LRU/斷線回收

**Writeback persistent draft**（ADR-0008）:
analyze 完成時一次算出 writeback items 存 `TabState.writeback_items`；preview/UI/agent/apply 全讀/改同一份（不重算，重算會丟編輯）。每個 module/waveform item 建一棵 gc=False session（種子=edit_schema）、持 `editor_id`；agent 經 `editor.set_field` 改、user 點 Edit 時 widget attach 同一 model。識別碼 `session_id`（`<kind>-<n>`）穩定且與套用目標名 `target_name`（可改）解耦。`writeback.set` 改 selected/target_name/proposed_value（cfg 走 editor.*）；`writeback.apply` 讀持久草稿、snapshot 各 item 的 model → lower、不收 selections。rerun/reanalyze 先 teardown 舊 model 再重算。
_Avoid_: 每次 preview/apply 重算 items、讓 agent 直接改 State 繞過 model（user 會看到過期 cfg）、用 target_name 當識別碼

**Role template / `create_from_role`**（憑空建 ml entry，唯一 create 入口）:
為每種角色提供具名 factory（`role_catalog`，gui 介面 + experiment 啟動填充，倒置同 `register_all`）。兩類 role：**md-aware**（res_probe/bath_reset…，eval 預設、lower 成 md **當前值**）＋ **`:blank`**（`<disc>:blank`，每 discriminator 一個，結構零值，涵蓋裸 pulse 與 drag/flat_top/gauss/arb 等無 md-aware role 的形狀）。「選 role + 給名字 = **一次性**建好落 ml（ml 只存 concrete）」，**建立時不編輯**；要改走 modify。**create=新建語義，撞名 fail-fast**（與 `editor.commit`/`set_ml_module` 的 upsert-覆寫區分）。雙端同一機制：user 經 inspect「Create…」、agent 經 `ml.list_roles` + `ml.create_from_role`。**create 與 modify 彻底分開**：`editor.open` 已 from_name-only（modify 既有），`discriminator` 從 RPC 移除（內部 seed 保留）；inspect `_MlConfigDialog` 拆成 `_MlCreateDialog`(role) + `_MlModifyDialog`(固定形狀,**不換 type**)。
_Avoid_: 在 create 路徑混進編輯/換形狀（換形狀=刪了重建）、留兩條憑空建入口（discriminator 下拉 + role）、create 撞名靜默覆寫
_Avoid_: 用 role 推斷 discriminator（反過來：選 role 直接決定 factory）、讓 create 撞名靜默覆寫、user/agent 一端有 role 一端沒有

**懸空引用處理 + ml rename**:
cfg 用 `ModuleRefValue.chosen_key`（庫名）by-name 引用 ml entry。entry 被刪/改名 → 引用方在 `ML_CHANGED` **按 binding state 分流**（刻意不對稱，各取所長）：**LINKED**（純庫引用、無 override）→ 保留 chosen_key、標 missing+invalid+紅字，**可復原**（register 同名即自動 re-link）；**MODIFIED**（user 改過 value）→ 降級 `<Custom:label>` 保留編輯（已偏離庫、不復原但 override 保住）。`rename(old,new)` 是**通告式**：`delete+register+emit 一次`，**不遷移引用名**（引用靠上述分流）。撞名 fail-fast。**md-side**：EvalValue 懸空**維持 invalid 不 fallback**（無上次值可保、等同輸錯名）。
_Avoid_: 不分 binding state 一律 self-heal（LINKED 失去 re-link 復原、MODIFIED 失去 override）、讓 rename 去掃 cfg 遷移引用名（刻意不做）、給 md 懸空做 fallback

### Spec / Value / Adapter Default

**Spec 樹**:
一份 cfg 的**靜態結構契約**——欄位名、型別、label、choices、`editable`（語義可改性）、以及 `LiteralSpec.value`（固定值）。由 adapter 的 `cfg_spec()`（classmethod，**不讀 ctx**）建立。契約是 **static, never mutated**：所有 spec 節點 `frozen=True`，任何覆寫都回**新的 frozen 物件**（`dataclasses.replace`），不 in-place 改。**spec 不帶 GUI 渲染概念**：「要不要顯示 widget」是 GUI 決策（如 GUI 對所有 `LiteralSpec` 都不畫 widget），不是 spec 欄位；`editable` 保留是因為它是**語義**屬性（這欄位該不該被使用者改），非渲染指令。
_Avoid_: 把使用者填的值放進 spec、in-place 改 spec、把 spec 當可變 builder、在 spec 加「hidden/visible」這類純渲染旗標

**Value 樹**:
一份 cfg 的**使用者可編輯狀態**——`DirectValue`/`EvalValue`（scalar）、`SweepValue`、`ModuleRefValue`/`WaveformRefValue` 的選擇 + 子值、`None`（停用 optional ref）。由 `make_default_value(ctx)`（讀 md/ml）建立。**value 樹永遠完整**（每個 spec 欄位都有 entry、無缺 key，ADR-0010）；`fields: dict[str, Optional[CfgNodeValue]]`。Spec 與 Value 是**兩棵透過欄位名對齊的平行樹**，各自獨立建。
_Avoid_: 期望 value 節點能表達「不可編輯」（那是 spec 的 `editable`（語義）/ `LiteralSpec`（固定值））；把「停用（`None`）」與「選 None Reset」（實驗層真 reset）混為一談

**「空」的表示（ADR-0010，取代舊 `DisabledRefValue`）**:
value 樹裡一切「空」統一用 `None`，由 value 自述、不靠旁路 flag、不反推 spec：
- **停用 optional `ModuleRef`/`WaveformRef`** = `fields[k] is None`（裸 `None`，無 payload；重新啟用走 helper 預設）。`make_*_ref_default(optional=True)` 庫無時回 `None`，adapter 直接放進 fields（免 `if x is not None`）。`ModuleRefLiveField` 停用 `get_value()` 回 `None`、`set_value(None)` 設停用；父 `SectionLiveField` 無條件收集子層自述，不省略 key。
- **未填 scalar** = `DirectValue(value=None)`（**包裝層保留**以保 direct/eval 模式身份；scalar 型別合法值永不為 None，故 None 一義表未填，無 `is_unset` flag）。
- **「停用→消失」只在 lowering**（run/save 出口 omit）；persist 忠實序列化停用 ref 為 `{"__kind":"disabled"}`、還原回 `None`。
_Avoid_: 用 scalar 裸 `None`（抹掉 direct/eval 模式）；停用 ref 記 chosen_key（違無 payload）；改 `make_default_value` 全域行為去迁就單一 adapter 偏好（偏好走 OO 鏈式/工廠）；用 `None Reset` 表達停用（那是真 reset 選擇）

**鎖定欄位（Locked field）**:
adapter 宣告某欄位「固定、不參與編輯」（notebook 的 `freq: 0.0, # not used`）。**屬於 Spec 樹**，用既有 `LiteralSpec(value)` 表達：lowering 永遠取 `spec.value`、`set_value` no-op；**GUI 對所有 `LiteralSpec` 都不畫 widget**（渲染決策在 widget 層 `containers.py`，不是 spec 旗標）。spec-only，不碰 value、不需 editable。
_Avoid_: 把鎖定放進 value 樹、用 spec 的 hidden 旗標表達「不畫」（渲染由 widget 看 LiteralSpec 自決）

**Spec 覆寫（Spec override，spec 層）**:
adapter 在 `cfg_spec()` **內**拿 helper 回傳的深層 spec 樹後，把某個葉換成 `LiteralSpec`（鎖定）。機制是 **frozen-recursive `dataclasses.replace` 回新 frozen spec**：method **掛在 spec 型別上**（`CfgSectionSpec.lock_literal(path, value)` 與 `ModuleRefSpec.lock_literal(path, value)`），回**同型新 frozen spec**，無 wrapper、無 `.build()`/`.done()`。多覆寫靠「回同型」自然鏈式（`spec.lock_literal(p1, 0).lock_literal(p2, 0)`）。**鏈式起點可從根 `CfgSectionSpec` 或子樹 `ModuleRefSpec`**：對 helper 回的 readout 子樹直接 `make_pulse_readout_module_spec().lock_literal("pulse_cfg.freq", 0.0)`（path 從子樹起算、較短、鎖定與該 spec 內聚）勝過從根走 `modules.readout.pulse_cfg.freq`。path 走 dotted 字串，每種 spec 型別知道怎麼往自己的子結構遞迴：`CfgSectionSpec` 走 `fields`、`ModuleRefSpec` 走 `allowed`（**duck-type：含該 path 的 allowed shape 就套、不含就跳；全部不含才 raise**）。目前只有 `lock_literal`（`readonly`/`hidden` 之類無真實使用者，未提供——要設 `editable=False` 直接 `ScalarSpec(editable=False)`）。
_Avoid_: 可變 SpecBuilder 鏡像、`.build()`/`.done()` 收尾、把 spec method 寫成回 wrapper（回 spec 才能無 done 鏈式）、為對稱而加無使用者的 fluent（YAGNI）

**Default factory**（`make_*_default` / `make_*_ref_default`）:
產生 value 樹預設的兩層 helper。**第一層**（`make_*_default`）按 module type 組 blank value + 填 md 衍生預設;**第二層**（`make_*_ref_default`）先查 ml 庫有無 `preferred_names` 命名 module，有就引用、無則 fallback 到第一層。adapter 覆寫 default 用 **value 上的 OO 鏈式方法**（`make_pulse_readout_default(ctx).with_gain(0.05).with_ro_length(1.4)`）取代 factory 長參數列。**不在 default factory 裡鎖欄位**（鎖是 spec 的事）。
_Avoid_: 在 default factory 摻入 spec/鎖定邏輯、把覆寫寫成一長串 factory 參數

**角色 default（語義 wrapper）**:
對齊 notebook 真實角色的薄 wrapper，把「該角色的固定 fallback 策略」封進名字，使用上是獨立 helper，實作上委派少數通用 default factory：`default_pi`（優先 pi_amp→pi_len）、`default_pi2`（優先 pi2_amp→pi2_len→降級 pi）、`default_qub_probe`（blank pulse + ch/waveform/freq value 預設）、`default_res_probe`、`default_reset`。fallback 是 notebook 觀察到的**一致策略**（非單純 rename）。**default factory 零鎖定**：只產 value 預設，不預設鎖任何欄位（即使高頻），鎖定一律由 adapter 在 cfg_spec 用 lock_literal 宣告——鎖屬 spec 層、factory 屬 value 層，不跨層。
_Avoid_: 每個角色重寫一份 factory（應委派通用 helper + 填 preferred_names/結構）、把角色 wrapper 當成只換名（它封裝 fallback 策略）、在 default factory 預設鎖欄位（即使高頻也歸 adapter 的 cfg_spec）

**Value OO 覆寫（value 層，與 spec 層不對稱）**:
value 容器（`CfgSectionValue`/`ModuleRefValue`）上的 `with_*` 方法 **in-place 改 fields 再回 self**（value 樹本就可變、現有 `_patch_*_fields` 本就 in-place、且 default factory 每次新建全新 value 樹不跨呼叫共享 → in-place 安全）。**刻意不對稱於 spec 層**：spec 層回新 frozen（spec never-mutated 契約），value 層 mutate-and-return-self。讀者須知 `spec.lock_literal()` 是衍生新物件、`value.with_gain()` 是就地改。
_Avoid_: 把 value `with_*` 寫成回新物件（與 spec 對稱但破壞「default factory 即 _patch_ 同構」最小改動）、期望 value 物件被多處共享（每次 factory 呼叫新建）

### 並發感知（agent vs 人）

**並發感知三層分工**（脊椎，見 `docs/adr/0002`）:

- **RPC = mechanism**:持版本表、提供 `resources.versions`、`expected_versions` 原子比對。
- **mcp = policy + 簿記 + 翻譯**:持 last-seen 版本、知依賴對應、組 `expected_versions`、把拒絕翻成語義。版本號**只在 RPC↔mcp 流動**。
- **agent（LLM）= 只收語義**「tab X cfg 過時了」,**從不看到版本號**。

**Resource version**（`State.version` = `VersionTable`）:
每個資源的單調遞增整數(per-resource,非 wall-clock)。資源 key 中粒度:`context` / `soc` / `device:<name>` / `tab:<id>:cfg`·`:result`·`:save_path` / `tab:<id>`(存在) / `editor:<id>`。tab 資源綁 `tab_id`(uuid4,永不重用)。**bump 的責任歸該資源的 owner service,且只在 Qt 主線**(state mutator / 各 service terminal slot / `DeviceService._emit_device_changed` / `Controller.bump_editor_version`)—— 是「State 寫入只在主線」不變式的推論。VersionTable 是被動容器(只 +1);tab close 時 `drop_prefix` 忘掉該 tab 全部 key(依賴一個 dropped key 讀作 0 = 視同 stale)。
**version bump = 狀態真的變了，不含「值未變的快取同步」**:bump 對應「資源狀態實際改變」(client 寫入,或讀取時發現外部變化);**讀取的快取刷新**(`get_device_info` 把 driver 回值寫回 `State.devices[name].info`,經 `refresh_device_info_cache`)的判定為:**值與快取相同 → 純同步,不 bump、不 emit**(否則一次純讀會 spurious 推進版本號、誤使其他 client 的 `expected_versions` 失效);**值不同(pydantic `!=`)→ driver 值在外部變了 = 真實狀態變化,bump `device:<name>` + emit `DEVICE_CHANGED`**(讓 readers requery、讓依賴此值的 guard 能擋)。原則精確化:不 bump 的是「值未變的同步」,不是「所有讀取路徑」。
_Avoid_: 讓「值未變的快取同步」bump;讓「讀到值真的變了」靜默不 bump/不 emit

_Avoid_: 用 wall-clock 時戳取代版本號、在 worker thread bump、把 bump 散進每個 emit(綁 owner 而非 emit)、期望版本號區分「誰改的」(只答「變了沒」)

**Version guard**（`_guard_versions`）:
一道 **optimistic-concurrency 閘**(If-Match 式):受護操作(`run.start` / `save.*` / `editor.commit`)帶 optional `expected_versions`(資源→版本),server 在主線 `_dispatch._run()` 單一同步序列內**原子**比對當前版本;不符(含 key 不存在=資源已 drop)→ `PRECONDITION_FAILED(reason=stale_version)`。沒帶=不檢查(同普通 RPC)。比對與真人 GUI 寫同在主線 → 無 TOCTOU。guard 是純 mechanism,**不懂依賴語義**(哪些 key 重要由 mcp 決定)。
_Avoid_: 把它當永久鎖、讓 RPC 端懂「什麼叫 stale」或「run 依賴什麼」(那是 mcp policy)、在比對前 auto-refresh(會抹掉真人的變動)

**expected_versions**（wire-only,MCP-hidden）:
guard 操作的 optional 參數,由 mcp 依 `_GUARD_DEPS`(依賴對應表:run→cfg/tab/soc/context/device:*;save→result/save_path;commit→editor/context)從 last-seen 組出。`ParamSpec.mcp_hidden=True` → 驗證+達 handler 但**不進 MCP inputSchema**(版本號不洩漏給 agent)。mcp 每次成功 RPC 後 `_refresh_versions()` 更新 last-seen(每次 round-trip = agent 觀察到當前,故不擋自己;真人在兩 RPC 間的改動才被擋)。
_Avoid_: 讓 expected_versions 出現在 agent schema、讓 agent 自己組版本、save 依賴 cfg(存檔來自 result 自帶 cfg_snapshot)

**Notification face**（EventBus subscribe + blocking poll,正交於版本表）:
agent 知道「什麼變了」走 EventBus —— `gui_events_subscribe` 訂閱事件名 + `gui_events_poll`(blocking,事件到或 timeout 才返)。event payload 帶受影響資源身份,**不帶版本號**。與版本 guard 是兩條獨立的線(通知=質化「變了什麼」、版本=量化「擋不擋」)。editor 專屬變更流(細節 push)同屬通知面。
_Avoid_: 把版本號塞進 event payload、把通知面與版本 guard 綁在一起、輪詢 snapshot 自己 diff(改用 event-driven poll)

**Off-main handler**（`MethodSpec.off_main_thread`）:
標記一個 wire method 的 handler **不 marshal 上 Qt main thread**,而在 IO worker thread 直接執行。唯一用途:blocking 等待型 handler（`operation.await`）—— 若上主線,handler 用 `evt.wait()` 阻塞會卡住 main thread event loop,而它等的 worker terminal signal 正需要 event loop 處理 → **死鎖到 timeout**。移出主線後 event loop 恢復轉動,`threading.Event` 跨 thread 喚醒正常。
_Avoid_: 讓 off-main handler 碰 main-thread-owned 狀態（version table / CfgEditor / `_snapshots` 等)、需要 version guard

- **嚴格契約**:off-main handler **只能做 thread-safe 的等待**,同步必須靠 `threading` 原生語（service 持 per-name `threading.Event`,main thread terminal 時 set,off-main handler 只 wait），**不得在 IO thread 讀寫 main-thread-owned dict 或 connect/disconnect Qt signal**。

### Wire 層型別契約

**ParamSpec**:
一個 wire method 的單一參數型別契約（name / json_type / required / default / description）。是型別的**唯一來源**：MCP inputSchema 從它生成、執行期 per-param 驗證也由它驅動，兩者永不漂移。handler 收到的是已驗證的 typed params。
_Avoid_: field schema, arg spec

**Coercion**（`coerce_*`）:
把多個已驗證的 wire 欄位**結構化組裝**成 domain request（frozen dataclass，如 `ConnectRequest`）的那一步。與 ParamSpec 互補：ParamSpec 管 per-param 型別，Coercion 管 multi-param→request。需要 Coercion 的操作 = MCP 生成的覆寫對象。
_Avoid_: per-param validation（那是 ParamSpec 的事）

### Service 互依（三問規則，見 `docs/adr/0004`）

**不維護 service 分層表。** 每條「A 要用到 B」的邊落地時只問一個局部問題 —— A 需要的是 B 的**答案**、要 B**做一件事**、還是想在 B**變化後反應**：

- **Query**（要 B 現在的值）→ 兩者都讀 **State**，A 不碰 B。最優先（State-as-SSOT 天生支持）。
- **Command**（叫 B 做事，且 B 不需回頭找 A）→ A 構造注入 B、直接呼叫。天然單向。
- **Reaction**（B 變了 A 要跟著動）→ A 訂閱 **EventBus**、不持有 B。依賴反轉，環被 bus 截斷。

**撞到「會成環」= 幾乎一定是把 Reaction 誤當 Command。** 改走 EventBus 即解（Phase 98 `startup` 反應 `DEVICE_CHANGED` 即此）。`ControllerProtocol` / `_EditorCtrl` 等 narrow Protocol 是「需呼叫 owner（Controller/View）能力」時的 Command 受控形式（依賴介面非具體類）。
_Avoid_: service 分層 tier、卡循環就往 ctrl 塞、把 reaction 寫成 command 互持

**狀態放哪（兩軸正交，勿坍縮）**：軸 1「進不進 State」= 除 owner 外還有誰要**讀**（有 → 進 State）；軸 2「persist 投不投影」= 重啟後有無意義（有 → 投影）。**不可序列化只影響軸 2，不影響軸 1** —— State 可持有不可序列化的共享活物件（如 `ExpContext.soc`：多 service 讀 → 進 State；重啟連線沒了 → persist 跳過）。配套：State 存**成品**、初始化邏輯留 owner service（`add_tab`/`put_device` 只收已造好對象）；persist 是**選擇性投影**非全量序列化。見 `docs/adr/0004`。
_Avoid_: 把「不可序列化」當「不能進 State」、把初始化邏輯搬進 State、persist 全量序列化整個 State

**Service 角色（DDD+Hexagonal，見 `docs/adr/0005`）**：`services/` 的東西按**角色**而非**話題**聚合，每個必須說清是哪種 —— **App Service**（被動編排、無 domain 邏輯、經 port 依賴 infra、不依賴其他 app service）、**Aggregate Root**（一等公民帶**自己的行為**，外界經 id 進出，反模式=貧血 dataclass）、**Repository**（造/查/毀 aggregate）、**Driving Adapter**（user-facing，`MainWindow`+`RemoteControlAdapter`=兩個 driving adapter，user 可以是人或 another server）、**Driven Adapter**（persistence/driver/socket，**只經 port 被呼叫**）。三大系統性違規已由 Phase 99（原 M1–M6）遷移消除：貧血 aggregate（M2/M3 升 aggregate root）、app-service 互依（M4 改窄 port / 直讀 State，AST gate `test_app_service_decoupling` 守）、infra 未經 port（M1 `services/ports.py`）。M5（目錄 vertical-slice）決定不做、M6（RemoteControlAdapter 正名）由 ADR-0013 落地。
_Avoid_: 按話題聚合 service、entity 寫成哑 dataclass（貧血）、app service 互相依賴、直接 import 基礎設施（繞過 port）

### 持久化（Persistence，Memento + Caretaker，見 `docs/adr/0015`）

**PersistenceCaretaker**（Memento 的 Caretaker，**Driven Adapter**）:
GUI app state 的單檔存讀者。由 `run_app` 在 app 層建立、注入 Controller 持有（窄 port `PersistOriginatorPort`，單向 Command）。**與 RemoteControlAdapter 不同級不同向**——後者 driving（驅動操作進系統），它 driven（被 lifecycle 觸發做 I/O）。職責只有 disk I/O + 何時讀/寫:**不訂 event、不碰 UI、不碰 State、不懂 cfg**，只收 Controller 吐的 memento。`flush()` **觸發無關**:每次呼叫都重新 capture 現組 memento 落盤;目前唯一觸發是 lifecycle（close→`persist_all`、startup→`restore_all`），未來可加 timer/button/event/RPC 觸發而 Caretaker 不改。
_Avoid_: 讓 Caretaker 訂 event / 碰 State / 認識 cfg、把「只在關閉寫」當成 Caretaker 的不變式（是當前唯一觸發、非設計上限）

**AppPersistedState memento**（`persistence_types.py`，pydantic v2 frozen）:
整個 GUI app state 的不可變快照——`{version, startup: PersistedStartup, session: PersistedSession}`，存單一 `gui_state_v1.json`。**單一 top-level `APP_STATE_VERSION`**（不分 startup/session 子版本）;`model_validate` 做 load 驗證（壞檔/版本不符 Fast-Fail→default），`model_dump` 產 JSON。是「選擇性投影」——capture 當下從 State/View 造出，**非 State 全量序列化**（State 可含不可序列化活物件，投影只取可序列化的偏好/tabs）。`cfg_raw` 對 Caretaker 不透明（其 raw↔live codec 是 WorkspaceService 內部實作 `session_codec`）。
_Avoid_: 讓 memento 帶 live/不可序列化物件、startup/session 各自版本號、把 cfg codec 洩漏進 Caretaker

**三層 capture/restore**（Originator 鏈）:
持久化是一條 `Caretaker（I/O）↔ Controller（單一 Originator，彙整/派發）↔ 各 service（capture/restore，序列化內化）` 的鏈。Controller `capture_persisted_state()`/`restore_persisted_state()` 是 Memento Originator 面;各 service（Workspace `capture_session`/`apply_session`、Startup `capture_startup`/`restore_startup`）各自吐/吃自己那半的 memento，序列化邏輯（cfg lowering、device 投影）內化於此、不外露。
_Avoid_: 讓 Caretaker reach into 各 service / State（god 依賴）、把 codec/投影抽成獨立物件（它是 service capture/restore 的內部實作）

**startup 預填值 `State.startup_prefs`**:
「記住的 startup 偏好」（project chip/qub/res/dir/db + ip/port + left_panel_width），**與 active `ExpContext` 分開的一塊 State**。語意是「下次該預填什麼」非「當前 active 什麼」。**連線值與預填值不分**——因為重啟不自動連線，套用/連線時就把用過的值同步寫進 `startup_prefs`（寫入當下），capture 只讀它，restore 寫它且**不自動套 active context**（project 等 user 在 setup dialog 套用）。device 記憶集**不**進 startup_prefs（它在 `State.devices` 的 `remember` 旗標，capture 時即時投影）。
_Avoid_: 把 startup_prefs 與 active ExpContext 混為一談、restore 時自動連線/套 project、capture 時才做「連線 vs 預填」二選一（改在寫入當下同步）

## 範例對話

Dev：「agent 送 run.start，要怎麼確保跟 UI 按 Run 行為一致？」
Expert：「兩個 client 都得先跟 GuardService 要一張 RunPermit。GuardService 檢查 context 是不是 active、committed cfg 過不過 lowering、需不需要 SoC。拿到 permit 才能呼叫 `start_run`。」
Dev：「那 tab 正在跑的時候呢？permit 會擋嗎？」
Expert：「不會，那是動態狀態，不歸 permit。permit 只證明『這個請求本身合法』。『此刻 tab 在不在跑』是 Lease 的事 —— `start_run` 內部 acquire OperationLease 時才判，因為它隨時會變。permit 證明靜態前置，lease 證明動態資源，兩者語義不同。」
