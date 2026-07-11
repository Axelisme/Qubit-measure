---
status: accepted
---

# Adapter capability contract — explicit `AdapterCapabilities`, validated at import

**狀態：** accepted（已實作）。本檔以現在式描述目前生效的契約。
**關聯：** spec/value fluent + role default 見 [[0009]]；value 樹完整性見 [[0010]]；成品邊界驗證見 [[0011]]；value 層組裝 builder 見 [[0012]]。

## 脈絡

adapter 要對框架宣告自己支援哪些 lifecycle：要不要 SoC、analyze 是 FIT 擬合 / INTERACTIVE 選點 / 無、有沒有第二層 post-analysis。早期 grilling（design_v1）探討過幾種「讓宣告與實作天然同步」的機制：capability-composition class-body 物件（DEC-1）、`@fit` 薄標記 decorator（DEC-3）、單一 typed 屬性（ii-a，DEC-4）。

系統分析 ~50 個 experiment cell × ~40 個 concrete adapter 後，v2 reconciliation 收斂到更小的方案：**一個顯式 `AdapterCapabilities` 宣告 + import-time validation**。理由：`AnalysisMode` enum 本身已表達三態互斥，宣告與實作的 desync 用 import-time Fast-Fail 抓即可，不需要 decorator / typed-attribute machinery，也不需要重寫 40 個 adapter。

## 決策

### 1. 單一顯式宣告 `AdapterCapabilities`

每個 concrete adapter 宣告 class 屬性
`capabilities = AdapterCapabilities(analysis=AnalysisMode.{NONE|FIT|INTERACTIVE}, requires_soc=…, post_analysis=…)`。
框架契約 `ExpAdapterProtocol`（generic-free Protocol）只認這個屬性 + 必備 method，不認 decorator 或約定俗成的 method-presence。

**被否決**：class-body capability 物件（DEC-1）、`@fit` decorator（DEC-3）、ii-a 單一 typed 屬性（DEC-4）。`AnalysisMode` enum 已表達互斥、import-time validation 已抓 desync，額外 machinery 是 YAGNI，且會逼 40 個 adapter 改寫。

### 2. `__init_subclass__` import-time Fast-Fail

`BaseAdapter.__init_subclass__` 在 class 定義／import 當下跑 `_validate_capability_contract`：宣告的 capability 與實作的 hook 不一致就 raise `TypeError`（早於任何 runtime 路徑）。規則：

- `analysis=FIT` → 必須實作 `analyze()`、禁止 `setup_interactive_analysis()`。
- `analysis=INTERACTIVE` → 反之（必須 `setup_interactive_analysis()`、禁止 `analyze()`）。
- `analysis=NONE` → 禁止全部 analyze hooks（含 `get_analyze_params`）。
- `post_analysis=True` → 要求 `analysis=FIT` 且實作 `get_post_analyze_params()` / `post_analyze()`。

### 3. MRO-aware `_is_method_implemented`（getattr-identity，不 hardcode base 清單）

判斷某 method 是否被覆寫用 `getattr(cls, name) is not getattr(BaseAdapter, name)`——對「cls 或其中間 class base」覆寫皆成立。**必須泛型**：`singleshot/t1_tone_sweep.py` 的中間 class base 把 `analyze()` 給子類繼承，naive 的 `name in cls.__dict__` 會誤報 false violation。**不**維護中間 base 白名單（`reset/*`、`bath`、`time_domain` 的 `_shared.py` 是 module-level free function／常數，不進任何 MRO，不需列入）。

### 4. analyze-params override 只在「params 需要值」時必須

`get_analyze_params` 的 base default 回 `params_cls()`（當 `_can_construct_without_args(params_cls)` 為真——每個欄位都有 default，含 `NoAnalyzeParams` 與「把常數折進欄位 default」的 params），否則 raise `NotImplementedError`。validation 對應放寬：只有 params class **無法全 default 建構**（有欄位無 default）時才要求 override。

兩半是同一條規則的鏡像：**base 能產生 params ⟺ `params_cls` 全 default 可建構 ⟺ override 非必須**。靜態取型的 `analyze_params_cls()`：adapter 有覆寫 `get_analyze_params` 時讀其 return annotation；否則（多數 adapter 把常數折進欄位 default、不覆寫）落到 class 的第 4 個 generic arg `BaseAdapter[..., <Params>]`。

### 5. Framework-called hook 明列為 mandatory Protocol surface

framework 實際呼叫的 adapter hook 都在 generic-free `ExpAdapterProtocol` 明列。
`validate_run_request(req, raw_cfg)` 是 mandatory surface、optional override：`BaseAdapter` 提供純
no-op default，`GuardService` 永遠直接呼叫。所有 adapter 都能執行這個 preflight seam，因此它不是
「支援／不支援」或 routing 差異，不加入 `AdapterCapabilities` bit。

保障分為三層且責任不同：`BaseAdapter.__init_subclass__` 在 import-time 驗 capability 與 conditional
hook 是否一致；pyright 驗 Protocol / Registry 的 structural signature compatibility；registry traversal runtime
test 驗 mandatory member presence。Base subclass 若原本想 override 卻拼錯 method name，仍會合法繼承
no-op default；本契約不宣稱能辨識這種實作者意圖。

## 後果

- adapter 作者只填 `capabilities` + 對應 hooks；裝錯／忘記在 **import 時**就炸，不到 runtime。
- 常數 analyze-params 不再需要樣板 override——值折進 params dataclass 欄位 default，base default 直接 `params_cls()`。
- 驗證是泛型的（getattr-identity），不隨新增中間 base 或 helper 檔失效。
- Guard 不保留 optional `getattr` compatibility branch；mandatory preflight 缺失由 structural contract
  與 registry conformance test 揭露。
- 取代 design_v2 reconciliation 中 DEC-1／DEC-3／DEC-4 對宣告機制的探索；analyze-params 規則放寬連動「常數折進 default」的去樣板批次。

## 替代方案（綜述）

| 維度 | 選擇 | 否決 |
| --- | --- | --- |
| capability 宣告 | 顯式 `AdapterCapabilities` + import-time validation | class-body 物件 / `@fit` decorator / ii-a typed 屬性 |
| 宣告↔實作 同步 | `__init_subclass__` Fast-Fail（getattr-identity） | runtime 才發現 / hardcode 中間 base 白名單 |
| analyze-params override | 只在 params 無法全 default 建構時必須 | 凡非 `NoAnalyzeParams` 都強制 override（樣板） |
| framework mandatory hook | Protocol 明列 + Base default + direct call | optional `getattr` / broad hook-name registry |
