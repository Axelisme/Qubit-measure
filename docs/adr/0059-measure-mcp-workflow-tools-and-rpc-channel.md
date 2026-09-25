# ADR-0059：measure-gui MCP 以 workflow tools 加 live RPC channel 取代 1:1 generated tools

**狀態：** accepted（未實作），部分被 [[0060]] 取代：決策 1 的 workflow tool 清單由 [[0060]] 的特化 tool 取代；決策 2–5（RPC channel、guard policy 宣告、exposure、不採用 code execution）沿用，RPC channel 對量測 agent 開放。
**關聯：** 修訂 [[0014]] 決策 4 的工具生成方式；沿用 [[0002]]（version guard / operation handle）、[[0013]]（remote adapter 為第二個 View）、[[0035]] 與 [[0047]]（tool error 契約）。

## Context

measure-gui MCP 目前暴露 81 個 tool：50 個由 `METHOD_SPECS` 1:1 generated，31 個 hand-written（bundle、lifecycle、override）。另有 12 個 wire method 為 `internal`。這個形狀有三個問題：

1. **選擇負擔與雙路徑。** 同一件事常有 bundle 與細粒度兩條路，例如 `gui_tab_run` 與 `gui_tab_set_cfg` + `gui_tab_run_start`、`gui_tab_analyze_review` 與 `gui_tab_analyze_start` + `gui_tab_get_analyze_result` + `gui_tab_writeback_list`。agent 每一步都在 81 個名字中挑選，run-measure-gui skill 必須額外教「優先用哪條」。
2. **tool 面在 MCP 啟動時凍結。** MCP server 在自己的 process import `METHOD_SPECS` 並產生 tool，policy table（`GUARD_DEPS`、`READ_REVEALS`、`OPERATION_KEY_OF`）也是 MCP 端的 import。GUI 端新增或修改 wire method 後，即使 `gui_stop` + `gui_launch` 重啟 GUI，agent 仍看不到也呼叫不到新形狀；Claude Code session 內無法重啟 stdio MCP server。這讓 agent 無法以 e2e 方式驗證自己的改動。
3. **exposure 層在維護第二份 API。** generated tool 已靠 `tool_name` 別名改名（`context.ml_get` → `gui_context_ml_list`、`tab.writeback_preview` → `gui_tab_writeback_list`），MCP 名稱與 wire 名稱逐漸分岔。

外部參考：Blender MCP 只暴露少量觀察 tool、截圖與 `execute_blender_code` 逃生口；Anthropic〈Writing tools for agents〉建議合併成 workflow 形狀的 tool，不把 endpoint 1:1 暴露。measure-gui 驅動實體儀器，任意程式碼執行會繞過 permit、version guard 與 operation handle，因此只採用「少量 workflow tool + 通用通道」的形狀，不採用 code execution。

## Decision

### 1. 三層 MCP surface

| 層 | 內容 | schema 來源 | 會隨 GUI 重啟更新 |
| --- | --- | --- | --- |
| **Workflow tools** | primary flow、MCP-only 組合、硬體安全動作 | MCP 端 typed `inputSchema` | 否 |
| **RPC channel** | `gui_rpc_list`、`gui_rpc_describe`、`gui_rpc_call` | live GUI 的 `rpc.catalog` | 是 |
| **Dev** | `gui_debug_operations`（讀 MCP session 狀態） | MCP 端 | 否 |

一個 wire method 對 agent 恰好只有一個入口：綁定到某個 workflow tool、經 RPC channel 呼叫，或 `internal`（agent 不可達）。三者互斥，不存在雙路徑。

Workflow tool 依七類主要操作組織。每類只特化常用操作，其餘能力留在 RPC channel；一個操作被特化，須至少符合其一：

- 屬於 run-measure-gui skill 教的 primary flow，或是該類別中最常用的讀取；
- 是 MCP-only 組合（bundle、lifecycle、operation wait、short-wait START），wire 上沒有單一對應；
- 直接改變或停止硬體狀態，typed schema 與說明文字本身是安全措施。

初始清單（35 個 workflow tool，加上 RPC channel 3 個與 dev 1 個；最終以待決問題 1 的使用數據校正）：

| 類別 | 特化的 workflow tools | 留在 RPC channel 的能力 |
| --- | --- | --- |
| **GUI 生命週期** | `gui_launch`、`gui_stop`、`gui_bridge_connect`、`gui_bridge_detach` | — |
| **Overview 讀取** | `gui_overview`、`gui_screenshot` | `resources.versions`、`tab.list_all`、`tab.snapshot`、`soc.info`、`result_scope.list` |
| **Project 操作** | `gui_project_apply`、`gui_soc_connect`、`gui_context_list`、`gui_context_switch`、`gui_context_create` | — |
| **Tab 操作** | open：`gui_tab_open`<br>cfg：`gui_tab_get_cfg`、`gui_tab_set_cfg`<br>run：`gui_tab_run`、`gui_tab_run_cancel`<br>analyze：`gui_tab_analyze_review`<br>post：`gui_tab_post_analyze_review`<br>writeback：`gui_tab_writeback_apply`<br>save：`gui_tab_save_data`、`gui_tab_save_image`<br>figure：`gui_tab_get_figure`<br>operation：`gui_op_wait` | `tab.new`、`tab.close`、`tab.set_active`、`tab.load_data`、`analyze.cancel`、`tab.get_analyze_params`、`tab.get_analyze_result`、`tab.get_post_analyze_params`、`tab.get_post_analyze_result`、`tab.writeback_preview`、`tab.writeback_set`、`adapter.list`、`adapter.guide` |
| **Inspect 操作** | `gui_context_ml_list`、`gui_context_ml_inspect`、`gui_context_md_read` | `value.list`、`value.read`、`context.ml_list_roles`、`arb_waveform.list`、`arb_waveform.preview` |
| **Device 操作** | `gui_device_list`、`gui_device_connect`、`gui_device_apply`、`gui_device_disconnect` | `device.snapshot`、`device.setup_spec`、`device.active_operations`、`device.cancel_operation`、`device.forget` |
| **Predictor 操作** | `gui_predictor_info`、`gui_predictor_predict` | `predictor.load`、`predictor.set_model_params`、`predictor.clear` |
| 使用者互動 | `gui_prompt_user` | — |

不屬於七類的寫入能力全部只經 RPC channel：ModuleLibrary 建立、改名、刪除（`context.ml_*`）、MetaDict 寫入與刪除（`context.md_set_attr`、`context.md_del_attr`）、cfg editor session（`editor.*`）、arbitrary waveform 寫入（`arb_waveform.set`）。

與現況相比的變動：

- `gui_op_wait` 吸收 `gui_op_poll`，`timeout=0` 即 poll。
- `gui_tab_post_analyze_review` 是新組合，對稱於 `gui_tab_analyze_review`：post-analyze START 加上 `post_analysis` pane 的 writeback 預覽。
- `gui_tab_run_start`、`gui_tab_analyze_start`、`gui_tab_post_analyze_start` 的 short-wait START 由對應 bundle 承接，不再另列 tool。
- `gui_context_md_write`、`gui_context_md_delete`、`gui_editor_*`、`gui_debug_resource_versions` 與 50 個 generated tool 中未列上表者移除，改經 RPC channel。
- 名稱沿用現有 tool；`gui_context_ml_list`、`gui_device_list`、`gui_predictor_info`、`gui_predictor_predict` 由 generated 轉為 `TOOL` exposure。

### 2. `rpc.catalog`：catalog 由 live GUI 提供

GUI 新增 wire method `rpc.catalog`，回傳每個非 `internal` method 的 `method`、`description`、`params`（沿用 `schema_property` 的 JSON schema）、`timeout_seconds`、exposure，以及決策 3 的 guard policy。

- `gui_rpc_list(domain?)` 回傳 method 名稱與一行說明，以 `domain`（`tab`、`context`、`device`⋯）過濾，控制 context 用量。
- `gui_rpc_describe(method)` 回傳完整 params schema 與說明，承接原 generated tool description 的使用指引。
- `gui_rpc_call(method, params)` 以 catalog 驗證 method 存在與 exposure，再走既有 guarded `send_gui_rpc`；transport timeout 取 catalog 的 `timeout_seconds` 加 slack，`operation.await` 仍要求 caller 明確給 timeout。
- MCP 在每次 connect／launch 後重新取 catalog 並快取；呼叫到已不存在的 method 以 tool error `unknown_method` 回報。

參數驗證仍在 GUI 端 `validate_params` 完成（dispatch 前、handler 前），因此 RPC channel 在 MCP 端雖為弱型別，實際仍有 typed validation；錯誤沿 [[0035]]／[[0047]] 的 stable `reason` 回報。

### 3. guard policy 隨 method 宣告，由 catalog 攜帶

`GUARD_DEPS`、`READ_REVEALS`、`OPERATION_KEY_OF` 從 MCP 端 `session_policy.py` 移到 measure-gui 的 `RemoteMethodEntry`，以宣告式欄位表示：guard deps 與 reveals 維持現有 pattern 字串，operation key 由 lambda 改為同語法的 template（例如 `"device:{name}"`）。`MeasureMcpSession` 從 catalog 讀取這些欄位並以現有邏輯執行。

欄位放在 app-specific 的 `RemoteMethodEntry`，不放共用層 `MethodSpec`，維持 [[0014]]「version guard 只在 main、共用層零知識」的邊界。GUI 本來就擁有 version table 並檢查 `expected_versions`，由它宣告 method 依賴哪些 resource 讓 policy 與 handler 同處一地；guard 仍由 MCP session 組裝 `expected_versions`、由 GUI 判定。

### 4. exposure enum

`McpExposure` 改為：

- `TOOL`：由 spec 產生一個 workflow tool（取代原 `GENERATED` 中被收錄者）；
- `RPC`：預設，只經 RPC channel 可達；
- `OVERRIDE`：綁定 hand-written workflow tool，`gui_rpc_call` 以 `reason="use_tool"` 拒絕並指名 tool；
- `INTERNAL`：agent 不可達。

`tool_name` 別名只允許 `TOOL` 使用；RPC channel 一律以 wire 名稱定址，不再產生 MCP 專屬別名。

### 5. 不採用 code execution

不提供在 GUI process 執行任意 Python 的 tool。所有 agent 動作都經 wire method、permit、version guard 與 operation handle；這是 measure-gui 與 Blender 類 app 在安全模型上的根本差異。

## 使用範例

以下回傳內容為示意。

### GUI 生命週期與 Overview

```text
gui_launch(port=8765, clean=true)          → {note, overview}
gui_overview()                             → {state: {has_project, has_context, has_active_context, has_soc}, tabs, running, ...}
gui_screenshot(target="window")            → {path}
gui_stop()                                 → {stopped: true}
```

### Project 操作

```text
gui_project_apply(chip_name="Q12_2D", qub_name="Q1", res_name="R1")
gui_soc_connect(kind="mock")
gui_context_list()                         → {active: "flux_0p5", labels: [...]}
gui_context_switch(label="flux_0p5")
```

### Tab 操作：run → analyze → post → writeback → save

```text
gui_tab_open(adapter_name="t1_rabi")                               → {tab_id: "t1", cfg, guide}
gui_tab_run(tab_id="t1", edits={"gain.sweep.stop": 0.8})           → {status: "finished", handle, figure}
gui_tab_analyze_review(tab_id="t1")                                → {summary, writeback: [...]}
gui_tab_post_analyze_review(tab_id="t1", updates={})               → {status: "pending", handle: 17}
gui_op_wait(handle=17, timeout=60)                                 → {status: "finished", summary, writeback: [...]}
gui_tab_writeback_apply(tab_id="t1", subtab_id="analysis")         → {applied, destination_context}
gui_tab_save_data(tab_id="t1")
gui_tab_save_image(tab_id="t1", subtab_id="analysis")
gui_tab_get_figure(tab_id="t1", subtab_id="post_analysis")         → {path}
```

`gui_tab_run_cancel()` 在任何時點停止目前的 run（同一時間只有一個 run）；已綁定 workflow tool 的 method 不可經 RPC channel 繞過：

```text
gui_rpc_call(method="tab.run_start", params={tab_id: "t1"})
→ tool error reason="use_tool": "tab.run_start is bound to workflow tool gui_tab_run"
```

### Inspect 操作

```text
gui_context_ml_list()                                          → {modules: [{name, kind}], waveforms: [...]}
gui_context_ml_inspect(item_kind="module", name="pi_q")        → {cfg}
gui_context_md_read(keys=["r_f", "q_f"])                       → {values: {r_f: 5923.1, q_f: 842.7}}
gui_rpc_call(method="value.read", params={key: "device.flux.value", type: "float"})
```

### Device 操作

```text
gui_device_list()                                                      → [{name, type_name, connected, ...}]
gui_device_connect(name="flux", type_name="YOKOGS200", address="GPIB0::1::INSTR")
gui_device_apply(name="flux", updates={"value": 0.5e-3})               → {status: "pending", handle: 21}
gui_op_wait(handle=21, timeout=30)
gui_device_disconnect(name="flux")
```

### Predictor 操作

```text
gui_predictor_info()                                           → {loaded: true, ...}
gui_predictor_predict(device_value=0.5e-3, from_level=0, to_level=1)   → {freq_mhz: 842.7}
```

安裝 predictor 屬低頻操作，經 RPC channel：

```text
gui_rpc_describe(method="predictor.load")                      → {params: {path: string, flux_bias?: number}}
gui_rpc_call(method="predictor.load", params={path: "params/Q1_predictor.json"})
```

### RPC channel 的長尾操作

```text
gui_rpc_list(domain="context")                                 → [{method: "context.ml_rename_module", summary}, ...]
gui_rpc_describe(method="context.ml_rename_module")            → {params: {old: string, new: string}}
gui_rpc_call(method="context.ml_rename_module", params={old: "pi_q", new: "pi_q_v2"})
```

參數型別錯誤由 GUI 端 `validate_params` 以 `invalid_params` 回報；宣告了 guard deps 的 method 在資源被 GUI 端改動後回報 `stale_version`，行為與 workflow tool 相同。

### 開發時的 e2e 循環

agent 在 lane 裡新增 GUI 端 wire method `tab.duplicate` 後，不重啟 MCP server 即可驗證：

```text
gui_stop()
gui_launch(worktree="/path/to/lane", clean=true)      # worktree 參數另案處理
gui_soc_connect(kind="mock")
gui_rpc_describe(method="tab.duplicate")               # 新 method 已在 catalog 中
gui_rpc_call(method="tab.duplicate", params={tab_id: "t1"})   → {tab_id: "t2"}
gui_screenshot()
```

改動落在 workflow tool 或 MCP 端程式碼時，仍需重啟 MCP server。

## Consequences

- agent 預設面對 39 個 tool（35 workflow + 3 RPC channel + 1 dev），依七類操作組織，primary flow 沒有替代路徑；長尾能力經 `gui_rpc_list` 按需發現。
- GUI 端 wire method 的新增與修改，經 `gui_stop` + `gui_launch` 即可由 agent 以 `gui_rpc_call` e2e 驗證；只有 workflow tool 與 MCP 端程式碼的改動仍需重啟 MCP server。搭配 `gui_launch` 指定 worktree（另案）即可覆蓋 lane 開發。
- 長尾操作多一次 `gui_rpc_describe` 往返；MCP 端失去長尾 method 的 JSON schema 型別提示，錯誤延後到 GUI 端 `INVALID_PARAMS`。
- wire 新增 `rpc.catalog`，`WIRE_VERSION` 遞增。
- run-measure-gui skill 與 MCP server instructions 需依新 tool 面改寫；移除的 tool 名稱不保留相容別名。
- fluxdep／dispersive／autofluxdep 的唯讀 MCP 各約 5～10 個 tool，不在本 ADR 範圍。

## Alternatives considered

- **維持 81 個 tool，只靠 Claude Code deferred tool loading。** token 成本已低，但選擇負擔、雙路徑與凍結問題都不變。
- **可重載 stdio proxy（`dev_reload` + `tools/list_changed`）。** 能讓全部 tool 熱更新，但依賴 client 在 session 中途重抓 tool 清單，且在 MEASUREMENT 實際路徑上加入開發用機制。
- **Blender 式 code execution。** 繞過 permit／guard／operation handle，對實體儀器不可接受。
- **toolsets 分組（GitHub MCP 式）。** 減少預設數量，但被關閉的 toolset 仍在 MCP 啟動時凍結，且 agent 無法在 session 中途切換。
- **guard policy 留在 MCP 端。** RPC channel 呼叫新 method 時 MCP 不知道其 guard deps，GUI 端新增的 mutating method 需重啟 MCP 才受 guard 保護，違背決策 2 的目的。

## 待決問題

原待決問題 1（以使用數據校正 workflow tool 清單）與 3（移除 tool 的過渡）隨 [[0060]] 取代決策 1 而失效：tool 清單改由逐一討論定案，舊 tool 一次切換、不保留相容別名。原待決問題 2（guard policy 移往 GUI 端）依決策 3 接受。
