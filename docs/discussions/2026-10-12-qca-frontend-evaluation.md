---
date: 2026-10-12
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-12-agent-interface-layers.md
  - docs/adr/0075-agent-interface-layers.md
basis:
  - main@9c956097f
  - NVIDIA/Quantum-Calibration-Agent-Blueprint@d8d37bc（2026-07-29，v0.1.0）
---

# QCA 作為前端的可行性評估

## 結論

NVIDIA Quantum-Calibration-Agent-Blueprint（下稱 QCA）不適合取代 measure-gui MCP，也不值得作為本 repo 的前端導入。QCA 強在 agent 應用與介面，儀器層很薄：沒有參數存放處，參數由 LLM 在實驗之間搬運，每次執行都開新的 subprocess，300 秒後強制終止。本 repo 的 MCP 已經處理硬體擁有權、寫入一致性、writeback 與統一保存，這些正是 QCA 缺少的部分。

接入有兩條路，各有結構性限制。Script 路徑保留 QCA 的歷史、面板與 VLM，但 QCA 成為第二個硬體 owner，無法與 GUI 共存。MCP 路徑沿用 GUI 的全部保證，但 QCA 的領域功能全部失效，只剩聊天介面與換模型能力。使用者指出 Claude 與 Codex 的桌面 app 已能接 MCP，聊天介面不再是理由。

值得借鏡的只有兩點：規劃時先與使用者約定每步的成功與失敗條件，以及用校準專用 VLM 判圖作為第二意見。

## 問題

評估 QCA 作為本 repo 前端的可行性，並在過程中釐清：只寫實驗 core 能否接上、存檔格式如何對齊、agent 如何得到結果與做判斷、參數如何在實驗之間傳遞，以及 QCA 相較 measure-gui MCP 的優缺點。

## QCA 的結構

以下事實取自 `d8d37bc`。

- **Backend**：`server.py` 以 FastAPI 加 `deepagents==0.5.3`（LangGraph）組成。對話狀態存在 `MemorySaver`，重啟即失。`interrupt_on = {}`，人工核准預設關閉；`LocalShellBackend` 開放 shell。
- **實驗介面**：`core/discovery.py` 以 AST 掃描 `scripts/*.py`，取第一個回傳註記為 `dict` 的公開函式。參數範圍寫成 `Annotated[float, (lo, hi)]`，只認字面預設值；`float | None` 會被當成 `str`。Docstring 只有第一行交給 agent，`Args:` 說明不會出現在 schema。
- **執行**：agent 呼叫 `run_experiment(experiment_name, params: dict, notes, python_path)`。`core/runner.py` 驗證參數（未知 key 拒絕、型別不轉型、`int` 不收 `100.0`、預設為 `None` 時放行 `null`），再以 `python -c` 起子程序，參數以 JSON 寫入 stdin 後立即關閉 stdin。stderr 即時寫入 log，stdout 只能是結果 JSON。逾時固定 300 秒，到時 `process.kill()`。
- **結果**：script 回傳 `data` 與 `plots`。Runner 把 `data` 與 `results` 合併後原樣回給 LLM，因此放在 `data` 的陣列會整個進入 context；頂層 `arrays` 只存檔不回傳。陣列只接受實數，複數會被略過且不報錯。
- **保存**：`core/storage.py` 寫 `data/experiments/YYYY/MM/DD/HHMMSS_<type>.h5`，root attrs 存 params、results 等 JSON，`/arrays` 平坦無單位，`/plots` 存圖的 bytes；SQLite `index.db` 一列一筆。同一秒同類型實驗會撞名。
- **判斷**：程式碼沒有任何判定邏輯。判準與重試策略寫在 `data/knowledge/skills/experiment-execution/SKILL.md`（SNR、單峰、範圍放寬 1.5 倍等），依據有兩種：回傳的 scalar，以及 `vlm_inspect` 把圖送給另一個 VLM（預設 `nvidia/ising-calibration-1-35b-a3b`）回傳文字。
- **參數傳遞**：沒有參數存放處。Workflow 的 `extracted` 與 `context` 存在 `data/workflows/<id>/workflow.json`，下一個 node 由 LLM 讀出後明確傳入。輸出對應到哪個參數、單位如何換算，都由 LLM 推斷。
- **知識**：全部是 Markdown。`data/knowledge/system-prompt.md` 是 Web 版唯一自動注入的內容；`documents/`、`skills/<name>/SKILL.md`（Agent Skills 格式）與 `memory/` 由 agent 用檔案工具按需讀取。CLI 走 `deepagents-cli`，另讀 `.qca/` 下的 skills 與 `AGENTS.md`，這點未驗證。
- **環境設定**：`.env` 放 API key 與 `QCA_MODEL`，`config.yaml` 放 FastAPI 與 VLM 設定。`QCAL_SCRIPTS_DIR`、`QCAL_DATA_DIR` 只有 CLI 子命令會讀，agent 的工具路徑寫死。

## 考慮過的選項與比較

### 整套採用

用 QCA 的實驗、保存、workflow 取代現有 MCP。這等於拿已完成的儀器層換一個較完整的介面層，放棄硬體單一 owner、寫入一致性、writeback provenance 與統一保存。不採用。

### Script 路徑（不改 QCA 程式碼）

在 QCA 的 `scripts/` 放只用 stdlib 的轉發 stub，stub 轉進本 repo 的 uv 環境，由本 repo 的無頭 host 執行 prepare、run、analyze、save，再把結果投影成 QCA 的 dict。討論中以五個角色拆解責任：

| 角色 | 本 repo | 此路徑中 |
| --- | --- | --- |
| 實驗 core | `zcu_lab/.../core.py` | 不變 |
| 實驗 adapter | GUI adapter、MCP recipe | 新增 QCA adapter，參數用 recipe 的方式宣告 |
| Host | GUI、`NotebookAdapter`、MCP | 新增無頭的 Notebook 式 host |
| 資料結果 | `RunRecord`、`AnalysisRecord`、`RunMetadata` | 經 `save_run` 寫入 |
| 儲存格式 | `AXES_SPEC`、`datafile`、`SaveLayout`、host 的存檔時機 | 不變，QCA 的 h5 只是當次快照 |

格式對齊的結論：`data.h5` 是唯一權威，QCA 的副本是衍生投影，帶 `run_id` 與 native path 指回權威檔案，之後重新分析不回寫。投影時把陣列放頂層 `arrays`、`data` 只放 scalar 與單位說明、複數拆成 I/Q。

硬限制：

- 單次 300 秒上限，長掃描跑不完。
- SIGKILL 只殺 stub，本 repo 的子程序會成為孤兒繼續操作硬體。Runner 寫完參數就關閉 stdin，不能用 stdin EOF 偵測；stub 必須自己持有通往子程序的 pipe，被殺時 pipe 關閉才觸發 `StopSignal`。
- 每次執行重連 Pyro 與儀器，且不能與 GUI 共用同一套設備。
- QCA 沒有 writeback。
- 需要新增無頭 `PlotHost`（現有只有 `QtPlotHost` 與 `NotebookPlotHost`）、每個實驗的 QCA adapter，以及從 adapter 宣告產生 stub 的 codegen。

### MCP 路徑

改 QCA 的 `server.py`，用 `langchain-mcp-adapters` 載入 measure-gui MCP。`MultiServerMCPClient` 預設每次呼叫開新 session（依 adapter 文件，未在本機驗證），會讓 seen guard 與 execution ID 失效，必須在 lifespan 維持常駐 session。

此路徑下仍可用的：瀏覽器聊天、換模型、Knowledge 面板。失效的：Experiments／Apparatus 面板、`lab` 歷史查詢、`vlm_inspect`、聊天中顯示圖（`server.py` 以 `str(msg.content)` 呈現工具結果）。另外要手動把 MCP server instructions 搬進 system prompt，並重寫整份 prompt。

### 不採用，只借鏡

保留 MCP 作為儀器邊界，從 QCA 借規劃與判圖的做法。採用此選項。

## 被否決的選項與理由

- **整套採用**：儀器層較弱，取代會失去現有保證。
- **Script 路徑**：300 秒上限與硬體雙 owner 是結構性限制；要補齊無頭 host、adapter、codegen、孤兒偵測與 writeback 入口，投入大於收益。
- **MCP 路徑**：只剩聊天介面與換模型，這兩項的需求已由桌面 app 涵蓋（見下節使用者判斷）。
- **另做寫回實驗補 writeback**：使用者否決。

## 討論中使用者的判斷與修正

- 不侵入性修改 QCA。
- 平坦參數可以沿用 recipe 的宣告方式。
- 關注重點是存檔格式，因此展開了五個角色與格式分層的討論。
- 不加寫回實驗；關心的是參數如何在實驗之間傳遞。
- Claude 與 Codex 都有桌面 app 並支援 MCP，「不用終端機的使用者需要 Web UI」不成立。
- 進行中的[控制身份](../adr/0070-control-identity.md)重構把 seen guard、`CfgRef` 改為單一 holder 的控制權模型，回應了「MCP 協定對模型負擔過重」的缺點。重新取得控制權後仍需先重讀狀態。

## 留下的問題

- 從 QCA 借鏡的兩點尚未落地：把「規劃時約定判準」寫進 `run-measure-gui` skill 或 recipe／workflow 的接受度設定；評估校準 VLM 作為判圖第二意見的小工具。
- 桌面 app 使用 MCP 的前提尚未逐一確認：呼叫逾時需超過 300 秒、server instructions 是否被載入、repo 的 skill 與角色規則是否讀得到、GUI 須在同一台機器。

## 依據

- QCA：`core/discovery.py`、`core/runner.py`、`core/storage.py`、`tools/lab_tool.py`、`tools/vlm_tool.py`、`tools/workflow_tool.py`、`server.py`、`cli.py`、`prompt.py`、`data/knowledge/`，版本 `d8d37bc`。
- 本 repo：[ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0068](../adr/0068-remote-transport.md)、[ADR-0069](../adr/0069-experiment-core-and-frontends.md)、[datafile README](../../lib/zcu_tools/datafile/README.md)、[notebook README](../../lib/zcu_tools/notebook/README.md)、[measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)、`zcu_lab/v2/twotone/time_domain/t1/`。
