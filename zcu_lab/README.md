# zcu_lab

**Last updated:** 2026-11-01，作者入口與驗收命令分層

**讀者：** lab-author、agent。查找實驗室定義的套件邊界、註冊入口與操作文件。

`zcu_lab` 只放實驗定義、實驗室宣告與各 workflow 自己的小工具。`zcu_tools` 提供實驗與前端框架，不反向 import 這個套件。GUI 組合根在 `apps/`，`scripts/app/` 只委派它們；agent MCP 組合根在 `mcp_servers/`。三個 GUI 與 measure MCP 取得同一種 lab 描述後注入框架；兩個 analysis MCP 只傳入 GUI module 名稱。

使用者擴充入口見 [tutorial](tutorial/README.md)，程式碼規則與檢查方式見 [POLICY](POLICY.md)。

預設的 session 還原腳本屬於框架，位於 `lib/zcu_tools/gui/app/measure/default_startup.py`，行為見 [ADR-0064](../docs/adr/0064-process-startup.md#保存與關閉)。這裡只放使用者自己的 startup 腳本（`simulation_envs/`、`kickstart/`）；自訂腳本要先還原上次 session 時，import 並呼叫 `zcu_tools.gui.app.measure.default_startup.startup`。`--clean` 略過全部腳本，`--script` 完全取代預設腳本，兩者互斥。建立 entry 與 alias 的步驟見 [startup 教學](tutorial/02-startup-and-entry.md)。[Six-fluxonium kickstart](kickstart/six_fluxonium/README.md) 建立六顆 fluxonium 的 entry，不連硬體、不載模擬。

`v2/` 放各實驗的 core 與前端附件，`recipes/` 放量測流程定義。`workflows/` 的 `coherence_fluxdep/`、`mist_fluxdep/`、`overnight/` 各自擁有流程、表單與科學工具，`common/` 只放真正共用的實驗步驟。`workflows/registry.py` 提供有序註冊清單。Window、App、RPC、MCP、表單協調、layout 與 input 保存都由 lib 宿主負責。實驗室宣告位於 `components/`、`definitions.py` 與 `templates.py`。[`simulation_envs/`](simulation_envs/FTF/README.md) 放 GUI／Notebook 共用的模擬參數檔與 startup 範例，不提供離線建庫入口。[Six-fluxonium](simulation_envs/six_fluxonium/README.md) 提供六顆無耦合 fluxonium、六條局部 DC 線、共用 coil 與 mux 讀出。每顆各自設定局部線與 coil 的 current offset／period，兩路 reduced flux 相加，不含串擾或 fast-flux pulse。

GUI 啟動時 alias 表是空的，沒有 preset 檔。Alias 只存在各 point 的 `aliases.yaml`：開 entry 清空 alias，選 point 載入該 point 的表；Inspect 編輯目前表的草稿，Apply 才保存至目前 point。要在啟動時有初值，由 startup 腳本先建立 point，再用 `ctx.context.apply_alias_rows` 初始化並保存其表（simulation_envs 與 kickstart 的腳本都這樣做）。Alias 表只選元件焦點。表單從元件的 flux_dev／rf_source 取得裝置初值，框架不從 cache 推測裝置。

## Lab 組裝與註冊

`assembly.lab_definitions()` 是 GUI 與 measure MCP 共用的唯一組裝入口。
它回傳 frozen `LabDefinitions`，只保留宣告，不註冊元件、不開 entry、不啟動 runtime，也不操作硬體。
新增 lab 定義種類時，在此描述交付欄位；根不逐項重組定義或 module 邊界。

| 欄位 | Lab 來源與用途 |
| --- | --- |
| `register_components` | `components.register_all`，向接收的 ComponentRegistry 註冊 model 與角色 |
| `register_experiments` | `definitions.register_all`，向接收的 measure Registry 註冊 adapters，不重複註冊 components |
| `templates` | `templates.TEMPLATES` 的唯讀 mapping，提供新建 module／waveform 的持久範本 |
| `workflow_attachments` | `workflows.registry.build_registry()` 的有序 attachment tuple |
| `recipes` | `recipes.RECIPES`，供 measure MCP 列出與執行 recipe |
| `catalog_reload_policy` | Measure 擁有的 `CatalogReloadPolicy`，描述 source packages、reload／preserved 模組與 catalog module |
| `logging_namespaces` | 額外 logger namespace tuple，交給 GUI launcher |
| `fluxdep_export_planner` | `fluxdep_export.plan_fluxdep_export`，產生 fitted alignment points 的元件寫回提案 |
| `component_flux_context_provider` | `component_flux.component_flux_context`，投影元件 flux frame 與預設裝置 |

Source package 路徑由 `zcu_lab.__file__` 與 `zcu_tools.__file__` 推得。
Templates 發布後不修改其內容；frozen dataclass 不會遞迴凍結範本。

`flux_reset` 範本建立 `reset/pulse`，可存到 qubit 的 reset library 分組，再由
`$qubit.module.reset` 指向該 module。它在 `$qubit.wiring.lo_flux_ch` 播放 frequency
0 MHz 的 const 方波，gain 讀 `$qubit.ext.flux_reset_gain`，長度讀
`$qubit.ext.flux_reset_length`（us）。缺少來源時，對應欄位保持 Invalid。
`pulse_cfg.post_delay` 是回程後等待 flux 尾巴穩定的時間（us），沿用 `PulseSpec`
的 0.0 預設值，0 代表不等待。實機應依 flux 尾巴量測（例如
`fastflux/distortion`）設成足夠長，不能把預設 0 當成已穩定。

各入口的最小 Protocol 與依賴理由見 [ADR-0072](../docs/adr/0072-module-boundaries.md#lab-定義與組合根)。

Measure behavior 先註冊 components，再以 `lab.templates` 建 Registry，呼叫 `lab.register_experiments` 填入 adapters。
Library 新建直接讀 registry 的範本，不另建 catalog。
Reload 保留既有範本與元件註冊，不重新 bootstrap components。
Import 不執行註冊，重複 bootstrap 按接收 registry 的規則報錯。

`components/` 擁有具體元件 model、wiring、module 槽與內建角色。`base.py` 放共用欄位與基礎元件；`qubit.py` 放共用 qubit 欄位與 wiring。Fluxonium、transmon、resonator、JPA 與 mux 各有獨立子模組。`__init__.py` 提供公開匯出與明確註冊入口。Model 自行宣告 ext 與 extra 策略，普通字串引用在 resolve 時解析。GUI behavior 與 measure MCP server 消費 lab 的 registrar，明確 bootstrap；兩個 analysis MCP 不註冊 components。Notebook caller 可呼叫 `components.register_all(component_registry)`。框架只提供容器、kind 查表、角色解析與來源記錄。

`MuxGroupSchema` 以 `group/mux` 註冊於同一 ComponentRegistry。必填 resonators 保存 1 至 8 個有序名稱。選填 qubits 必須等長且不重複。`flux_dev` 保存 global coil 的 device 名稱。每顆 qubit 的 `flux: {offset, period: {<device>: ...}}` 保存零輸出時的 flux 與 local、coil 等儀器的 signed periods，group 不另存校準。Component YAML 使用 `schema_version: 3`；舊文件經[長期升級入口](../docs/parameter-upgrades.md)轉換，runtime 不自動升級。提供 qubits 時，lab Form 在 Open／Reset 以固定 ParameterContext 檢查兩邊 component 存在、kind 正確，並逐項要求 qubits[i].resonator 等於 resonators[i]。錯誤指名欄位、index 與實際值。Resonators 不去重；沒有 qubits 的 onetone 保留手填 readout 與缺來源的既有政策。

Scalar alias `mux=MG12` 讓 mux adapter 從固定 capture 的 `$mux.module.readout` 取得 Readout 初值。完整 tones[i] 綁定 resonators[i]。participants 以 list[int] 選擇量測的原始序號並指定順序，不改綁定或保存的 resonators。Module owner 是 group，不是第一個 resonator；重排不會改 owner。Framework 不知道 group schema。定位與 writeback capture 邊界見 [ADR-0072](../docs/adr/0072-module-boundaries.md#mux-group-是實驗室的-component-定義)。

<a id="component-欄位單位"></a>

## Component 與實驗作者入口

元件型別別名在既有 Field 附靜態 `json_schema_extra` 單位資料。Frequency 用 MHz，Duration 用 us，Energy 用 GHz，PumpPower 用 dBm。Qubit 的 `flux.offset` 使用 Φ0，`flux.period.<device>` 使用各儀器原生單位／Φ0，方向保留。Global flux 原值仍可存於 `general.ext.flux_value`。Qubit 不再保存 `flux_unit`，JPA 的 `J1.flux_unit` 維持原契約。Flux 不標固定單位。未宣告單位不代表無因次。Resources 以公開路徑查詢提供固定欄位的靜態 unit。GUI 的 WB 列只用查詢結果加標籤，不換算。Ext 與 flux 類欄位不顯示單位；run context 不消費這些標記。

Recipe 的宣告、Inputs／Report、tab 操作、來源與 overrides、Analyze／取消及 Save 契約見
[recipes README](recipes/README.md)。Mux 的逐元素操作見 [mux recipe guide](recipes/mux-guide.md)。

Measure 的 `form.py` 提供 `make_form(ctx)`，在 Open／Reset 依固定來源選初值與形狀。
`gui.py` 宣告 Adapter／Analysis／Interactive／PostAnalysis 值，不繼承 adapter class。
Form 輸出 adapter 的輸入型別，adapter.run 轉成實際實驗 cfg 並回傳 RunRecord。
Module 與掃描欄位平鋪於頂層，只有 pulse Advanced 保留既有分組與摺疊。
Load 不轉換舊 2.0 cfg，也不反填目前 Form；資料與 metadata 保留，record.cfg 為 None 並附原因。
`onetone/flux_dep` 與 `twotone/fluxdep` 的 core 用具名 `flux_dev: str`；
JPA 使用具名 `jpa_flux_dev` 與 `jpa_rf_dev`。Form 在 Open／Reset 讀取
`$qubit.flux_dev`、`$mux.flux_dev` 或 JPA 的 `$jpa.flux_dev`／`$jpa.rf_source`，
保存可覆寫的 direct 裝置名稱。既有 input 不隨 alias 或元件參數更新。
缺來源時留空，必填欄位保持 Invalid，可選欄位保留 None，不使用靜態名稱 fallback。Core 透過 RunContext.devices
取得 driver，不認得 alias。Analyze／WB／Save 使用 Run 的固定來源，不重讀目前表。
具名欄位與範本契約見 [ADR-0073](../docs/adr/0073-measure-form-derivation.md)。

Lab async 的 await 與序列執行限制見 [POLICY](POLICY.md#2-在-recipe-寫出操作流程)。

[`v2`](v2/README.md) 是實驗的使用者 namespace。每個實驗的 core 與可選 GUI、notebook 附件放在同一資料夾。Core 不依賴 GUI。

有擬合的分析在 summary 附 `fit_quality`，建議使用 `zcu_tools.analysis.fitting.compute_fit_quality`。實驗定義者負責選擇各 fit 的資料、參數及 covariance；GE 的 joint、ground、excited 可分開回報。這是實驗定義者的約定，框架不檢查、不特別解讀品質，也不修改 optimizer 或 writeback 政策。

個別實驗的測試邊界見 [程式碼品質](../docs/code-quality.md#測試驗證可觀察契約)。離線驗收依下節入口執行。

Predictor 使用 entry 的 `predictors/*.json` 或手動參數，不從實驗室 component 轉換輸入。

## 實驗繪圖

實驗繪圖使用 `plots.subplots(num=...)` 建圖、`fig.show()` 顯示，
顯示後透過 `@plots.on_plot_thread` 更新。可選的 `live_line(ax)`、`live_heatmap(ax)`
與 `live_scatter(ax)` 只管理本地 artists，`set(...)` 回傳 changed artists。
詳細寫法見 [v2 authoring](v2/experiment-authoring.md#圖的建立顯示與更新)。

## 實驗定義的離線驗收

實驗定義改動，包含 live plot 與分析圖的畫面改動，依 [實驗定義驗收規則](../tests/README.md#實驗定義) 執行離線 probe。
框架 UI 的 ComponentDriver 驗收仍走該頁的框架 UI 分支。

### 準備 Python 檔

在 task scratch 寫 `probe.py`，按 [驗收規則](../tests/README.md#實驗定義)
選階段與輸入。Cfg、Result、options 與 RunMetadata 使用其 owner 的型別。
Stage API、生成器、PNG 格式與環境清理責任見
[experiment probe](../lib/zcu_tools/experiment/README.md#offline-experiment-probe)。

下列片段各自獨立，假設 caller 已填好具名物件。按改動選用，不串成固定流程：

Run 的 caller-owned YAML、`white_noise` 參數與環境清理見
[simulation environment](../lib/zcu_tools/experiment/README.md#explicit-simulation-environments)。

```python
from zcu_tools.experiment.probe import analysis_stage, run_stage, save_stage

# Run input includes an explicitly wired environment owned by this script.
acquired = run_stage(experiment, cfg, environment)

# Save can use a directly constructed RunRecord without acquisition.
saved = save_stage(record, experiment.AXES_SPEC, output_dir)

# Analysis can use a directly constructed Result without run or save.
analyzed = analysis_stage(experiment, record, options)
```

Run／analysis 的圖片由腳本保存，例如：

```python
output_dir.mkdir(parents=True, exist_ok=True)
for name, png in analyzed.pngs.items():
    (output_dir / name).write_bytes(png)
```

### 檢查與執行

`repo` 指向目前 worktree。`main_repo` 是 Git common dir 的上層主 checkout，
scratch 放在這裡，清除 lane 後仍保留。
將 `<task-id>` 換成既有任務，先完成 `$scratch/probe.py`，再執行下列命令。
受管理 lane 的 pytest 與 affected checks 仍須經當次任務指定的 runner，遵守其授權與時間上限。

```bash
repo="$(git rev-parse --show-toplevel)"
git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
main_repo="$(dirname "$git_common_dir")"
scratch="$main_repo/.agent_state/plans/<task-id>/scratch/experiment-probe"
mkdir -p "$scratch"
export PYTHONPATH="$repo:$repo/lib${PYTHONPATH:+:$PYTHONPATH}"
uv run --directory "$repo" --no-sync -- pyright -p "$repo/pyproject.toml" "$scratch/probe.py" \
  > "$scratch/pyright.log" 2>&1 &&
uv run --directory "$repo" --no-sync -- ruff check "$scratch/probe.py" \
  > "$scratch/ruff.log" 2>&1 &&
env -u QT_QPA_PLATFORMTHEME QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  uv run --directory "$repo" --no-sync -- python "$scratch/probe.py" \
  > "$scratch/probe.log" 2>&1
status=$?
printf '%s\n' "$status" > "$scratch/exit-status.txt"
```

腳本、命令、回傳結果與逐張看圖的完成條件依 [tests 驗收規則](../tests/README.md#實驗定義) 記錄。
離線驗收不代表實機校準或物理模型已獲驗證。
