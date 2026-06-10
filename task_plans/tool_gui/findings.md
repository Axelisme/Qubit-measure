# Findings：fluxdep_fit GUI 移植可行性評估

**基準 commit：** 1e4cb1d8 | **評估日期：** 2026-06-03

本檔是評估的實質核心：把 `notebook_md/analysis/fluxdep_fit.md` 這條分析流程移植成獨立 GUI（理想為獨立 skill + MCP server）的逐項調查結果。所有結論附 `file:line` 證據。

---

## 1. fluxdep_fit 分析流程的本質（與 measure 領域的根本差異）

`fluxdep_fit.md` 是一條 **「載入靜態資料 → 人在迴圈中標註 → 數值搜參 → fit → 存參數」** 的離線分析流程，**完全不碰硬體**：

| 維度 | measure-gui（single_qubit.md） | fluxdep_fit |
| --- | --- | --- |
| 資料來源 | 即時 run on FPGA（soc/soccfg） | 載入已存的 `.hdf5` spectrum 檔 |
| 核心動作 | 設 cfg → run → analyze → writeback | 定線 → 選點 → 搜庫 → scipy fit |
| 人在迴圈 | 看 analyze 圖判斷好壞 | **互動拖線/手繪選點**（演算法無法全自動） |
| 硬體概念 | soc、device、reps/rounds、sweep | **全無** |
| 生命週期 | RunService（QThread worker + 進度條 + cancel） | 主要是同步運算（搜庫/fit 是 CPU-bound，需進度回呼） |
| 輸出 | md/ml writeback + hdf5 data | `spectrums.hdf5` + `params.json`（EJ/EC/EL） |

**關鍵判斷：** fluxdep 分析的領域物件是「**多張 spectrum 疊合 + 標註點雲 + fluxonium 參數**」，跟 measure 的「**tab = 一次硬體實驗 session**」是不同的領域模型。硬把它套進現有 `adapter / tab / run / writeback` 框架會是**錯誤抽象**（詳見 §5 架構抉擇）。

---

## 2. 可直接重用的 headless 純運算層（🟢 低風險，~60% 邏輯）

這些模組**零視覺化依賴**，純 numpy/scipy/numba/scqubits，新 GUI 可直接 import：

### `notebook/analysis/fluxdep/processing.py`（互動工具的演算法核心）
- `cast2real_and_norm()` (14-28)：複數信號 → 實數歸一化（振幅或相位）
- `spectrum2d_findpoint()` (31-58)：自動峰值檢測（scipy `find_peaks`，逐 column）
- `downsample_points()` (61-88)：點雲下採樣（移除近鄰）
- `diff_mirror()` (91-129)：鏡像損失（auto-align 定線用）

### `notebook/analysis/fluxdep/models.py`（躍遷數學，純 numpy）
- `energy2linearform()` (32-116)、`energy2transition()` (176-213)、`compile_transitions()` (108-174)、`count_max_evals()` (21-29)

### `notebook/analysis/fluxdep/fitting.py`（搜參 + fit 核心）
- `search_in_database()` (26-238)：njit 平行 kernel 在 hdf5 資料庫搜 (EJ,EC,EL)
- `fit_spectrum()` (242-300)：scipy `least_squares`
- ⚠️ **但兩者 `plot=True` 時回傳 matplotlib `Figure`**（fitting.py:9,12 import matplotlib）——需把「運算」與「畫圖」分離；好消息是 `search_in_database` 有 `@overload plot: Literal[False]`（fitting.py:42-55），**已支援 headless 純回傳數值**。
- 依賴 numba（fitting.py:13,85）、scqubits（fitting.py:277）——重，但已是專案既有依賴。

### `simulate/fluxonium/energies.py` + `simulate/__init__.py`
- `calculate_energy_vs_flux()` (energies.py:27-63)：scqubits 物理計算
- `value2flux()` / `flux2value()` (simulate/__init__.py:9-40)：座標轉換

### `notebook/persistance.py`（資料型別 + I/O，完全可重用）
- TypedDict：`SpectrumResult`/`SpectrumData`/`PointsData`/`TransitionDict`/`FluxDepFitResult`（30-112）
- I/O：`dump_spectrums`/`load_spectrums`（115-167，HDF5）、`dump_result`/`load_result`/`update_result`（54-90，JSON）

---

## 3. 必須重寫的互動工具（🔴 高風險，~30% 工作量）

四個互動工具全綁 **ipywidgets + matplotlib widget + `FuncAnimation`**，無法直接搬：

| 工具 | 檔案:line | 互動本質 | 內部狀態 | 取值 API |
| --- | --- | --- | --- | --- |
| `InteractiveLines` | find_line.py:17-545 | 拖兩條垂直線（half/int flux）+ auto-align | `flux_half/flux_int/picked/active_line` | `get_positions()→(half,int)` |
| `InteractiveFindPoints` | find_point.py:12-260 | 手繪 mask 選/擦點（TwoTone）+ threshold slider | `mask`（bool 陣列） | `get_positions()→(devs,freqs)` |
| `InteractiveOneTone` | onetone.py:13-100+ | threshold slider 自動峰值篩選（OneTone） | `max_freq_idx` | `get_positions()→(devs,freqs)` |
| `InteractiveSelector` | point_select.py:20-265 | 多 spectrum 疊合點雲，圓形 brush 選/擦 | `selected/filter_mask` | `get_positions()→(fluxs,freqs,selected)` |

### 🟢 重大利好：互動「演算法核心」可重用，且 Qt canvas 支援 mpl event
- 四個工具的**數值邏輯**都委派給 §2 的 `processing.py`（純運算）——互動層只是「滑鼠事件 → 更新狀態 → 重畫」的薄殼。
- measure-gui 的 plot canvas 是 **`FigureCanvasQTAgg`**（mpl_backend.py:23,66；plot_host.py:120），它**原生支援 matplotlib 的 `mpl_connect`/`button_press_event`/`motion_notify_event`**。
- **意涵：** 四個互動工具的 `mpl_connect` 滑鼠互動邏輯（拖線、點選、brush）可幾乎原樣移到 Qt canvas；只有 ipywidgets 的控制項（slider/button/dropdown/checkbox）要換成 Qt widget（`QSlider`/`QPushButton`/`QComboBox`）。

### ⚠️ 但現有 GUI plot 層是「純展示」，無互動骨架
- `plot_host.py` 全是 `attach_canvas`/`refresh_figure`/`close_figure`（52-266），**零 `mpl_connect`/`pick_event`**——measure-gui 從不需要圖上互動。
- liveplot 也零互動（grep `mpl_connect|button_press|Slider` 於 liveplot/*.py 無結果）。
- **意涵：** 「圖上互動」是這個 GUI **要新建的能力**，不是現成可複用的元件。這是工作量主體。

---

## 4. 視覺化層（🟡 中風險，~10%）

- `notebook/analysis/fluxdep/utils.py:69-213` 的 `FreqFluxDependVisualizer` 是 **Plotly** fluent API（`go.Figure/Heatmap/Scatter`），鏈式 `.plot_background().plot_simulation_lines().plot_points()....get_figure()`。
- 最終 fit 結果檢視圖（fluxdep_fit.md:254-271）用它產 plotly + 存 html。
- **移植選項：** (a) 保留 Plotly 邏輯，Qt 端用 `QWebEngineView` 嵌入 html；或 (b) 把 fluent API 後端改寫成 matplotlib（與 §3 的 Qt canvas 一致，較內聚）。fluent API 的組裝邏輯可保留，只換繪圖後端。

---

## 5. 現有 measure-gui「Qt+RPC+MCP」框架的可複用性（架構抉擇核心）

探查 `lib/zcu_tools/gui/` 後，分層的領域耦合度如下：

### 🟢 完全通用、可直接複用的傳輸/機制層（與「實驗」無關）
| 層 | 檔案 | 通用度 | 說明 |
| --- | --- | --- | --- |
| 進程入口 | `script/run_gui.py`(99-143) + `gui/app.py`(`run_app`) | 100% | composition root：組裝 generic 層 + 領域 registry，新 GUI 只換 registry 內容 |
| Wire framing | `services/remote/framing.py` | 100% | NDJSON over TCP，零實驗概念 |
| Wire 信封 | `services/remote/wire.py`(19-99) | 100% | Request/Response dataclass + 版本握手 |
| Param 規格 | `services/remote/param_spec.py`(30-159) | 100% | `ParamSpec`/`JsonType`/`validate_params`/`build_input_schema`（生成 MCP inputSchema） |
| MCP tool 生成 | `mcp_server.py` `_generate_tools`/`_make_forwarder` | 95% | 從 method_specs 自動生成 MCP tool |
| MCP 進程模型 | `mcp_server.py`(795-818, 2179) | 100% | MCP server fork `run_gui.py` subprocess，走 TCP socket bridge |
| 版本表 + 樂觀鎖 | `state.py` `VersionTable` | 100% | per-resource 單調版本 + guard，資源鍵命名通用 |
| EventBus 機制 | `event_bus.py` | 機制 100% | emit/subscribe 通用；event enum 成員是領域資料 |

### 🔴 綁死 measure 領域、新 GUI 不該複用（要新寫對應領域版本）
| 概念 | 檔案 | 為何不可複用 |
| --- | --- | --- |
| `tab = experiment session` | `state.py` `Session`(92-140) | fluxdep 的領域物件是 spectrum 集合+點雲，不是「一次實驗」 |
| Run 生命週期 | `services/run.py` + QThread worker | fluxdep 無「run on hardware」；搜參/fit 是同步 CPU 任務 |
| Adapter（cfg→run→analyze→writeback） | `gui/adapter/` + `experiment/v2_gui/adapters/` | fluxdep 無 cfg-schema / reps / sweep / device |
| Writeback（md/ml） | `services/writeback.py` | fluxdep 輸出是 params.json，非 MetaDict/ModuleLibrary |
| Context（md/ml/soc） | `services/context.py` | fluxdep 不需 md/ml/soc，只需 result_dir 定位檔案 |
| Device / Connection | `services/device.py`/`connection.py` | fluxdep 不碰硬體 |

### 結論：兩種架構路線
- **路線 A（重用框架、改名 tab→session）**：第一份子 agent 報告傾向此路。**評估後不建議**——會被迫保留大量不適用的 cfg/run/device/writeback 概念，或做大幅改名重寫（~2000-2500 行改動），且語義錯位（最小驚訝原則違反）。
- **路線 B（只抽傳輸/機制層，領域層全新寫）**：**建議**。複用 framing/wire/param_spec/mcp 生成/版本表/EventBus 機制（這些已是 generic-free），fluxdep 領域層（State/Service/互動 UI/MCP method）全新設計，對應「spectrum 集合 + 標註 + fit」的真實領域模型。

⚠️ **但路線 B 有前提問題：** 現有 remote/mcp 機制層雖 generic-free，**目前並未被抽成一個獨立可 import 的套件**——它和 measure 的 dispatch/method_specs 物理混在 `services/remote/` 下。要走路線 B，需先判斷「是把這層抽出共用，還是複製一份」。這是**待用戶定奪的架構決策**（見 task_plan 決策 D2）。

---

## 5.5 可復用元件清單（為「日後抽共用層」盤點，按分層）

現有分層 = `app → driven adapter(View) → controller → services → state`（ADR-0013：`MainWindow` 與 `RemoteControlAdapter` 是兩個平級 View，都接 `Controller` façade）。新 app 可照搬此**形狀**（接口不強求一致）。下表標出每個檔案在「日後抽共用」時的歸屬：

**🟢 可抽成共用工具（你願景中 `gui/` 頂層「剛好能共用」的）—— 純機制、零實驗概念：**
| 檔案 | 角色 | 共用理由 |
| --- | --- | --- |
| `mpl_backend.py` / `mpl_backend_setup.py` | mpl 轉發後端 + GuiFigureCanvas | 你明確點名要共用；跨線程 draw 吸收、canvas 註冊 |
| `plot_host.py` / `plot_routing.py` / `figure_export.py` | figure 容器掛載/路由/匯出 | 純 Qt+mpl 容器管理，無實驗概念 |
| `pbar_host.py` | progress bar 後端 | 你明確點名要共用 |
| `event_bus.py` | emit/subscribe 機制 | 機制通用（event enum 成員是各 app 自填的領域資料） |
| `io_manager.py` | 檔案 IO 管理 | 通用 |
| `expression.py` | AST 安全求值（scalar 表達式） | 通用工具；fluxdep 的 EJ/EC/EL bound 輸入可能用得上 |
| `services/remote/` 全套 | NDJSON-TCP RPC + MCP 生成 | framing/wire/param_spec/mcp_server 生成/errors —— generic-free（領域只在 method_specs/dispatch/events 的內容） |
| `services/operation_gate.py` / `guard.py` / `progress.py` / `shutdown.py` | 互斥/樂觀鎖/進度/關閉協調 | 純機制 |
| `state.py` 的 `VersionTable` | per-resource 版本表 | 100% 通用（資源鍵命名通用） |

**🟡 形狀可仿、內容要重寫（照搬 pattern，填 fluxdep 領域）：**
| 檔案 | 為何不能直接共用 |
| --- | --- |
| `controller.py` | façade 方法是 measure 領域動作（run/analyze/writeback）→ fluxdep 換成 load/pick-line/select-points/search/fit |
| `state.py` 的 `State`/`Session` | Session=experiment session；fluxdep 換成 spectrum 集合+點雲+params |
| `services/remote/method_specs.py` / `dispatch.py` / `events.py` | RPC 方法清單/handler/event 序列化是領域內容（機制在 framing/wire/param_spec） |
| `app.py` `run_app` | composition root pattern 可仿，wiring 的 service 換 fluxdep 版 |
| `registry.py` / `role_catalog.py` | adapter/role 註冊機制——fluxdep 無 adapter，可能整個不需要 |

**🔴 measure 專屬、fluxdep 不需要（複製後刪）：**
`services/`：`run.py`/`analyze.py`/`save.py`/`writeback.py`/`tab.py`/`cfg_editor.py`/`context.py`/`device.py`/`connection.py`/`startup.py`/`workspace.py`/`session_codec.py`/`caretaker.py`/`persistence_types.py`、`cfg_schemas.py`/`live_model.py`/`sweep_model.py`/`runner.py`/`role_catalog.py`、`adapter/` 與 `adapters/` 整個、`specs/`、`ui/` 大部分。
→ fluxdep 換成自己的：`load_service`(讀 hdf5)/`spectrum_store`(點雲集合)/`fit_service`(搜庫+scipy,worker thread)/`export_service`(spectrums.hdf5+params.json) + 自己的互動 UI。

**抽共用的順序建議（對齊你的願景）：** 當前先**整包複製**（含 🔴 死碼），先讓 fluxdep 跑起來；等兩個 app 都穩定，再把 🟢 那批「被兩邊都用到且形狀一致」的抽到 `gui/` 頂層 + `app/{main,fluxdep}/` 子模塊化。**不要現在就猜哪些共用**——複製後在真實使用中浮現的共用點才可靠（避免過早抽象）。

## 6. skill + MCP server 封裝範本（現成可仿）

`run-measure-gui` skill 是完整範本，新 skill 可同構：
- 三檔結構：`SKILL.md`（驅動指引）+ `smoke.py`（raw socket 端到端煙測）+ `sync_skills.sh`（同步 .claude→.agent/.codex 三副本，**非 hard link**，cp 整檔）。
- skill frontmatter 有 `skill_version`，改後 bump + 跑 sync。
- MCP server 在 `.mcp.json` 註冊；`gui_launch` fork GUI subprocess + 版本握手 banner。
- **新 GUI 要做對等封裝：** 新 `run_fluxdep_gui.py` 入口 + 新 MCP server（或同一 MCP server 多一組 tool）+ 新 skill 三副本。
