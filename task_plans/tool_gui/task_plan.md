# 任務計畫：fluxdep_fit 分析流程 GUI 移植可行性評估

**最後更新：** 2026-06-03（D1–D5 決策定錨：複製一份獨立 GUI、同構獨立 MCP server、共用層延後抽取，基準 1e4cb1d8）

## 目標

評估把 `notebook_md/analysis/fluxdep_fit.md`（fluxonium flux-dependence 擬合分析流程）移植成一個**獨立 GUI 應用**的可行性與架構規劃。理想終態：獨立的 **skill + MCP server**，可由 agent 驅動（如同現有 `run-measure-gui`）。用戶可接受重新渲染、獨立寫互動邏輯，不依賴 fluxdep_fit.md 中為 Jupyter 設計的工具（ipywidgets/plotly widget）。

**本階段交付物 = 評估報告本身**（findings.md + 本檔），不含實作。

## 設計準則（沿用專案 CLAUDE.md）

- Fast Fail、責任明確、最小驚訝、強型別、無 Legacy。
- 不臆測架構；不適合擴展處告知用戶決定（見「待用戶定奪」）。

## 可行性總評

**可行，建議採「路線 B：複用傳輸/機制層 + 領域層全新寫」。** 整體可行性 **7.5/10**。

- 純運算層（~60%）可直接 import 重用（processing/models/fitting/persistance/simulate），且 `search_in_database` 已支援 `plot=False` headless 回傳。
- 互動工具（~30%）必須重寫，但**演算法核心已在 processing.py（純運算）**，且 Qt canvas 是 `FigureCanvasQTAgg`（支援 mpl event），故拖線/選點/brush 的滑鼠邏輯可移植，只換 ipywidgets→Qt 控制項。
- 視覺化（~10%）Plotly fluent API 可保留組裝邏輯、換後端或 QWebEngine 嵌入。
- skill+MCP 封裝有 `run-measure-gui` 現成範本可同構。

**最大不確定性 = 架構抉擇（路線 A vs B）+ remote 層是否抽共用（D1/D2），需用戶定奪後才進入實作規劃。**

## 階段規劃

### Phase 0–3：探查 / 架構定錨 / 領域設計 ✅（細節見 findings.md / phase2_domain_model.md / phase3_v1_design.md）

- **Phase 0（可行性）**：摸清 fluxdep_fit 資料流 + 四互動工具後端依賴 + 純運算可重用層 + measure-gui 分層耦合度；確認 Qt canvas 支援 mpl event（互動可移植）。產出可行性總評 + 兩條架構路線。
- **Phase 1 架構決策（D1–D5，用戶定錨，仍生效）**：
  - **D1 領域全新寫**：不套 measure 的 adapter/tab/run/writeback；核心 = spectrum 集合 + 標註點雲 + fluxonium 參數。
  - **D2 減法搬運**：新 app 從零組起，只 copy（非 import）gui/ 的通用骨架，無 measure 死碼；兩份獨立、先不抽共用、需求穩定後再收斂。
  - **D3 互動後端**：Qt `FigureCanvasQTAgg` 重接 mpl event；演算法核心複用 `processing.py`。
  - **D4 統一 matplotlib**（放棄 QWebEngine——repo 無裝且要 Chromium 重依賴）：Plotly fluent 組裝邏輯保留、底層 `go.Figure`→mpl。
  - **D5 同構獨立 server**：獨立 run_fluxdep_gui.py + MCP server + skill，照 measure fork-subprocess-over-TCP。
- **Phase 2 領域模型**：fluxdep = 線性有序 pipeline（非平行多 tab）→ `FluxDepSession`；State 形狀（spectrums + alignment + selection + 複用 VersionTable）；Service 切分；視覺化 Plotly→mpl 對照；標出 Q1–Q5 開放問題。
- **Phase 3 v1 詳細設計**：通用骨架搬運清單（🟢直接搬/🟡去 measure 依賴/🔴不搬）；v1 State 定稿（**v1 不做 session 還原**，不搬 caretaker）；Service 介面（LoadService 只依賴底層 load_data+format_rawdata，不碰 experiment.v2）；互動 widget 自建 canvas 不走 plot_host；MCP method_specs + skill 草案（無 connect-to-hardware）。
- **順序評估（2026-06-03）**：先遷 main→app/ vs 先做 fluxdep 兩工作量基本獨立。**選順序 B**（fluxdep 放獨立位置 `lib/zcu_tools/fluxdep_gui/`、measure 不動）——用真實第二個 app 驗證共用層邊界，再一次遷 app/ 時知道抽什麼，不過早抽象。

## === Phase 4：v1 完成（互動 pipeline，2026-06-04，b36c13bf）===

**v1 全功能實作完畢、全面驗證**（9 commits）：State / 6 service / EventBus / Controller / app+MainWindow / 4 互動 widget（LinePicker 定線 / FindPoints brush / Selector 跨譜篩選 / OneTone 自動）/ 編輯區階段切換接線 / remote RPC + MCP（16 tools）+ skill 三副本。獨立於 measure-gui（零 runtime 耦合）。

- **v1 pipeline**：載入 hdf5 → 拖線定 half/integer flux → 選點（OneTone 自動 / TwoTone brush）→ 累積多譜（可繼承對齊）→ 跨譜篩選 → 匯出 spectrums.hdf5。
- **設計要點**：State 用 persistance 的 SpectrumData/PointsData（TypedDict）；LoadService 不碰 experiment.v2；EventBus 以 payload-type 為訂閱 key（比 measure 精簡）；互動 widget 自建 FigureCanvasQTAgg + mpl_connect。
- **worker 化 LoadService 不做（YAGNI）**：真實檔載入僅 15–22ms，同步夠快。
- **多輪反饋**（v1 後）：軸轉置載入（transpose_axes）；markers/contrast/result preview/redo/resizable；inherit 修復 / OneTone magnitude 鎖 / async findpoint worker / spectrums 重載；type 持久化修復（h5 group attr，OneTone restore 不再變 TwoTone）；OneTone 拖動效能（重用 scatter + debounce redraw）。
- **驗證**：126 tests passed、pyright 0、ruff clean、MCP 端到端 smoke 綠（真實 Q3_2D fixture）。

---

## === Phase 5：v2 search + 視覺化 完成（2026-06-04）===

**v2「搜資料庫」實作完畢**（只做 database search，**不做** scipy `fit_spectrum` 精修——用戶定）。

**核心 commits（129f2716→4aec0253）**：
- `129f2716` v2 fit 領域：`FitState`（State singleton，version key `fit`，持 db/EJb/ECb/ELb/transitions/r_f/sample_f + 結果 params）、`FitService`、matplotlib `viz`（重寫 notebook plotly `FreqFluxDependVisualizer`）。
- `501f5ccc` v2 fit UI：AnalyzePanel + `_SearchWorker` + TransitionsForm + 進度條。
- `56ea0a16` v2 fit RPC + MCP：search / result / export_params（**註：這批的 mutating RPC 後在 Phase 6 移除**）。
- `4aec0253` v2 fit 測試。
- `962b9190` 把 `search_in_database` 進度走 `make_pbar`（非 gui scope）。

**State 邊界（守 main-thread 不變式）**：search 拆 `compute_search`（純函式，snapshot 輸入後跑 search，**不寫 State**，可 worker 跑）+ `record_result`（唯一寫 State 處，只主執行緒）。GUI worker 路徑用此分拆。

**能量模擬加速（database 生成的瓶頸，~20h→幾分鐘）**：
- `2039b5a1`/`5225dd6c`/`9ce7a2b1` `calculate_energy_vs_flux` 快路徑（cos(φ±β) 分解避開 per-flux `scipy.linalg.cosm`，~100x）。
- `70f0d20e` matrix-element vs-flux sweep 用 `threadpool_limits` 釘 BLAS 1 thread（40×40 小矩陣多線程是淨損，~150x）。
- `0a3810ff` 還原 `spectrum_data` passthrough 死參數（用戶跳過重複計算用，不是死碼）。
- `2a5ce1ea` `generate_fluxonium_sample.py` 重構成 CLI flag + preset + overwrite guard。

**v2 UI 多輪反饋（37044d83→bcb11ec7）**：
- `37044d83` **內嵌 matplotlib 後端**（plot_host + 自訂 `module://` backend，修 worker `plt.show()` 崩潰；仿 measure-gui plot_host/mpl_backend）。
- preset 綁 EJ/EC/EL **bounds** 不綁 transitions（`15852ddc`/`f58208f0`）。
- `5f9da3ef` filter/search/show 三步合進**單一 Analyze 面板**（QTabWidget）。
- friendly 錯誤訊息（`f2371f4e`/`a67c5122`）。
- Project 對話框（chip/qub + Browse + 自動衍生 result_dir）、unknown_* 預設、file dialog 預設目錄（`0a7281b9`/`6450bfa5`/`3d5c8190`/`b3b72e1d`）。
- `39261a79` r_f/sample_f 改 Optional + 缺值前置檢查。
- Filter 渲染/UX 修正（`273b7a7b`/`2eb74276`/`f1cc9c74`/`4789e838`/`541aad30`/`bcb11ec7`/`e2569a05`）：首開即渲染、點色對比、選中點在上層、控制面板靠左、乾淨關閉。

**路徑語意重構**（`2f3c6d0c`/`02707302`）：`ProjectInfo.result_dir`/`database_path` 改 **eager 衍生**（`__post_init__` 從 chip/qub 衍生 `result/<chip>/<qub>`，永遠實值非空 sentinel），存檔點不再 per-call-site fallback；釐清 `database_path`=raw spectrum root（衍生）vs `Database/simulation`=search db（共用、project-independent）兩角色。

## === Phase 6：remote/MCP 轉只讀 完成（2026-06-04，87bdb923）===

**決策**：fluxdep 的 RemoteControlAdapter 從「第二個 driving view」翻轉成 **read-only observing view**。agent 只觀測、**user 在 GUI 驅動分析**。記 memory `project_fluxdep_readonly_remote`。

**Why**：MCP 自測（用 Database/TestChip/Q1）暴露——選點（拖線/刷 brush）與判斷軸向是否正確本質需人眼看 GUI preview，agent 沒有。agent 用 peak-detection 代替人工挑點，產出的點集品質差到 fit 撞 bounds。**工具能跑通 ≠ 結果可信**。承用戶反問「user 有 preview 能判斷軸對不對，agent 呢？」。

**改動（commit 87bdb923，+305/−2110）**：
- **移除 11 個 mutating wire method**（project.setup / spectrum.load·load_processed·remove·set_active / alignment.set / points.set / selection.set / fit.set_params·search·export_params / export.spectrums）+ 其 dispatch handler + coercion helper + ParamSpec 工廠。
- remote 只剩 **6 個純查詢**（project.info / spectrum.list / selection.pointcloud / fit.result / resources.versions / state.check）。`test_registry_is_read_only` 守線。
- **MCP 工具集**：4 讀工具（resources.versions 不曝露）+ launch/connect/disconnect。**無 fluxdep_stop**（agent 不關 user 的 GUI；`tool_fluxdep_stop` 函式保留只給 server `_cleanup_on_exit`）。
- server instructions + SKILL（v7，三副本）重寫成只讀/user-drives。
- **刪 smoke.py / make_fixtures.py / fixtures**（操作 RPC 沒了，socket 層端到端 smoke 不再可能）；`sync_skills.sh` 只同步 SKILL.md。

**順帶釐清（同 session）**：
- `points.set` freqs 單位描述誤標 MHz → 改 GHz（`9bdfe0dc`）。整條 pipeline（loader Hz→GHz → SpectrumData → search）全程 GHz，無 MHz 介面。
- **onetone 軸反根因**：`datasaver.load_data` 寫死假設 Labber step channel = `[Flux, Frequency]`；OneTone 量測常存成 `[Frequency, Flux]` → 軸反。**非固定特性**（TwoTone 通常正、OneTone 常反），GUI「Transpose axes」toggle 讓 user 從 preview 判斷後交換。

**驗證**：199 tests passed、pyright 0、ruff clean、MCP 工具集確認（只讀 + lifecycle，無 stop、無 mutating）。

**待用戶實機**：拖線/選點/fit/視覺化全流程視覺手感（agent 只能 launch + 讀狀態，不可代為操作）。
**共用層抽取**（gui/ 頂層共用 + app/{main,fluxdep}/）待兩 app 穩定後——仍未排程。

---

## 最終架構願景（用戶定錨，收斂目標非當前步驟）

```
lib/zcu_tools/gui/
  <最基本共用工具>     # mpl 轉發後端+註冊機制、progress 後端 —— 剛好能共用就好,不強迫
  app/
    main/             # = 目前的 measure-gui
    fluxdep/          # 獨立分析 app 子模塊
    <other apps...>   # 平行的獨立 app
```

**演化順序（重要）：** 1) 複製一份完整 gui/ → 改成 fluxdep 需求（已完成）。 2) fluxdep 需求穩定後，回頭把「剛好能共用」的工具（mpl backend/progress 後端等）抽到 `gui/` 頂層，measure-gui 收進 `app/main/`、fluxdep 收進 `app/fluxdep/`。

> 用戶判斷錨點：**先複製換取獨立演化自由度，需求穩定後再收斂共用層**——不在需求未明時過早抽象。（此收斂步驟對應 gui 計劃 `task_plans/gui/` 的 Phase 133。）

## 已知風險

- **互動 UI 是新建能力**：現有 GUI plot 層純展示零互動，圖上拖線/brush 是從零寫的部分（工作量主體）。
- **重依賴**：scqubits + numba（fitting.py 首次搜庫觸發 njit 編譯）。
- **search/fit 是 CPU-bound 同步任務**：用 worker thread + 進度回呼（借鏡 RunService 的 QThread 模式，不複用其 cfg/run 語義）。
- **scope 紀律**：本任務在 `task_plans/tool_gui/`，與 `task_plans/gui/`（measure-gui）分離。當前階段不動 measure-gui 任何碼；未來抽共用層階段才會動 `gui/`，屆時另立 scope + 取得批准。
- **複製的代價（已知並接受）**：傳輸/版本表/EventBus 機制暫有兩份，修 bug 需手動同步——換取獨立演化自由度，需求穩定後抽共用層收斂。
