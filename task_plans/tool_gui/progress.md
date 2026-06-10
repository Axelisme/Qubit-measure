# Progress Log：tool_gui（fluxdep_fit GUI 移植評估）

## 2026-06-03 — Phase 0–4：設計 + v1 實作（濃縮，細節見 task_plan.md / findings.md / phase{2,3}_*.md）

逐日推進（Phase 0 基準 1e4cb1d8 → v1 完成 b36c13bf）：
- **Phase 0–3 設計**：探查可行性 → 架構決策 D1–D5（領域全新寫 / 減法搬運 / Qt mpl event / 統一 matplotlib / 同構獨立 server）→ 領域模型（FluxDepSession 線性 pipeline）→ v1 詳細設計（骨架搬運清單 / State 定稿 / Service 介面 / 互動 widget 規格 / MCP+skill 草案）。Q1–Q5 開放問題逐一定奪，v1 範圍收斂為「前半骨架」（載入→定線→選點→篩選→存 spectrums.hdf5，不含搜庫/fit）。
- **上游重構（measure-gui 側，獲批准）**：`wire.py` 去設備領域耦合 + wire field 原語改公開（commit 25a3eee4），為 fluxdep 搬運鋪路。
- **Phase 4 v1 實作（9 批，順序 B）**：State+LoadService → 純運算 service 全套 → EventBus+Controller → app+MainWindow 空殼 → InteractiveMplWidget 骨架+OneToneWidget → 4 互動 widget 全完成 → 接編輯區階段切換 → remote RPC+MCP（16 tools）→ skill 三副本。worker 化 LoadService 評估後不做（YAGNI，載入 15–22ms）。
- **策略轉折**：用戶無法常駐測試 → 一口氣做完 v1 全部、最後我自己用真實 Q3_2D fixture 跑 MCP smoke。真實測試檔軸轉置（freq/flux 反）（見 memory project_fluxdep_test_files）。
- **Q3 處置（用戶定）**：search/fit 進度直接用既有 progress_bar 的 Qt 後端（注入 pbar factory），不加回呼——複用既有抽象、改動更小。

**v1 完成**：126 tests、pyright 0、ruff clean、MCP 端到端 smoke 綠。獨立於 measure-gui。

## 2026-06-04 — Phase 5 v2 search + 視覺化 + 能量加速完成

**v2「搜資料庫」全套（只 search 不 fit）**（commits 129f2716→4aec0253，見 task_plan Phase 5 完成區塊細節）：
- 領域：`FitState`(State singleton)/`FitService`(compute_search 純+record_result 主執行緒分拆守不變式)/matplotlib `viz`(重寫 notebook plotly)。
- UI：AnalyzePanel 三 tab(Filter/Search/Show) + `_SearchWorker` + TransitionsForm + 進度條。
- **內嵌 mpl 後端**(37044d83)：自訂 `module://` backend 攔 plt.figure/show 內嵌，修 worker plt.show() 崩潰。

**能量模擬加速**（database 生成 ~20h→幾分鐘）：`calculate_energy_vs_flux` 快路徑(cos(φ±β) 分解，~100x，2039b5a1/5225dd6c/9ce7a2b1) + BLAS 釘 1 thread(~150x，70f0d20e) + 還原 spectrum_data 死參數(0a3810ff) + generate 腳本 CLI 化(2a5ce1ea)。

**v2 UI 多輪反饋**：preset 綁 bounds 非 transitions、三步合一面板、friendly 錯誤、Project 對話框+衍生路徑、r_f/sample_f Optional+前置檢查、Filter 渲染/點色/層級/控制面板位置修正。**路徑語意重構**(2f3c6d0c/02707302)：ProjectInfo eager 衍生 + raw/search db 兩角色釐清。

## 2026-06-04 — Phase 6 remote/MCP 轉只讀（87bdb923）

**用 Database/TestChip/Q1 跑 MCP 自測 → 暴露根本問題**：選點與軸向判斷需人眼看 preview，agent 沒有；peak-detection 代挑點品質差到 fit 撞 bounds。**工具跑通≠結果可信**。承用戶反問「user 有 preview，agent 呢？」。

**決策**：RemoteControlAdapter driving → **read-only observing**。agent 只觀測、user 在 GUI 驅動。
- 移除 11 個 mutating RPC + handler + coercion helper + ParamSpec 工廠 → remote 只剩 6 純查詢，`test_registry_is_read_only` 守線。
- MCP = 4 讀工具 + launch/connect/disconnect，**無 stop**（agent 不關 user GUI；tool_fluxdep_stop 留給 server cleanup）。
- server instructions + SKILL(v7,三副本) 重寫；刪 smoke.py/make_fixtures/fixtures（socket 層端到端 smoke 不再可能）。
- 順帶：points.set freqs 單位誤標 MHz→改 GHz(9bdfe0dc)；釐清 onetone 軸反根因(load_data 寫死 step channel 順序假設，非固定特性)。
- 記 memory `project_fluxdep_readonly_remote`。**199 tests、pyright 0、ruff clean。**

**待用戶實機**：拖線/選點/fit/視覺化全流程（agent 只能 launch+讀狀態）。共用層抽取仍未排程。
