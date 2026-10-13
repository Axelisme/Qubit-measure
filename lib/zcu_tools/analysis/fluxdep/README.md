# `zcu_tools.analysis.fluxdep` 模塊重點文檔

**Last updated:** 2026-10-13，recorded quality、transition 能階需求、adapter 分工與 OneTone 背景正規化

本模塊提供 Flux-Dependence Analysis 的 notebook-neutral 數值規則。它承接 notebook
與 Qt GUI 共用的互動選點、filtering、line selection、one-tone peak detection 規則；adapter
只負責轉譯 UI 事件與渲染容器。

## 範圍

- `models.py`：`PointsData`、`SpectrumResult`（含每張譜的 flux calibration）、`TransitionDict`，以及 `energy2linearform(energies, transitions)` 與 `energy2transition(energies, transitions)`。兩者接受 `(N_flux, M_levels)` 能譜與躍遷定義；前者回傳縮放係數的線性形式 `(B, C)`，後者回傳 GHz 躍遷頻率與標籤。躍遷的 sideband／mirror 分支需要對應的 `r_f`／`sample_f`：`validate_transition_frequencies` 是兩者與 `search_models.compile_transitions` 共用的規則，只有清單非空的類別才需要頻率，mirror 類別一律需要 `sample_f`，有 `r_f` 也不能代替。物理能譜的生成仍屬 simulate。原始 `SpectrumData` 與 Hz→GHz 順序規則由 [`analysis.spectrum`](../spectrum.py) 擁有。
- `io.py`：`dump_spectrums` / `load_spectrums` 寫讀 HDF5 spectrum collection。每張譜有 calibration、原始頻譜與 points；`type` 是選用的 group attribute。這與 canonical Experiment Data File loader 是不同契約。
- `search.py`：`search_database` 讀取預先計算的 fluxonium HDF5 database，對 `(fluxs, freqs)`、transition 定義和 `ParamBounds(EJ, EC, EL)`（GHz 範圍）做 exact lower-bound-pruned 搜尋。`DatabaseSearchResult` 提供最佳 `(EJ, EC, EL)`、評分、逐 entry 距離/縮放、搜尋輸入與預測頻率，供 [診斷圖 builder](../../plotting/fluxdep/README.md) 使用。結果另有每點的 `transition_indices`、GHz 絕對殘差 `residuals` 與 `outlier_mask`。`quality` 統一提供平均殘差、偏大點數、門檻及每條 transition 的點數／平均／最大殘差，讓 GUI 和 RPC 不必重算。transition 順序沿用編譯順序；等距時取第一條，未用到的 transition 仍列出，平均／最大為 `None`。偏大點採嚴格大於全體平均殘差的 3 倍；全零殘差沒有偏大點。`search_models.py` 與 `search_njit.py` 保留原有 transition compilation 和數值核心；`fit_spectrum` 仍屬 Notebook。搜尋不決定 Figure 顯示、參數接受或 export。Caller 可提供 worker-safe cancel_requested predicate；停止 checkpoint raise SearchCancelled，不回 partial result。未結束的 HDF5／Numba call 會延後觀察取消，沒有固定延遲。既有 KeyboardInterrupt 的 best-so-far 回傳是另一條路徑。
- `processing.py`：頻譜轉實數/正規化、2D peak detection、point downsample、mirror difference。`onetone_dip_depth` 與 `contrast_limits` 是 OneTone 背景正規化，GUI 預覽、`TwoLinePicker` 與 plotting 共用；放在這裡讓 `TwoLinePicker` 不必 import plotting。
- `selection.py`／`stroke.py`：grid mask 與 joint point cloud 共用的 normalized brush 幾何與完整 stroke sampling budget；`BrushTool`／`BrushStroke` 同時供 TwoTone 與跨譜篩選使用。
- `cross_selection.py`：capture 聯合 calibrated flux／GHz cloud 與唯讀背景；完整 mask 先 brush 再 deterministic downsample。差量以 cloud index 保留重複座標身份；projection 可使用 previous snapshot 呈現 Undo inverse，不改 committed state。
- `twotone.py`：captured read-only spectrum、detector/settings/tool validation、detached mask transitions 與 sorted native points。Projection 比較點位與 mask 的新增／移除；可用 previous snapshot 呈現 Undo inverse，不更動 restored state。Stroke 幾何與 sampling budget 由 `stroke.py` 擁有。
- `onetone.py`：one-tone 最大色散頻率、切面平滑、peak detection 與點位輸出。`OneToneInputs` 保存 captured spectrum 與唯讀預處理；`OneTonePickState` 保存完整 threshold／indices。`analyze_onetone_pick` 驗證 indices 並輸出未校準 device／GHz 點位，空選集有效；排序與 flux calibration 留在 app。
- `line_state.py`：Qt-free `FluxPickState` 與唯讀輸入、初值折疊、移線／交換、mirror-loss 與 auto-align 候選計算。`analyze_flux_pick` 驗證終態最小間距，回傳 device-axis 單位的 FluxPickAnalysis，period 是兩線距離的兩倍。計算不改輸入；狀態不保存選線、preview 或 Figure。
- `line_picker.py`：既有 notebook Matplotlib picker；measure Qt frontend 只用它維護本地 artists/timer，並以 `show_state()` 將 committed state 重投影到畫面。`onetone=True` 只改主圖背景（OneTone dip 深度），對齊與 mirror loss 仍用投影。

## 輸入與能階契約

`search.database_info(path)` 讀 HDF5 的參數範圍與能階維度，不 materialize energies。
`search_models.transition_requires_levels` 先驗類別語法，指出是否需 DB 能階數。
非空 pairs 與動態類別（含空列表）需要 M；基本空列表與空 mapping 不需要。
`validate_transition_levels` 與 compilation 共用類別順序，拒絕未知類別、動態 order 不在 2..M-1 與索引不在 0..M-1。
`count_max_evals` 為 simulation 回最少能階數，包含非空 pair 最大索引 +1 與動態 order +1；空類別不增加計算量。GUI overlay 與 Notebook least-squares 都引用這個契約。
OneTone／TwoTone detector、brush radius 與跨譜 distance 的有效範圍由各 domain 模組常數提供，驗證與 GUI command schema 引用同一來源。

## 邊界

Kernel 的 quality 是同次 search 的數值證據，不是接受／拒絕 fit 的政策。GUI 保存它與原始 search cloud；MCP 只摘要 recorded quality，不另算殘差。Stage freshness、草稿與下游失效屬於 GUI app，不在數值 kernel。

本 kernel 不包含 search 診斷圖、plotly/matplotlib export 圖、params.json export，
也不負責 GUI/ipywidgets lifecycle。Notebook module 與 Qt interactive widget 透過 re-export 或
thin adapter 使用本 kernel。measure frontend 使用相同數值規則驗證本地 preview 與第二次點擊的提交；按鈕 release 不提交。
committed state 與 operation 的權威由 measure app service 管理，不在本 kernel。
