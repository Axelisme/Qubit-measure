# `zcu_tools.analysis.fluxdep` 模塊重點文檔

**Last updated:** 2026-09-27 — shared transition calculations

本模塊提供 Flux-Dependence Analysis 的 notebook-neutral 數值規則。它承接 notebook
與 Qt GUI 共用的互動選點、filtering、line selection、one-tone peak detection 規則；adapter
只負責轉譯 UI 事件與渲染容器。

## 範圍

- `models.py`：`PointsData`、`SpectrumResult`（含每張譜的 flux calibration）、`TransitionDict`，以及 `energy2linearform(energies, transitions)` 與 `energy2transition(energies, transitions)`。兩者接受 `(N_flux, M_levels)` 能譜與躍遷定義；前者回傳縮放係數的線性形式 `(B, C)`，後者回傳 GHz 躍遷頻率與標籤。躍遷的 sideband／mirror 分支需要對應的 `r_f`／`sample_f`。物理能譜的生成仍屬 simulate。原始 `SpectrumData` 與 Hz→GHz 順序規則由 [`analysis.spectrum`](../spectrum.py) 擁有。
- `io.py`：`dump_spectrums` / `load_spectrums` 寫讀 HDF5 spectrum collection。每張譜有 calibration、原始頻譜與 points；`type` 是選用的 group attribute。這與 canonical Experiment Data File loader 是不同契約。
- `processing.py`：頻譜轉實數/正規化、2D peak detection、point downsample、mirror difference。
- `selection.py`：brush selection 的幾何規則，供 grid mask 與 joint point cloud 使用。
- `onetone.py`：one-tone 最大色散頻率、切面平滑、peak detection 與點位輸出。
- `line_state.py`：Qt-free `FluxPickState` 與唯讀輸入、初值折疊、移線/交換、mirror-loss 與 auto-align 候選計算。狀態不保存選線、preview 或 Figure。
- `line_picker.py`：既有 notebook Matplotlib picker；measure Qt frontend 只用它維護本地 artists/timer，並以 `show_state()` 將 committed state 重投影到畫面。

## 邊界

本 kernel 不包含 database search、search 診斷圖、plotly/matplotlib export 圖、params.json export，
也不負責 GUI/ipywidgets lifecycle。Notebook module 與 Qt interactive widget 透過 re-export 或
thin adapter 使用本 kernel。measure frontend 使用相同數值規則驗證本地 preview 與第二次點擊的提交；按鈕 release 不提交。
committed state 與 operation 的權威由 measure app service 管理，不在本 kernel。
