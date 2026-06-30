# fluxdep 模塊重點文檔

**Last updated:** 2026-07-01

本模塊提供 Fluxonium 通量依賴光譜（flux-dependent spectrum）的擬合、資料處理、
與互動式標註工具。搭配 `notebook_md/analysis/fluxdep_fit.md` 使用。

## 模塊結構

```
fluxdep/
├── __init__.py        # 對外 API 匯出
├── models.py          # 能量 → 躍遷頻率的線性形式轉換
├── fitting.py         # 資料庫搜索 + least-squares 微調
├── njit.py            # numba JIT 核心：energy2linearform_nb、eval_dist_bounded、candidate_breakpoint_search、entry_lower_bound、_lower_bound_kernel、search_one_entry 等
├── processing.py      # re-export zcu_tools.analysis.fluxdep processing kernel
├── onetone.py         # InteractiveOneTone：onetone 峰值挑點 notebook adapter
├── utils.py           # 繪圖工具（plotly 可視化）
└── interactive/       # matplotlib + ipywidgets 互動式工具
    ├── find_line.py    # InteractiveLines：拖曳半通量/整通量定標
    ├── find_point.py   # InteractiveFindPoints：筆刷式選取光譜點
    ├── point_select.py # InteractiveSelector：多光譜點選擇/過濾
    └── two_line_picker.py # re-export shared TwoLinePicker kernel
```

共用的 Flux-Dependence Analysis 選點/filtering/line selection/one-tone peak detection
規則位於 `zcu_tools.analysis.fluxdep`（ADR-0028）。Notebook 這層保留 fitting/model/visualizer
與 ipywidgets shell；被抽出的互動與 processing API 透過 thin re-export 或 adapter 呼叫 kernel。

## 核心物理模型（`models.py`）

### `TransitionDict`

`TypedDict` 描述要擬合的躍遷類型，允許 key：

- `transitions`：`E_ji` 直接躍遷
- `blue side` / `red side`：`E_ji ± r_f` 色散旁帶（需 `r_f`）
- `mirror`：`sample_f - E_ji`（需 `sample_f`）
- `mirror blue` / `mirror red`：`sample_f ± r_f ∓ E_ji`
- `transitions{n}` / `mirror{n}`：n-光子躍遷 `E_ji / n`
- 特殊浮點 key：`r_f`（readout 頻率）、`sample_f`（sampling / drive 頻率）

每個躍遷 key 對應到 `list[tuple[int,int]]`，表示 `(i → j)` 能級。

### `energy2linearform(energies, transitions) -> (B, C)`

把能量陣列 `(N_flux, M_levels)` 轉為線性形式 `(B, C)`，
使得躍遷頻率 = `|a * B + C|`，其中 `a` 是整體縮放因子。
**這是資料庫搜索的關鍵技巧**：能量對 `(EJ, EC, EL)` 並非線性，
但在縮放 `a*(EJ, EC, EL)` 下能量線性縮放 `a*E`，所以可快速搜索最佳 `a`。

### `energy2transition(energies, transitions) -> (freqs, names)`

直接計算躍遷頻率與人類可讀標籤，用於繪圖疊加。

## 擬合流程（`fitting.py`）

兩階段擬合：

### 1. `search_in_database(fluxs, freqs, datapath, transitions, EJb, ECb, ELb)`

在預先生成的 Fluxonium 資料庫中做搜尋（**唯一一條精確路徑**，舊 fuzzy 啟發式已移除）：

- **資料庫結構**（由 `script/generate_fluxonium_sample.py` 產生）：
  - `fluxs`：(N_flux,) 通量點
  - `params`：(N_sample, 3) 的 `(EJ, EC, EL)` 取樣點（Fibonacci lattice 在球面上均勻分布，
    並按 `EJb/ECb/ELb` 立方體做射線相交篩選 → 代表「方向」而非「位置」）
  - `energies`：(N_sample, N_flux, M_levels) 預計算能譜
  - `Ebounds`：(3, 2) 對應 `EJb, ECb, ELb`
- 對每個樣本 `param` 計算線性形式 `(B, C)`，然後在允許的 `a_min, a_max` 區間
  內找最佳縮放 `a`，使 `F(a)=mean_i min_j |freq_i - |a*B_ij + C_ij||` 最小。
- **精確解 + LB-prune**：用 `candidate_breakpoint_search` 精確解（遍歷候選斷點 `a=(±A_i-C_ij)/B_ij`），但**不**掃全部 entry —— 先 `_lower_bound_kernel`（parallel）算每 entry 的 **exact lower bound** `LB = mean_i min_{a∈range} min_j |...|`（`entry_lower_bound`：鬆綁「共用 a」→ 每點各取最佳 a，必 ≤ min_a F(a)，O(N·K²)），按 LB 升冪用 `search_one_entry` full-search、維護 incumbent，`LB > incumbent` 即停（被略過 entry 數學上不可能更好）。**結果與掃全部 entry 的 full exact bit-identical**（真匹配 LB≈0 排最前、incumbent 速降），典型只 full-search 0.1–14% entry。
- 回傳 `(EJ, EC, EL)` 與診斷圖（預測 vs 目標頻率 + 3 個參數的 distance 散點圖；prune 下未搜 entry 的散點用 LB 當距離下界）。
- `n_jobs` 設 LB pass 的 numba 執行緒（`-1`=全核心）。

**效能（profiling，10091-entry DB、300-pt cloud）**：三個優化（前兩個 **bit-identical**，第三個讓精確解變預設）：
- **`load_database()` + `_load_database_cached` LRU**（keyed on path+mtime+size）：GUI 對同 DB 重複跑 search，省 cold-read ~0.13s/call。
- **只插值 referenced levels**：`used_levels = unique(tr_pairs)`、remap pairs 到 reduced index 後才 `_apply_interp`（15→~3 level，intermediate 363→73MB，插值 35→8ms，更好並行）。plot 分支重建 best entry 的 full-level energies。
- **exact LB-prune（演算法層）**：見上，4–15x 且精確（取代舊 fuzzy 預設）。等價測試 `test_prune_is_identical_to_full_exact_scan`（noisy cloud 0 mismatch）+ `test_entry_lower_bound_is_a_valid_floor`。
- **移除的舊 fuzzy 啟發式**：`smart_fuzzy_search`（直方圖密度 + 降採樣）+ 分支用的 `_search_kernel` 已刪 —— 它本身就是近似（noisy cloud 跟 exact 在 17/60 案例不同），LB-prune 精確版又更快，故統一掉。`candidate_breakpoint_search`/`eval_dist_bounded` 保留（精確搜尋核心）。
- **走過的死路**：coarse-to-fine 降採點 screen 雖 8x 但選到不同 DB entry、relerr 高達 240%（DB Fibonacci-lattice 樣本太密集退化）；向量化全候選比 njit 慢 88x（branch-prune 太有效 + (M,N,K) materialize 太大）。

### 2. `fit_spectrum(fluxs, freqs, init_params, transitions, param_b, maxfun)`

以 `search_in_database` 結果為初值，`scipy.optimize.least_squares`（`soft_l1` loss）
微調。residuals 每次重算 scqubits 能譜，故遠比資料庫搜索慢——資料庫搜索負責定區域，
這裡負責精修。

## 資料處理（`processing.py`）

`processing.py` 是 `zcu_tools.analysis.fluxdep.processing` 的 re-export，保留 notebook 既有 import
路徑；共用 domain 規則不在 notebook adapter 內維護。

- `cast2real_and_norm(signals, use_phase=True, sigma=1, smooth_method="wavelet")`：
  預設減去每列平均（去除 flux-independent background），沿頻率軸做 wavelet smoothing 後取 `|signal|` 並按列 std 歸一化；`smooth_method="gaussian"` 保留為舊行為對照。`use_phase=False` 退回純 magnitude 模式。
- `spectrum2d_findpoint(dev_values, freqs, real_signals, threshold, weight=None)`：
  對每個通量（每列）呼叫 `scipy.signal.find_peaks`，最多保留 3 個最高峰。回傳打平的 `(fluxs, freqs)`。
- `downsample_points(xs, ys, threshold)`：2D 空間去密集點（同一 x 下的點保留為一組）。
- `diff_mirror(xs, data, center)`：以 `center` 為中心的 1D 鏡像差，用於 `InteractiveLines` 的對稱性評分。

## 互動式工具

全部使用 `matplotlib` + `ipywidgets`，呼叫後會在 notebook 顯示 GUI，
呼叫 `.get_positions()` / `.get_cur_selected()` 拿結果（會自動 finish）。

| 類別 | 用途 | 輸入 | 輸出 |
|---|---|---|---|
| `InteractiveOneTone` | onetone 資料找共振 dip vs flux | `signals, dev_values, freqs, threshold` | `(dev_values, freqs)` 於最大梯度頻率切面 |
| `InteractiveLines` | 拖曳紅（half flux）/藍（integer flux）線做 mA→Φ 定標；支援 auto-align、swap、magnitude-only 切換 | `signals, dev_values, freqs, flux_half?, flux_int?` | `(flux_half, flux_int)` |
| `InteractiveFindPoints` | 筆刷式遮罩 + 自動 `find_peaks` 從 2D 光譜擷取點 | `signals, dev_values, freqs, threshold, brush_width` | `(fluxs, freqs)` |
| `InteractiveSelector` | 跨多個 `SpectrumResult` 用筆刷挑選/過濾點 | `spectrums: dict[str, SpectrumResult], selected?, brush_width` | `(fluxs, freqs, mask)` 3-tuple |

## 可視化工具（`utils.py`）

`plotly` 建構 figure 的 builder-pattern 類別（每個方法回傳 `self`）：

- `FluxDependVisualizer`：基底類別，負責加次要 x 軸（`flux` ↔ `dev_value` 雙軸顯示）。
- `FreqFluxDependVisualizer`：追加
  - `plot_background(spectrums)`：2D 熱圖底圖（多個 spectrum 疊加）
  - `plot_simulation_lines(fluxs, energies, transitions)`：疊加擬合出的躍遷線
  - `plot_points(fluxs, freqs, **kw)` / `plot_sample_points(sample_table, convert_fn)`：散點
  - `plot_constant_freq(freq, name)`：水平參考線（例如 readout 頻率）
  - `auto_derive_limits()`：依所有已加入的資料自動決定 x/y 範圍

輔助：`derive_bound(spectrums, convert_fn)` 取 min/max；`add_secondary_xaxis` 加次要 x 軸。

Figure builder 內部把 `self.fig` 視為 concrete `go.Figure`。建構時若 caller 未傳入 figure，就立即建立並儲存具體 figure；Matplotlib path 建圖後用 runtime assert 確認是 `Figure`。不要用 `cast()` 取代這個 invariant。

## 資料庫產生：`script/generate_fluxonium_sample.py`

- 輸出 `Database/simulation/fluxonium_all.h5`（或 `_dryrun.h5`）。
- 參數範圍由 `EJb, ECb, ELb` 的立方體定義；樣本以 Fibonacci lattice
  均勻分布於單位球正卦限，再經 `ray_intersects_box` 篩出穿過目標立方體的射線。
- 實際能譜由 `zcu_tools.simulate.fluxonium.calculate_energy_vs_flux` 計算
  （`cutoff=40, evals_count=15, fluxs = linspace(0, 0.5, 120)`），最後靠
  `Φ → 1−Φ` 對稱性把 120 點延伸為 240 點。
- `--dry-run` CLI 旗標會用隨機能量做 schema 測試（不要拿去擬合）。
- 預設 `num_sample=10000`。`search_in_database` 要能讀取此 HDF5 結構（見上節）。

## 常見擴充點

- 新增躍遷類型：同時更新 `energy2linearform` 與 `energy2transition`，
  並確保 `count_max_evals` 能正確從新 key 推出需要的能級數。
- 自訂擬合度量：改 `fitting.py` 裡 `eval_dist`（`@njit` 簽章固定，改時要同步改 decorator）。
