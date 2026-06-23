---
status: proposed
---

# 實驗資料持久化：labber_io 原生 axes-list（inner-first, N 維）為唯一介面 + per-experiment axes-spec

**狀態：** proposed（**決策已定、實作分批進行中**；code 尚未全面對齊本檔——遷移為增量 phase，見下「遷移」段。本檔以**目標生效設計**現在式描述）。
**關聯：** 與 [[0015]] 劃清（PersistenceCaretaker 是 GUI app-state 的 Memento 單檔持久化，與此**實驗量測資料**的 HDF5 round-trip 不同層、不同關注點）；save 動作位於實驗 run path 尾端，故與 runner / run-driver 設計相鄰。

## 脈絡

實驗量測資料（onetone/twotone/singleshot/… 的 1D/2D sweep 結果）的存取目前一律經 `lib/zcu_tools/utils/datasaver.py` 的 **dict 殼**：`save_data` / `load_data` / `save_local_data` / `load_local_data`，以 `x_info` / `y_info` / `z_info` 三個 `{name, unit, values}` dict 為介面。這層殼底下是真正的引擎 `lib/zcu_tools/utils/labber_io.py`（純 h5py/numpy 的 Labber Log Browser 格式 reader/writer）。

殼洩漏了它的核心不變式——**軸序**：
- `save_local_data` 收 z 為 `(Ny, Nx)`（外-y、內-x），直送 labber_io。
- `load_local_data` 卻在 load 時把內兩軸翻成 `(Nx, Ny)`「frequency-major」（`datasaver.py:173-179`），理由是一個 labber_io 自己**根本不使用**的「historical frequency-major contract」。

於是 **save 收 `(Ny,Nx)`、load 回 `(Nx,Ny)`，round-trip 非恒等**。每個 caller 私下補 transpose 來抵銷，且補法**互相矛盾**：ac_stark 系把記憶體 array `.T` 後存、load 直接用；power_dep 系直接存、load 卻 assert `(Ny,Nx)`（load 實際回 `(Nx,Ny)`，非方陣即 fail，且無測試覆蓋——是個 latent bug）。約 38 個 2D caller 各帶一份 ad-hoc 軸序記帳。

同時，約 64 個實驗各自**重抄** save/load 樣板（None-guard → make_comment → save_data；load_data → shape assert → 單位反轉 → parse_comment → validate_or_warn → rebuild last_result），差別只在軸名/unit/scale 與 Cfg 型別——同一個持久化不變式被實作 N 次。

關鍵事實：**labber_io 本身原生就支援 inner-first 的 axes-list + N 維 z**（save 驗 `len(axes)==z.ndim`、load 以 `Step dimensions` attr 重建 hypercube、有通過的 3D round-trip 測試）。軸序慣例**已經住在 labber_io**；擋在中間製造矛盾的只有 datasaver 殼的 load-flip。

## 決定

1. **唯一持久化介面 = labber_io 原生**。實驗資料一律經 `save_labber_data(path, z=(name,unit,values), axes=[(name,unit,values)…])` / `load_labber_data(path) -> LabberData`。**刪除** datasaver 的四個 dict 函式（`save_data` / `load_data` / `save_local_data` / `load_local_data`）；不保留相容殼（依 CLAUDE.md「不保留 legacy / 相容性邏輯」）。

2. **軸序慣例（唯一權威住 labber_io）**：`axes` 以 **inner-first** 排列；`z.shape == tuple(len(ax) for ax in reversed(axes))`，即 **inner 軸恆為 z 的最後一維**。1D `(Nx,)`、2D `(Ny, Nx)`、N 維 `(…, Ny, Nx)`。**load 是 save 的恒等逆**——任何一邊都**不做 caller-side transpose**。`.T` 與「transpose back」在 caller 中絕跡。

3. **per-experiment typed axes-spec**（把實驗資料持久化與 save/load 樣板合為一個 deep module）：每個實驗宣告一份 **typed axes-spec**——軸 `name`/`unit`/`scale`/順序、z channel、哪些是離散狀態軸、SI-on-disk 的單位轉換——由**一個共用 helper** 依此驅動原生 N 維 save/load。axis-order + unit + N 維不變式**只住一處**；per-experiment 的 save/load 樣板消失，實驗只留宣告式 spec。

4. **N 維折疊延後**。目前約 12 個站用「多次 `save_data` 疊」假裝高維（ckp 的 g/e、ac_stark 的 g/e、reset/bath 的 4 相位、mist 的 g/e、t1 的 state-pair…）。真 N 維 axes 介面可把同質的這批折成**單次 save + 一條離散軸**，但**留待後續獨立 phase**（會變更檔案佈局，下游分析/notebook 讀法須同步）。本批先讓每個 slice 各自走原生介面。autofluxdep 的**異質**疊（signals + length + fit，shape 不同）不折成同質 cube，維持多 save。

5. **存活的非資料 helper 另立家**。`safe_labber_filepath`（檔名自動編號）、remote `server_ip` 上/下載、`get_datafolder_path` / `create_datafolder` 不屬 dict 殼、且被 GUI / script / notebook 依賴 → **保留**，搬進獨立的路徑/傳輸 helper，不與被刪的 dict API 綁在一起。

6. **強型別 + Fast-Fail**。原生 `Channel` 是 namedtuple、`LabberData.z`/`.x` 為 `Any`；axes-spec 層補上型別化包裝（frozen `Axis` 與型別化結果），save 時驗 `z.shape` 與 axes 不符即 **raise**（不靜默 transpose）。

7. **遷移 = 增量分批**（非 big-bang）。先立本 ADR + 釘軸序慣例；第一步 = 刪 load-flip（`datasaver.py:173-179`）+ 修 power_dep latent bug + 改 `test_datasaver` 的 2D 斷言成 round-trip 恒等；新 axes-spec 介面就位後，約 75 個 caller 檔分批 phase 遷移；同一 session 不一次全改，每批可獨立驗證。

## 理由 / 取捨

- **Deletion test**：刪掉 datasaver dict 殼後，複雜度不會散到 38 個 caller，而是**集中**到 labber_io 一個有型別的 axes 介面 + 一份共用 axes-spec helper——這是「淺殼變深 module」的 deepening，而非 pass-through 搬家。
- **軸序只住一處** → 根除「每 caller 自決朝向」的漂移類 bug（含已存在、未測的 power_dep load 斷言錯誤）。新實驗無法再各自引入不一致。
- **labber_io 不動**（1420 行的 on-disk 引擎，earns its keep）：它早已是 inner-first / N 維 / 軸序權威，這次只是**移除擋在前面的殼**讓 caller 直接講原生介面。
- **N 維折疊延後**：折疊改變檔案佈局、牽動下游分析與 notebook 讀法，blast radius 大；獨立 phase 較安全，且不阻擋「去殼走原生」這個主幹。
- **跨模組**（`utils/` + `experiment/`、影響約 75 caller）→ 立 ADR 而非僅模組 README；模組局部速查另補 `lib/zcu_tools/utils/README.md`。
- **協調風險**：`lib/zcu_tools/utils` 與多個 `experiment/` 路徑當前被另一 session（`codex-wavelet-signal2real`）持 write claim → **實作 phase 須待其釋出或先協調**；本 ADR、慣例釘樁與規劃不受此阻擋，可先行。
