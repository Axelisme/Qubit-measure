---
status: accepted
---

# 實驗資料持久化：labber_io 原生 axes-list + per-experiment axes-spec + grouped experiment dataset

**狀態：** accepted（**已落地**：experiment data 持久化全面改走 labber_io 原生 axes-list；`PersistableExperiment` + per-experiment `AxesSpec`（typed Result builder、`Axis.dtype`、tag 內含）驅動 single-file 實驗；`zcu_tools.utils.datasaver` package facade 與 repo-wide import migration；Grouped Experiment Dataset low-level API；`CPMG_Exp` 單一 grouped `.hdf5` save/load；legacy CPMG `.npz` 透過 migration script 轉換；single-role 3D `RabiCheckExp` / `CKP_Exp` / bath reset freq-gain slice；legacy CKP `_ground` / `_excited` sidecar 透過 migration script 轉換。本檔以現在式描述生效設計）。
**關聯：** 與 [[0015]] 劃清（PersistenceCaretaker 是 GUI app-state 的 Memento 單檔持久化，與此**實驗量測資料**的 HDF5 round-trip 不同層、不同關注點）；save 動作位於實驗 run path 尾端，故與 runner / run-driver 設計相鄰。

**領域語言：** 本 ADR 使用 `docs/CONTEXT.md` 定義的 Experiment Result、Experiment Data File、Dataset Role、Grouped Experiment Dataset 與 Legacy Measurement Artifact。

## 脈絡

實驗量測資料（onetone/twotone/singleshot/… 的 1D/2D sweep 結果）的存取目前一律經 `lib/zcu_tools/utils/datasaver.py` 的 **dict 殼**：`save_data` / `load_data` / `save_local_data` / `load_local_data`，以 `x_info` / `y_info` / `z_info` 三個 `{name, unit, values}` dict 為介面。這層殼底下是真正的引擎 `lib/zcu_tools/utils/labber_io.py`（純 h5py/numpy 的 Labber Log Browser 格式 reader/writer）。

殼洩漏了它的核心不變式——**軸序**：
- `save_local_data` 收 z 為 `(Ny, Nx)`（外-y、內-x），直送 labber_io。
- `load_local_data` 卻在 load 時把內兩軸翻成 `(Nx, Ny)`「frequency-major」（`datasaver.py:173-179`），理由是一個 labber_io 自己**根本不使用**的「historical frequency-major contract」。

於是 **save 收 `(Ny,Nx)`、load 回 `(Nx,Ny)`，round-trip 非恒等**。每個 caller 私下補 transpose 來抵銷，且補法**互相矛盾**：ac_stark 系把記憶體 array `.T` 後存、load 直接用；power_dep 系直接存、load 卻 assert `(Ny,Nx)`（load 實際回 `(Nx,Ny)`，非方陣即 fail，且無測試覆蓋——是個 latent bug）。約 38 個 2D caller 各帶一份 ad-hoc 軸序記帳。

同時，約 64 個實驗各自**重抄** save/load 樣板（None-guard → make_comment → save_data；load_data → shape assert → 單位反轉 → parse_comment → validate_or_warn → rebuild last_result），差別只在軸名/unit/scale 與 Cfg 型別——同一個持久化不變式被實作 N 次。

關鍵事實：**labber_io 本身原生就支援 inner-first 的 axes-list + N 維 z**（save 驗 `len(axes)==z.ndim`、load 以 `Step dimensions` attr 重建 hypercube、有通過的 3D round-trip 測試）。軸序慣例**已經住在 labber_io**；擋在中間製造矛盾的只有 datasaver 殼的 load-flip。

## 決定

1. **唯一 public persistence facade = `zcu_tools.utils.datasaver` package**。實驗資料一律從 package root import public API：`save_labber_data` / `load_labber_data` / `save_grouped_labber_data` / `load_grouped_labber_data`、`Axis` / `LabberPayload` / `LabberMetadata` / `LabberData` / `GroupedLabberData` / `DatasetRole`、以及 path / transport helpers。package 內部依責任拆成不帶 underscore prefix 的子模組，例如 `models.py`、`labber.py`、`grouped.py`、`paths.py`、`transport.py`，並由 `__init__.py` re-export 對外需要的介面。feature branch 內一次完成 repo-wide import migration：移除舊 `zcu_tools.utils.labber_io` module 與舊 `zcu_tools.utils.datasaver.py` file，不保留 shim。**刪除** datasaver 的四個 dict 函式（`save_data` / `load_data` / `save_local_data` / `load_local_data`）；不保留相容殼（依 CLAUDE.md「不保留 legacy / 相容性邏輯」）。`zcu_tools.utils.labber_io` 不再是 canonical public import path。

2. **軸序慣例（唯一權威住 labber_io）**：`axes` 以 **inner-first** 排列；`z.shape == tuple(len(ax) for ax in reversed(axes))`，即 **inner 軸恆為 z 的最後一維**。1D `(Nx,)`、2D `(Ny, Nx)`、N 維 `(…, Ny, Nx)`。**load 是 save 的恒等逆**——任何一邊都**不做 caller-side transpose**。`.T` 與「transpose back」在 caller 中絕跡。

3. **per-experiment typed axes-spec**（把實驗資料持久化與 save/load 樣板合為一個 deep module）：每個實驗宣告一份 **typed axes-spec**——軸 `name`/`unit`/`scale`/順序、z channel、哪些是離散狀態軸、SI-on-disk 的單位轉換——由**一個共用 helper** 依此驅動原生 N 維 save/load。axis-order + unit + N 維不變式**只住一處**；per-experiment 的 save/load 樣板消失，實驗只留宣告式 spec。`Axis` / `LabberPayload` / `LabberMetadata` 是資料持久化模型，屬於 `zcu_tools.utils.datasaver`；`AxesSpec` / `ZSpec` 是 experiment Result / Cfg 到資料檔案的 mapping spec，留在 `zcu_tools.experiment.axes_spec`，避免 persistence facade 反向依賴 experiment 概念。grouped extension 遵守同一層次：`zcu_tools.utils.datasaver` / migration tooling 可暴露 generic `GroupedLabberData`（`DatasetRole -> LabberPayload` + one `LabberMetadata`），但 experiment load / analyze path 必須立即轉成 per-experiment typed Result，不把 raw dict 當作 experiment API。

4. **Grouped Experiment Dataset 用單一 Experiment Data File 表達**。一個 Experiment Result 只有一個 canonical Experiment Data File。若 Experiment Result 需要多個 Dataset Role，寫入同一個 `.hdf5` 中的多個 Labber log group。group 沒有 primary member；每個 Dataset Role 都是 peer，並共享同一個 Experiment Result Identity。每個 Dataset Role 可有自己的 axes、z label 與 shape；完整性由該 experiment 的 required roles 定義，不要求所有 role 共享 shape。on-disk layout 保留 Labber 原生 group naming：第一個 role 寫在 root log，後續 role 寫在 `Log_2` / `Log_3` / ...；role identity 不進 group name，也不進 z channel name，而是以 attrs 表達：file/root attr `zcu_tools.grouped_dataset_version = 1`、file/root attr `zcu_tools.dataset_roles = [...]`、每個 log group attr `zcu_tools.dataset_role = "<role>"`。

5. **Dataset Role 不是全域 enum，也不折成額外 numeric axis**。Dataset Role 是語意角色，不是 sweep dimension。role namespace 由每個 experiment 的 grouped spec 擁有；底層使用 value object / newtype string 表達 role，驗證 lowercase snake_case、required roles 與 group membership，但不建立全域 enum。role-as-axis 會強迫各 role 使用相同 axes / shape / z label，與 readout calibration、CPMG phase pair 等 result model 不符。多 Labber log group 保留每個 role 的 Labber Dataset 語意，也讓 load 階段能以 role 做 strict validation。

6. **Data payload 與 metadata 分離**。`LabberPayload` 表達單一 Labber Dataset 的 data channel、axes 與 shape-dependent per-entry timestamps；`LabberMetadata` 表達 Experiment Data File metadata（comment / tags / project / user / creation time 等）。single-file `LabberData` 是 `payload + metadata` 的組合。`GroupedLabberData` 是 `DatasetRole -> LabberPayload` 加上一份 shared `LabberMetadata`，不內嵌多個各自帶 metadata 的 `LabberData`，因此 metadata consistency by construction。

7. **Grouped save 接 role payload mapping，不接 required roles**。`save_grouped_labber_data(path, roles={...}, metadata=...)` 接一份完整 `DatasetRole -> LabberPayload` mapping 與一份 shared `LabberMetadata`；role value 不接受 `LabberData`，caller 若手上是 single `LabberData` 必須明確傳 `data.payload`，避免 saver 隱式丟棄 member metadata。low-level saver 驗證 role 格式、至少一個 role、每個 `LabberPayload` 自身 shape，並把 role list 寫入 attrs。required roles 是 experiment grouped spec 的 completeness policy；experiment save helper 在呼叫 low-level saver 前驗證 Result 是否產生完整 roles。

8. **Legacy Measurement Artifact 只透過 migration script 處理**。normal load / analyze 只接受 Complete Experiment Result；runtime 不保留 `.npz` 或舊多檔案格式的 compatibility path。migration script 以 `--experiment` 指定白名單 converter id，不以 reflection 任意 import experiment class；converter registry 留在 script 內部或 script-local private helper，不放進 `lib/zcu_tools` 作為 runtime API；input read-only；output 預設不得存在，只有明確 `--overwrite` 才可覆蓋；script 寫入 temp file 後，必須驗證可載入 Complete Experiment Result，再 atomic rename 到 output。

9. **Save path ownership 屬於 caller / runner**。persistence layer 寫入 caller 指定的 exact path，不 silently rename。`safe_labber_filepath` 保留為 caller 在保留 final path 時使用的 helper，不在 lower-level writer 裡改路徑。single-role Labber Dataset 與 Grouped Experiment Dataset 使用同一套 path ownership 規則。

10. **Out of scope：autofluxdep / overnight**。`autofluxdep` 與 `overnight` 的 output 是跨 task / 長時間 workflow 的 heterogeneous measurement collection，不等同於單一 Experiment Result 的 Dataset Roles。本 ADR 的 grouped dataset 延伸先落在單一 experiment result，例如 readout calibration 與 `CPMGExp`。

11. **強型別 + Fast-Fail**。原生 `Channel` 是 namedtuple、`LabberData.z`/`.x` 為 `Any`；axes-spec 層補上型別化包裝（frozen `Axis` 與型別化結果），save 時驗 `z.shape` 與 axes 不符即 **raise**（不靜默 transpose）。`load_grouped_labber_data(path, required_roles=(...))` 是 normal strict path：缺 role、多未知 role、duplicate role 都 raise。省略 `required_roles` 只供 tooling / diagnostic / migration 使用，回傳檔案內所有 roles，但仍驗證 role 格式與 duplicate。experiment loader 一律傳 `required_roles`，不能 implicit load whatever is there。

12. **`load_labber_data` 與 grouped loader 分流**。`load_labber_data` 保持 single `LabberData` 語意，對既有非 grouped 檔案的行為不變，且回傳值包含 payload（data / axes / per-entry timestamps）與 metadata（comment / tags / project / user / creation time 等）；不另設只讀 metadata 的 public loader。只有檔案帶有 `zcu_tools.grouped_dataset_version` marker 時 fast-fail，錯誤訊息指向 `load_grouped_labber_data`。grouped file 不回傳第一個 role，也不沿用既有 multi-log stacking 行為。

13. **遷移 = 增量分批**（非 big-bang）。已完成刪 load-flip（`datasaver.py:173-179`）+ 修 power_dep latent bug + 改 `test_datasaver` 的 2D 斷言成 round-trip 恒等；新 axes-spec 介面就位後，caller 檔分批 phase 遷移。grouped dataset extension 先落在 low-level grouped persistence + tests，釘住 single-file multi-log-group round-trip、不同 role 可有不同 axes / shape、strict required roles fast-fail、diagnostic load explicit opt-in、duplicate / invalid role fast-fail、exact path write 不 silently rename。`CPMGExp` grouped roles 固定為 `lengths` 與 `signals`，new save 只寫單一 grouped `.hdf5` Experiment Data File，不再寫 `.npz` 或 `*_length` / `*_signals` side files。CPMG grouped payload 使用 Result 原生方向，不沿用 legacy side file 的 `.T`：axes inner-first 為 `Time Index`、`Number of Pi`，`lengths` / `signals` z shape 均為 `(Ntime, Nlength)`；`lengths` 仍以 SI seconds 寫入。migration input 固定為 legacy `.npz`，不從 legacy side files 拼回來；`script/migrate_experiment_data.py --experiment twotone/cpmg --input old.npz --output new.hdf5 [--overwrite]` 以白名單 converter registry 執行，input read-only、output 先寫同目錄 temp file，驗證完整 grouped result 可 load 後再 atomic rename。`RabiCheckExp`、`CKP_Exp` 與 bath reset freq-gain slice 證明固定離散軸應進 Result model：`reset_states` / `initial_states` / `phases` 是普通 axis 欄位，不是 Dataset Role。`script/migrate_experiment_data.py --experiment twotone/ckp --input old_base --output new.hdf5 [--overwrite]` 從 legacy `<base>_ground` / `<base>_excited` sidecar 轉成 canonical single-role 3D HDF5。

14. **後續 single-role migration 固定採 Result-native disk order，不擴 `ZSpec` z-transform**。尚未遷移的 MIST / T1 / GE 類實驗若現有 legacy save 透過 `.T` 或拆 sidecar 方便 Labber 瀏覽，新 canonical file 仍以 Result dataclass 的原生 shape 寫入，並選擇 inner-first axes 使 `z.shape == tuple(len(ax) for ax in reversed(axes))`。`ZSpec` 不加入 per-experiment transpose / z-transform hook；需要保留舊 artifacts 時，由 migration script 在 legacy input 到 canonical output 的邊界做重排，不把 legacy orientation 帶進 runtime save/load。

15. **population / state / phase 是 axis，除非資料角色異質**。GE population、prepared state、initial state、tomography phase 這類離散狀態若共享同一 z label / unit / dtype，建模為 single-role N-D dataset 的 axis，不拆成 grouped roles。對只持久化 `g/e` 兩個 population components 的 Result，canonical file 只存 Result 實際欄位中的 `g/e` components；`other = 1 - g - e` 仍是 analysis/display derived value，不成為隱式 persisted channel。`T1WithToneSweepExp` legacy load 的 zero-filled `other` component 是 Legacy Measurement Artifact 行為，不進新 canonical runtime path。

16. **auto-optimize grouped roles 採 typed role split，不延續 mixed-unit `params` role**。`twotone/ro_optimize/auto_optimize.py::AutoOptExp` 的 grouped canonical roles 固定為 `readout_freq`（disk Hz、Result `params[:, 0]` memory MHz）、`readout_gain`（a.u.）、`readout_length`（disk s、memory us）、`snr`（a.u.）。`jpa/jpa_auto_optimize.py::AutoOptimizeExp` 的 grouped canonical roles 固定為 `jpa_flux`（A，identity）、`jpa_freq`（disk Hz、memory MHz）、`jpa_power`（dBm）、`jpa_phase`（integer phase index）、`snr`（a.u.）。typed Result boundary 仍可重建既有 `params` arrays 供 analyze 使用；mixed-unit `params` 不作為 grouped Dataset Role。

17. **每個移除 legacy runtime persistence 的 experiment 都提供 migration converter**。normal runtime 一律不讀 legacy sidecar / `.npz`，但 feature branch 內凡是把既有 hand-written `save/load` 或多 sidecar runtime path 改成 canonical Experiment Data File 的實驗，都要在 `script/migrate_experiment_data.py` 加白名單 converter。converter id 以 experiment path 命名、讀取明確 legacy inputs、輸出 canonical Experiment Data File，並在 temp file 驗證 typed load 成功後 atomic rename。converter 是可移除的離線工具，不是 runtime compatibility path。

18. **converter coverage 跟 migration slice 同步擴張**。已落地 `twotone/cpmg` 與 `twotone/ckp`；後續 single-role cleanup 需要 converter 包含 `twotone/reset/bath/freq`、`twotone/reset/bath/length`、`singleshot/ge`、`singleshot/len_rabi`、`singleshot/mist/power`、`singleshot/mist/freq`、`singleshot/mist/pre_freq`、`singleshot/mist/power_freq`、`singleshot/ac_stark`、`singleshot/t1/t1`、`singleshot/t1/t1_with_tone`、`singleshot/t1/t1_with_tone_sweep`。grouped cleanup 需要 converter 包含 `twotone/ro_optimize/auto_optimize` 與 `jpa/jpa_auto_optimize`。converter 可與該 experiment migration 同一 commit 落地，測試要覆蓋 legacy input → canonical output → typed load。

## 理由 / 取捨

- **Deletion test**：刪掉 datasaver dict 殼後，複雜度不會散到 38 個 caller，而是**集中**到 `zcu_tools.utils.datasaver` 一個有型別的 facade + 一份共用 axes-spec helper——這是「淺殼變深 module」的 deepening，而非 pass-through 搬家。
- **軸序只住一處** → 根除「每 caller 自決朝向」的漂移類 bug（含已存在、未測的 power_dep load 斷言錯誤）。新實驗無法再各自引入不一致。
- **不加 z-transform hook**：表面上能保留舊 Labber 瀏覽 orientation，但會讓每個 experiment 又能私下定義一套 shape 轉換，重建本 ADR 要移除的 caller-side transpose 分歧。Result-native disk order 讓 save/load 不變式單純且可測。
- **datasaver package 不是舊 datasaver API 復活**：package name 收斂 public imports，但舊 dict API 不回來；內部 Labber on-disk engine 仍是 inner-first / N 維 / 軸序權威，只是從 public `labber_io.py` 收進 facade 內部子模組。
- **不保留 import shim**：repo 內所有 caller 在 feature branch 一次改到 `zcu_tools.utils.datasaver`。留下 `zcu_tools.utils.labber_io` shim 會讓兩個 public import path 長期並存，與「唯一 public facade」相矛盾。
- **Axis / AxesSpec 分層**：`Axis` / `LabberPayload` / `LabberMetadata` 描述 persisted data model，可被 low-level saver、grouped saver、tests 與 tooling 共用；`AxesSpec` 描述某個 experiment Result dataclass 如何映射到 persisted data，屬 experiment boundary。把 `AxesSpec` 放進 datasaver 會讓 utils package 依賴 experiment-level concepts。
- **metadata loader 不另立 public API**：comment / tags / project / user / creation time 屬於 Experiment Data File metadata，`load_labber_data` 的回傳模型已承載；另設 `load_comment` 類 API 會讓 caller 繞過完整資料模型，重建 partial-loading 習慣。
- **metadata consistency by construction**：single `LabberData` 為 payload + metadata 的 convenience object；grouped model 只允許一份 shared metadata 與多個 role payloads，不存在 member metadata 與 group metadata 不一致的狀態。
- **grouped save 不自動剝 LabberData**：若 low-level saver 接受 `LabberData` role value 並自動取 payload，member metadata 會被 silent discard；要求 caller 明確傳 `data.payload` 讓 metadata 丟棄成為可見決策。
- **Grouped Experiment Dataset 取代多檔案 workaround**：多檔案把一個 Experiment Result 拆成 sidecar artifacts，容易讓 path reservation、identity、load completeness 與 analysis handoff 分裂；單一 Experiment Data File 才是 canonical result。
- **role metadata 放 attrs，不改 Labber group naming**：root log 沒有 role-style group name 可用，且 `Log_2` / `Log_3` 是 Labber 既有多 log 表示。把 role 寫進 attrs 可保留 Labber compatibility，又讓 grouped reader 以 explicit metadata 做 strict validation。
- **global role enum 被拒絕**：`ground`、`excited`、`phase_max` 等名稱可能跨 experiment 重用，但語意與 required set 屬於各 experiment result。全域 enum 會製造假的 taxonomy；per-experiment spec 才是正確 namespace。
- **role-as-axis 被拒絕**：Dataset Role 是語意分類，不是 numeric sweep axis；強行折成 axis 會讓不同 axes / shape / z label 的 role 失去自然表示。
- **axis-as-role 也被拒絕**：prepared state、initial state、phase、population component 若只是同一測量值的離散維度，把它拆成 sidecars 或 grouped roles 會把一個 homogeneous Result 人為切碎。這類資料應收斂成 single-role N-D dataset。
- **low-level grouped save 不收 required roles**：low-level persistence primitive 只知道它收到哪些 roles，不知道某個 experiment result 的完整性政策。required roles 留在 experiment grouped spec，避免把 domain completeness 混進 HDF5 writer。
- **runtime legacy compatibility 被拒絕**：舊 `.npz` 或舊多檔案格式是 Legacy Measurement Artifact，不是正常 loading format；migration script 明確、可移除，也避免長期污染 experiment load path。
- **converter 不等於 runtime compatibility**：每個 migrated experiment 都補 converter 是為了保存既有 artifacts 的離線升級路徑；normal experiment `load()` 仍只接受 canonical Experiment Data File。converter registry 是 script-local 白名單，所以未來可整段移除，不污染 runtime API。
- **N 維 folding 不解決 grouped semantics**：`LivePlotData` 與 `save_labber_data` 支援 N 維 data，但 grouped result 的核心問題是多個 Dataset Role 的 identity 與 completeness，不是單一 array 的 rank。
- **generic container 不外洩到 experiment API**：generic `GroupedLabberData` 對 migration、diagnostic、low-level round-trip 測試有用；normal experiment path 若回傳 raw dict，required roles、shape、unit 與 analysis contract 會退回 runtime convention。typed Result 讓完整性與欄位語意在 experiment boundary fast-fail。
- **single loader 不誤讀 grouped file**：`load_labber_data` 已有多個 production callers，不能破壞既有非 grouped load；但 grouped marker 代表一個 Experiment Result 需要多個 Dataset Role 才完整，single loader 若回傳其中一個 role 或 stack roles 都會產生 silent partial result。
- **跨模組**（`utils/` + `experiment/`、影響約 75 caller）→ 立 ADR 而非僅模組 README；模組局部速查另補 `lib/zcu_tools/utils/README.md`。
- **協調風險**：`lib/zcu_tools/utils` 與多個 `experiment/` 路徑當前被另一 session（`codex-wavelet-signal2real`）持 write claim → **實作 phase 須待其釋出或先協調**；本 ADR、慣例釘樁與規劃不受此阻擋，可先行。
