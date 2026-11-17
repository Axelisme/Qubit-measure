# Frozen Engine scenes

**讀者：** framework-dev。凍結輸入的 owner、範圍與讀寫契約。

## Owner 與用途

Pareto 集合包含 1 個重新宣告的 synthetic 場景、15 個完整 real 場景、3 個單段 kernel 場景與 14 個常數雙 tone 價格控制。
`frozen-synthetic.index.json` 與 `frozen-real.index.json` 依 category 保存這些身份，也可容納其他已宣告的場景。
兩份 version 1 清單只存每筆 name、physical_sha256、oracle_sha256 與 metadata_sha256。
Metadata hash 使用公開 `simulate.evolution.digest(record.metadata)`。
數值不入庫。每筆 gzip 放在 `<git common dir 上層>/.agent_state/benchmark/family_pareto/scenes/v1`。
檔名含 name、physical hash 前 16 碼及 metadata hash 前 16 碼。Task 清理不得刪除此共用快取。
首次從既有兩份 gzip 遷移 19 筆，逐筆核三個 hash、point order 與 reference key 不變，沒有跑 builders 或 solver。
落地的歷史不含兩個舊 gzip。冷快取完整重建尚未驗證，需另取運算授權。
公開 `family_pareto.select_inputs(index, scenario, *, scene_cache=...)` 集中來源與 report 身份。
未指定 index 或 scenario 時，先讀這兩份清單，只選 `select_inputs` 旁宣告的 33 個 Pareto 名稱。
缺少任何預設名稱就失敗，不讀其他列的 payload 或 reference。
`--index PATH` 不帶 scenario 時全選該份清單，`--scenario NAME` 可嚴格單選非 Pareto 列。
Profile 列不作 family 去留證據。`--scene-cache DIR` 接受絕對快取路徑。
跨兩份清單重名一律失敗，未選 scenario 也相同。缺檔列名稱、hash 與 producer 指令，不補建。
接縫只讀已凍結數值，不重建裝置、前端或使用者設定。
量測只在 `Engine.evolve` 上進行。

固定 Pareto 場景控制由 `tools/family_pareto_scenarios.py`、`family_pareto_kernel_scenes.py`、`family_pareto_multitone_scenes.py` 與固定 settings 持有。
`tools/dop853_segment.py` 持有 profile 座標與座標衍生名稱，透過 `profile_inputs()` 提供宣告。
Producer 預設 inventory 包含這些 profile 宣告，但固定 Pareto 預設名稱集合不變。
宣告與單列路由不會填入 index 或共用 cache。取得計算授權後，須顯式執行 producer 準備。
Single/integer 編譯與數值 source 的生命週期由 `family_pareto_scene_builders.py` 持有。
重產用法見 [tools README](../../README.md#freeze-numeric-scenes)，模型建構與求解須分別取得計算成本授權。
這份資料不包含參考解，不表示所有 reference cache 都已準備好。

## 場景範圍

- Synthetic：`idle_coherence`，3 能階、2 個點。其餘 accuracy、guard、history 與放大 workload 留在 tests 側，不當作工具場景。
- 常數雙 tone 價格控制：`mtrwa_c2b_l16_128`、`mtrwa_c2b_l20_128`、`mtrwa_c2b_l16_512` 是 INTEGER_MODEL 的 leading energy-basis projection。Pump/Stokes 頻率為 E10+0.5/0.01 GHz，DAC 量化前 gain 為 0.02/1e-4。它們各有一個同起同止的支援區間，沒有 frontend gap 或 readout lead。
- `mtrwa_star_*` 用固定 M=16 的 synthetic 三能階 blocks。Base 有 2 個 active blocks、gain 0.16、restored detuning 0.08 GHz、duration 128 ns。每個 low/high 只改 action、retained dimension、restored bandwidth 或 duration 之一；其餘參數保持 base。九列的實際 retained D 為 22/28/40，restored B_eff 為 0.075/0.08/0.1 GHz，duration 為 32/128/512 ns。模型不是日常 FTF 輸入的替身。
- `mtrwa_star_n20`、`mtrwa_star_n100` 接續相同 base support，以 neutral boundaries 保留 20/100 個段。每段保留全域相位。RWA prepare key 含 absolute start，因此這些段不宣稱共用同一個 prepare key。
- 完整 F/L/I/N：`carrier_sweep`、`length_sweep`、`idle_sweep`、`gate_sequence`，各 24 點、21 能階。保留編譯後的 gaps、readout lead、point order、控制相位和 companion 宣告。
- 其餘完整 real 場景包含 gain scan、joint simultaneous drive、constant/shaped tone overlap，以及 biased、long、random gate 序列。它們不是四組 F/L/I/N 的替代品。
- 單段 kernel：`kernel_single_constant`、`kernel_single_sampled`、`kernel_multitone_sampled`，各 4 點、21 能階。它們使用公開 `SingleFluxoniumModel`，每個點只有一個同時支援區間，不從完整序列裁切。

Kernel 模型為 EJ/EC/EL = 4/1/1 GHz、DC flux = 0.5 Φ₀、cutoff 40、temperature 0.03 K，關閉 loss。
四個 gain 是 0.01、0.04、0.07、0.1，support 為 100 ns，charge coefficient 為 0.1 GHz/gain。
Sampled envelope 使用 9 個 Gaussian knots、sigma 20 ns。
Multitone 在同一個 port 加上 1.25 倍 DC E1−E0 的第二個 carrier。

## 數值與容器契約

`simulate.evolution.read_frozen_index(path)` 讀回有序 `FrozenIndexEntry`。
`read_scene_records(cache, entries)` 依清單讀取單筆檔，核對 version、name 與完整三個 hash。
單筆檔沿用 `write_frozen_manifest`／`read_frozen_manifest`，只容納一個 `FrozenRecord`。
`write_scene_record` 與 `write_frozen_index` 用同目錄暫存檔加 replace 原子發布。
`read_physical_input(record.physical)` 建立 caller-owned source 和 batch，使用完畢要關閉 source。
Codec 用 canonical JSON 和 stdlib gzip，mtime 固定為 0，不寫檔名。
同輸入寫到不同路徑仍得到相同 bytes。
未壓縮的舊容器不入庫。

`physical_sha256` 包含 model identity namespace。
`oracle_sha256` 只排除該 namespace。
`reference_key` 另加入既有兩組 tolerance、version 1 與 convergence limit 2e-6，不加入 solver identity。
完整 API 見 [evolution README](../../../lib/zcu_tools/simulate/evolution/README.md#frozen-numeric-inputs-and-references)。

場景 reason、expected families、companions、routes 與 provenance 放在 `record.metadata`，只有 tools 解讀。
原有 real 場景的業務 `metadata` 欄位仍位於這個 opaque object 內，沒有展平或刪除。
Tests 側 producer 只讀 record name 和 physical。呼叫端以 `--converged` 指定 synthetic index，以 `--interaction` 指定 real index，另傳顯式 `--scene-cache`。
Prepare 重建並比對清單，hash 不符就停。只有明確 `--update-index` 會在所有選定建構成功後改所選列。
已在 inventory 宣告、兩份清單皆無的新名稱，須以 `--update-index --scenario NEW` 建構並寫快取後，才加到該 category 清單尾端。
既有列更新保留其他列與順序。完整 prepare 仍嚴格核對名稱集合，不同步新增或刪列。
Tests reference producer 的 `--scenario NAME` 在明確清單中要求恰好一列，選取後才讀 payload 與求解。
刪列或換組須明確編輯 tracked builder 與清單，不從名稱缺失自動推導。
冷快取重建尚未核對，不以首次 payload 遷移宣稱全部 builders 可逐位元重產。
它不讀 metadata、不靠名稱清單選 solver，也不在失敗後換 solver。
參考解準備與 cache-only 讀取的分工見 [tests README](../../../tests/README.md#evolution-精度矩陣) 與 [ADR-0078](../../../docs/adr/0078-evolution-module-boundary.md)。

Native thread 設定會改模型的最後幾位浮點，因此改 physical hash 與 reference key。快取遺失且重建不能逐位元重現時，只能明確 `--update-index`，並重算受影響的參考解。
