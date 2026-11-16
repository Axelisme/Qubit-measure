# Frozen Engine scenes

**讀者：** framework-dev。凍結輸入的 owner、範圍與讀寫契約。

## Owner 與用途

`frozen-synthetic.index.json` 釘住 1 個重新宣告的 synthetic 場景。
`frozen-real.index.json` 釘住 15 個完整 real 場景與 3 個單段 kernel 場景。
兩份 version 1 清單只存每筆 name、physical_sha256、oracle_sha256 與 metadata_sha256。
Metadata hash 使用公開 `simulate.evolution.digest(record.metadata)`。
數值不入庫。每筆 gzip 放在 `<git common dir 上層>/.agent_state/benchmark/family_pareto/scenes/v1`。
檔名含 name、physical hash 前 16 碼及 metadata hash 前 16 碼。Task 清理不得刪除此共用快取。
首次從既有兩份 gzip 遷移 19 筆，逐筆核三個 hash、point order 與 reference key 不變，沒有跑 builders 或 solver。
Tip 刪檔不等於歷史清除。歷史 blob 清理與冷快取完整重建驗證是不同的後續義務。
公開 `family_pareto.select_inputs(index, scenario, *, scene_cache=...)` 集中來源與 report 身份。
預設讀這兩份清單，`--index PATH` 只選一份，`--scene-cache DIR` 接受絕對快取路徑。
跨兩份清單重名一律失敗，未選 scenario 也相同。缺檔列名稱、hash 與 producer 指令，不補建。
接縫只讀已凍結數值，不重建裝置、前端或使用者設定。
量測只在 `Engine.evolve` 上進行。

場景控制由 `tools/family_pareto_scenarios.py`、`family_pareto_kernel_scenes.py` 與固定 settings 持有。
Single/integer 編譯與數值 source 的生命週期由 `family_pareto_scene_builders.py` 持有。
重產用法見 [tools README](../../README.md#freeze-numeric-scenes)，模型建構與求解須分別取得計算成本授權。
這份資料不包含參考解，不表示所有 reference cache 都已準備好。

## 場景範圍

- Synthetic：`idle_coherence`，3 能階、2 個點。其餘 accuracy、guard、history 與放大 workload 留在 tests 側，不當作工具場景。
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
刪列或換組須明確編輯 tracked builder 與清單，不從名稱缺失自動推導。
冷快取重建尚未核對，不以首次 payload 遷移宣稱全部 builders 可逐位元重產。
它不讀 metadata、不靠名稱清單選 solver，也不在失敗後換 solver。
參考解準備與 cache-only 讀取的分工見 [tests README](../../../tests/README.md#evolution-精度矩陣) 與 [ADR-0078](../../../docs/adr/0078-evolution-module-boundary.md)。
