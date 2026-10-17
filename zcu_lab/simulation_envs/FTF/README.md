# 模擬環境

**Last updated:** 2026-10-17，v2 欄位指南

這裡放 GUI 與 Notebook 共用的可信 Python 環境。`environment.py` 定義無參數的 `build_environment()`，讀取相鄰 `ftf.yaml` 並回傳已組裝的 FTF 環境。

`default.yaml` 是納入 Git 的樣板；`ftf.yaml` 是各自的工作副本，不納入 Git。第一次使用時複製一份：

```bash
cp zcu_lab/simulation_envs/FTF/default.yaml zcu_lab/simulation_envs/FTF/ftf.yaml
```

缺少 `ftf.yaml` 時，`build_environment()` 直接報錯。完整欄位、單位、合法範圍、三迴路映射與 entry 接線對照見 [FTF YAML v2 欄位指南](field-guide.md)。

## 建立 entry 與啟動

在 repo root 執行一次 entry script，再啟動 GUI：

```bash
uv run --directory . --no-sync -- python scripts/create_entry.py \
  zcu_lab/simulation_envs/FTF/entry-template.yaml ftf-demo
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --simulate zcu_lab/simulation_envs/FTF/environment.py
```

Script 建立 `result/ftf-demo/setup.yaml`、`aliases.yaml`、第一個 point `initial` 及 `Database/ftf-demo/`。任一 entry 目錄已存在就報錯，不覆寫。預設 roots 跟 measure GUI 一樣，以 script 所在專案為準，不跟隨 shell cwd。`--project-root PATH` 可指定別的專案，`--point-label NAME` 可改第一個 point 的名稱。後續磁碟寫入失敗可能留下部分新 entry，script 不提供跨檔交易。

在 GUI 的 Setup 選 Open，開啟 `ftf-demo`，再選取 point `initial`。Open 只裝載 setup 與 aliases，不會自動選 point。Q1、Q2、coupler、R1、R2 已在 setup 與初始 point 中。Environment 另外建立 f1_bias、coupler_bias、f2_bias 三個 live device，不把 driver 存進 entry。

`entry-template.yaml` 使用 `default.yaml` 的 EJ/EC/EL 與 resonator 頻率。Energies 用 GHz，R1/R2 頻率為 7190/7080 MHz。Qubit 頻率隨 bias 改變，範本保留 null，不推算頻譜。範本提供接線與引用，不提供 module library 或校準完成的量測參數。若修改工作副本的模型或接線，也要調整 entry 範本或已建立的 entry。

初始 alias `qubit` 指向 Q1，`resonator` 與 `flux_dev` 分別讀取其 resonator 與 flux_source。把 `qubit` 改為 Q2 後，兩個連結會跟著切換。`control`、`target` 固定為 Q1、Q2，`coupler` 是普通 alias，指向 qubit/transmon 元件，不是 Notebook role。Inspect > Aliases > Load preset 可選 FTF，再按 Apply 保存。載入 preset 只替換 draft，不會建立元件或 device。

## 工作副本與調參

`default.yaml` 使用 `format: zcu.ftf`、整數 `version: 2`。V1 文件不能載入，不自動轉換既有工作副本，也不自動更新既有 entry。三個 DC 電源與校準位於 `dc_bias.sources`，樣板使用 A。環境每 round 讀取 live DC snapshot，保留靜態磁通掃描。Charge microwave drive 位於 `drive_lines`，不是 flux drive。

FTF 不建構 flux lines 或 flux controls，不提供快速 flux pulse；`flux_lines` 即使為空 map 也按未知欄位拒絕。未綁定 generator 沿用既有處理，沒有額外 flux-pulse 專用報錯。共享 `simulate` 引擎仍保留 flux-pulse 能力，但 FTF 不使用。採用新樣板時請自行保留或重新填入需要的設定，不覆寫尚未保存的本機調整。舊工作副本或預覽若仍含 `flux_lines`，須自行移除。

`--simulate` 必須接 Python 檔路徑，不直接接受 YAML。環境檔會執行 Python，只載入可信來源。修改環境或 YAML 後須重啟 GUI；載入失敗會中止，不回退正常模式。

`default.yaml` 的 circuit energies、couplings 與 resonator frequencies 採論文 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。Channel 數量由最大索引加一推導，不是綁定條數；樣板保留 readout gen0 與 microwave gen2，gen1 是未綁定空洞。完整訊號路徑與接線表見 [欄位指南](field-guide.md)。更換樣品時可複製環境檔與 YAML，並調整 builder 指向的檔案。

[Notebook 範例](../../../notebook_md/single_qubit.md) 使用同一個環境。Builder 成功回傳後，caller 負責關閉環境的 device manager。完整組裝與數值限制見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md)。

## 演化與重播

FTF 預設使用分段引擎的快速算法。模型與 prepare 快取只保留在記憶體，不需要離線建庫或 GPU，也不保存 exact density records。

Builder 可用 `ftf(..., private_replay=absolute_directory)` 保存私有輸入。`replay_round(round_path)` 在相同數值、RNG 與相依套件版本下重算一個 round，不讀取演化資料庫。私有檔案含模型真值，不放入一般 Run metadata。
