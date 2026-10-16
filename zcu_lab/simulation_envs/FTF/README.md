# 模擬環境

**Last updated:** 2026-10-17

這裡放 GUI 與 Notebook 共用的可信 Python 環境。`environment.py` 定義無參數的 `build_environment()`，讀取相鄰 `ftf.yaml` 並回傳已組裝的 FTF 環境。

`default.yaml` 是納入 Git 的樣板；`ftf.yaml` 是各自的工作副本，不納入 Git。第一次使用時複製一份：

```bash
cp zcu_lab/simulation_envs/FTF/default.yaml zcu_lab/simulation_envs/FTF/ftf.yaml
```

缺少 `ftf.yaml` 時，`build_environment()` 直接報錯。

## 啟動與調參

在 repo root 執行：

```bash
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --simulate zcu_lab/simulation_envs/FTF/environment.py
```

`--simulate` 必須接 Python 檔路徑，不直接接受 YAML。環境檔會執行 Python，只載入可信來源。修改環境或 YAML 後須重啟 GUI；載入失敗會中止，不回退正常模式。

`default.yaml` 是示意的 F1–coupler–F2 樣品，包含模型參數、channel 數量與接線。它不是校準過的晶片，也不保證截斷收斂。更換樣品時可複製環境檔與 YAML，並調整 builder 指向的檔案。

[Notebook 範例](../../../notebook_md/single_qubit.md) 使用同一個環境。Builder 成功回傳後，caller 負責關閉環境的 device manager。完整組裝與數值限制見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md)。

## 演化與重播

FTF 預設使用分段引擎的快速算法。模型與 prepare 快取只保留在記憶體，不需要離線建庫或 GPU，也不保存 exact density records。

Builder 可用 `ftf(..., private_replay=absolute_directory)` 保存私有輸入。`replay_round(round_path)` 在相同數值、RNG 與相依套件版本下重算一個 round，不讀取演化資料庫。私有檔案含模型真值，不放入一般 Run metadata。
