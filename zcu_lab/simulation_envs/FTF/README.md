# 模擬環境

**Last updated:** 2026-10-16

這裡放 GUI 與 Notebook 共用的可信 Python 環境。`environment.py` 定義無參數的 `build_environment()`，讀取相鄰 `ftf.yaml` 並回傳已組裝的 FTF 環境。

## 啟動與調參

在 repo root 執行：

```bash
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --simulate zcu_lab/simulation_envs/environment.py
```

`--simulate` 必須接 Python 檔路徑，不直接接受 YAML。環境檔會執行 Python，只載入可信來源。修改環境或 YAML 後須重啟 GUI；載入失敗會中止，不回退正常模式。

`ftf.yaml` 是示意的 F1–coupler–F2 樣品，包含模型參數、channel 數量與接線。它不是校準過的晶片，也不保證截斷收斂。更換樣品時可複製環境檔與 YAML，並調整 builder 指向的檔案及數值資料路徑。

[Notebook 範例](../../notebook_md/single_qubit.md) 使用同一個環境。Builder 成功回傳後，caller 負責關閉環境的 device manager。完整組裝與數值限制見 [FTF 契約](../../lib/zcu_tools/experiment/ftf/README.md)。

## 數值資料與離線建庫

範例把數值資料放在相鄰的 `evolution_data/`，此目錄不納入 Git。執行期可直接求解並保存 exact records，不要求先建插值表。

`build_database.py` 使用同一份 YAML 與數值資料目錄。它把參數交給可 import 的 `experiment.ftf.database.main(argv)`，不自行實作建庫。離線建庫需要 Intel OpenCL 與 `gpu` extra，必須明確提供 duration、ratios 與 scale 範圍；參數見：

```bash
uv run --directory . --no-sync -- python zcu_lab/simulation_envs/build_database.py --help
```
