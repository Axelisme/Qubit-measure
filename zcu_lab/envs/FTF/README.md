# FTF entry kickstart

此腳本只建立或重開 FTF entry，不連線 SoC 或 device，也不載入模擬環境。
在 repo root 執行 `uv run --directory . --no-sync -- python scripts/run_measure_gui.py --script zcu_lab/envs/FTF/startup.py`，或在 MCP 呼叫 `connect(launch="new", script="/absolute/path/to/zcu_lab/envs/FTF/startup.py")`。
首次執行會按相鄰的 `entry-template.yaml` 建立 Q1、Q2、coupler、R1、R2、MG12 的 setup 參數，建立並選取 `initial` point，再保存該 point 的 aliases。
FTF 已存在時只重開並選取 `initial`，不覆寫元件或 aliases；缺少該 point 或 entry 不完整時會報錯，已完成的操作不會自動回滾。
範本不含 module library 或校準完成的量測參數，量測前須自行確認接線、填入有效參數並連線所需硬體。
