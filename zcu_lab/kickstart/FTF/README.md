# FTF entry kickstart

**讀者：** lab-author、agent。用 startup 腳本建立或重開 FTF entry 與 initial point。

此腳本只建立或重開 FTF entry，不連線 SoC 或 device，也不載入模擬環境。
在 repo root 執行 `uv run --directory . --no-sync -- python scripts/run_measure_gui.py --script zcu_lab/kickstart/FTF/startup.py`，或在 MCP 呼叫 `connect(launch="new", script="/absolute/path/to/zcu_lab/kickstart/FTF/startup.py")`。
首次執行會按相鄰的 `entry-template.yaml` 建立 Q1、Q2、coupler、R1、R2、MG12 的 setup 參數，建立並選取 `initial` point，再保存該 point 的 aliases。
FTF 已存在時只重開並選取 `initial`，不覆寫元件或 aliases；缺少該 point 或 entry 不完整時會報錯，已完成的操作不會自動回滾。
Kickstart 範本預填示意 qubit energies 與 R1／R2 頻率 7190／7080 MHz；模擬用 entry 範本則省略這些值，由 FTF YAML 定義物理模型。兩份範本都不含 module library 或校準完成的量測參數。使用 kickstart 量測前須自行確認接線、填入有效參數並連線所需硬體。
R1／R2 wiring 保留 pulse ch 0／ro_ch 0。MG12 另填 mux `wiring.gen_ch: 1`、`wiring.ro_ch: [1, 2]`，依序對應 R1／R2。八張 mux 表單缺 readout slot 時，Open／Reset 從 group wiring 補通道，tone gain 補 0.1；頻率與 timing 等其餘必填值仍須確認。Qubit `freq` 保留 null，待量測後再填。若要使用校準過的完整 readout，可按[接線步驟](../../simulation_envs/FTF/field-guide.md#generatoradc-與-entry-接線對照)建立 library 模組並綁定 `MG12.module.readout`。
