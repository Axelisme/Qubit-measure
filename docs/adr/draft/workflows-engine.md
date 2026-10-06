---
status: draft
---

# 具名 workflows 與可丟棄的步驟執行

本篇是 workflows task 的期 0 候選，不代表 main 已採用。顧問 review 與使用者 CP-A 通過後才實作。既有 autofluxdep 與 executor 仍依 [[0062]] 運作，期 3 的 CP-C 通過前不退役。

## 問題

現行 autofluxdep 的 Node、cfg override、feedback 與共享結果陣列綁在一起。定義者修改量測 program、fit、參數生成與圖時，需要理解這些 owner 的交互作用。暫停在點邊界結束，不能丟棄當前點後用新參數重做。

overnight 也需要校準、單次實驗、間隔等待與存檔，但不需要 flux 外圈或 feedback registry。把兩者展開成通用圖會增加定義者必須理解的概念。

## 候選決策

### 模組與 owner

| Module | 責任 | 不負責 |
| --- | --- | --- |
| `experiment.workflows` | workflow 宣告、迭代、effect 執行、取消、deepcopy、revision、journal、manifest、迭代目錄 | flux、fit 接受度、校準策略、Qt、單次資料檔格式 |
| `zcu_lab.workflows` | 具體 workflow、experiment 函式、分析、完整 cfg 組裝、record 與 state | 宿主控制、通用檔案格式 |
| `gui.app.workflows` | controller、硬體 session、operation segment、表單、圖快照、RPC | 第二份 engine state、量測策略 |
| `mcp.workflows` | RPC client、工具與 image 投影 | engine、Figure、硬體 session |
| `datafile` 與 experiment persistence | 單次 cfg／Result 的格式、schema 與 writer | workflow 路徑、迭代提交 |

位置沿用 [模組定位](module-boundaries.md)。組合根注入使用者 catalog，不讓 framework import `zcu_lab`。

### Workflow 與 experiment

一個 workflow 是具名宣告加 `step(env, plan, tun, state)` generator。宣告附在原函式上，不寫入全域 registry。catalog 的 `register_all(registry)` 在宿主啟動或重載時明確登記。engine 每次重新呼叫 step，不知道迭代的是 flux、輪次或其他領域資料。

plan 是 frozen pydantic model。tunables 是可由 JSON 驗證的巢狀 pydantic model。state 是純資料 dataclass，可包含陣列。record 是步驟摘要，不含量測曲線。engine 在呼叫前複製 plan、tunables、state，在提交時再複製 record、state。

experiment 是 `run_t1(run: Run[T1Cfg]) -> T1Result` 這類普通函式。分析也是普通函式。兩者沒有共同的 Experiment wrapper 或 GUI 基底類別。workflow 明確傳入 saver，engine 在交還 Completed 前保存 cfg 與 Result。期 0 使用測試替身，期 1 接 storage-redesign 4b 的 native saver。

### 可取消呼叫與不可取消提交

step 回傳 `Next(record, state)` 時，engine 追加 journal 並切換 state。這段提交不接受 Pause 或 Stop 中斷。控制請求在提交後生效。尚未回傳的 step 被取消時，engine 丟棄其副本，關閉 generator，保留該次呼叫的檔案。

Pause 與 Stop 都發出協作 cancel，不等同於工作已停止。Pause 等正在執行的 effect 返回後才進入 paused。Resume 用已提交 state 與最新 tunables 重新呼叫 step。Stop 不再呼叫 step。沒有 rollback、finalize 或隱式硬體收尾。

GUI 每個 Start／Resume segment 使用 [[0066]] 的 operation 與 lease。Pause 的 segment 真正結束後才釋放 lease；Resume 先重新取得 lease，宿主再注入 device snapshot。engine 不取得或釋放 GUI lease。超限設定由宿主 device port 拒絕，且與 cfg.dev setup 使用同一套限制。這層保護在期 2 開放 MCP 啟動之前完成。

### 圖與進度

workflow 取得原生 matplotlib Axes，從 state 重畫累積資料。Live 綁定只把單次實驗 buffer 投影到 artists。engine 執行緒是 figure 唯一寫者；宿主在明確 refresh 時取得快照，GUI 不掛 live canvas。快照策略需在期 2 驗證，不把 worker 呼叫當成 Qt owner 寫入。

進度沿用 `progress_bar.BaseProgressBar`。同名 bar 跨步驟保留，total 必須相同。engine 在 terminal 關閉 bar。Pause 保留 bar，重做步驟使用 `set_progress`。

### 保存權威

沿用 [[0063]] 的區分。單次資料檔由 saver 擁有格式。engine 擁有 journal、manifest、呼叫序號與 `runs/` 路徑。workflow 只寫入自己的 `files/`。journal 的提交行是離線讀取依據，不以目錄存在或圖上的半行判定提交。

每次呼叫建立獨立目錄，包括 Done 與 Aborted。序號是呼叫次序，不是 flux index 或提交次序。Pause、Stop、例外都不刪除目錄。state 不持久化，也不提供崩潰後接續。

## 可執行依賴投影

實作時新增 `.importlinter` contract ID `workflows-core-no-ui`，候選編號 C18。source 為 `zcu_tools.experiment.workflows`，禁止依賴 `zcu_tools.gui`、`zcu_tools.mcp`、`zcu_tools.notebook`、QtPy 與各 Qt binding。Qt 外部套件檢查需啟用 import-linter 的 external package 分析。這條 contract 不增加 ignore_imports。

期 0 尚未建立 package，因此本波不加入無法解析 source 的 contract。本篇在 contract 落地並通過之前保持 draft。既有 C8、C9、C16 同時限制分層、循環與 framework 反向 import 使用者套件。核心對 progress_bar 與 runtime 的依賴使用 C8 的既有方向，不把 GUI 排除規則的債務當成先例。

## 取捨

步驟可丟棄，使定義者不用寫暫停恢復分支。代價是當前步驟已完成的校準也會重做，硬體副作用不回滾，圖上半行會暫時保留。

deepcopy 隔離可變資料，使定義者可以直接修改 state 副本。代價是每步複製累積資料。state 接近數百 MB 時再評估，不預先增加增量容器。

每次實驗明確給 saver，保持函式物件介面與 cfg／Result 型別配對。代價是 helper 需多寫一個 keyword，不引入 Experiment wrapper 或全域註冊。

D10 已核准所有 effect helper 使用 `yield from`，讓 Python 型別系統保留各 experiment 的回傳型別。engine 在入口拒絕非 effect 的 yield 值並提示正確寫法。axes、pbar 與資料存取仍直接呼叫。

D9 依失敗來源分流。Schedule 捕捉的 Exception 回 Failed，journal 保存例外型別、訊息與 traceback；Schedule 外例外使 run failed；KeyboardInterrupt 視為宿主 Stop。SoC 在 acquire 內斷線可以逐步 Failed，直到 Schedule 外操作拋出例外才停，符合 R3。不建立斷線例外分類表。

## 分期與排除

期 0 凍結介面並實作離線核心。期 1 接真實 T1 與 4b saver。期 2 接 GUI、RPC、MCP、4a 工作點與 ledger workflow producer，並更新 [[0068]]。期 3 補齊其他 workflow、2D 合併與 Labber 匯出，通過 CP-C 才退役。

不支援修改 workflow 結構或 plan、回頭重跑、舊 run 轉換、GUI 舊 run 檢視、env.history、finalize 或 measure 實驗遷移。本篇不授權真機操作。
