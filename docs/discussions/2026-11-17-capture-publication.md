---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulate/evolution/README.md
  - ../../lib/zcu_tools/simulators/README.md
basis:
  repo_commit: 6b4644765544df7855321ec35930e48a34c1b84f
---

# Capture 的發布與 acquire 關聯

## 結論

當時選擇公開 `simulate.evolution.write_capture(buffer)` 與 `capture_directory()`。
Writer 自己命名，回傳成功的絕對 Path，發布失敗則 warning 並回 None。
三個 environment 的 `numerics.debug_capture` 預設關閉。
啟用後，一次 acquire 共用一個 buffer，在 on_end 先發布 capture，再建立 telemetry summary。
第九個必填共通欄位 `capture_file` 保存成功檔名，關閉或發布失敗都填 null。

這篇記錄當時比較，不另擁有執行契約。
API、失敗與保留政策見 [capture persistence](../../lib/zcu_tools/simulate/evolution/README.md#capture-persistence)。
Environment 生命週期與關聯見 [debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)。

## 問題

當時 Engine 已能把多次 evolve 的輸入與選路追加到 caller-owned buffer，但不負責寫檔。
Environment 需要把完整 acquire 的 buffer 發布一次，也要在編譯失敗而沒有 evolve 時留下零筆 manifest。
只靠時間戳或 family 無法可靠地把檔案對回同次 telemetry，猜測模擬器身分也不是資料契約。

## 考慮過的選項與比較

### 直接函式與共同 cache root

當時採用單一函式，不建立 writer class、options record 或額外 acquire ID。
呼叫端只交 buffer，writer 產生 uuid 檔名並回傳結果。
與 Numba 共用 simulator 自己的 cache-root resolver，避免複製解析規則或重新依賴 utils。
公開目錄函式讓後續工具取得位置，不必重寫目錄版面。

這遵循 [ADR-0072 的封閉子模塊](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明)。
Writer 只依賴 simulate，environment 依賴 simulate，不讓 simulate 認識 acquire 或 simulators。
這也遵循 [ADR-0078 的凍結容器](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)。
發布沿用既有 frozen manifest，沒有另一個 codec 或 solver 設定格式。

### 區分契約錯誤與可隔離的發布失敗

Cache override 必須非空且為絕對路徑，錯誤照拋。
取 records 的錯誤同樣照拋，不把 caller 的契約錯誤當磁碟故障。
之後的 OSError 與 JSON codec 拒絕可以隔離，warning 一次後回 None。
Warnings-as-errors 也不能讓這種 best-effort 診斷改變 acquire 結果。
其他例外不吞掉。

當時選擇同目錄暫存檔加 os.replace，避免 reader 看見半份檔案。
空 buffer 仍寫 manifest，因為 None 已代表發布失敗，不能再代表沒有 evolve。
同一 buffer 重寫時使用新檔名，不覆蓋先前快照。

### 明確檔名關聯，不推測身分

當時選擇在共通 summary 加 `capture_file`，而非放進 simulator-prefixed extras。
Writer 明列投影，保留 caller-owned summary。
先發布檔案，再建 summary，才能使用實際成功的 basename。
沒有檔案時一律 null，不從 null 推測關閉還是發布失敗。
舊行缺此欄位代表沒有關聯，不增加 legacy 轉換層。

此方案沿用 [共用 acquire telemetry](2026-11-16-shared-acquire-telemetry.md) 的中性 writer 與 namespaced extras 分工。
差別是 capture 屬於三個 environment 的共通資料，不是某個模擬器的延伸欄位。
[Acquire outcome 作用域](2026-11-16-acquire-outcome-scope.md)已讓 binding 在 observer I/O 前量完 wall，這次沿用它，不另外扣秒數。

## 被否決的選項與理由

- Writer 收檔名、目錄與 acquire options。當時沒有第二種政策需求，這會把命名與版面散給 caller。
- 在 Engine 寫檔。這會讓每次 evolve 決定 acquire 的發布時機，也無法保存編譯失敗的空 buffer。
- 依檔名、路徑或 family 推模擬器身分。這些不是穩定的身分欄位。
- 吞掉全部 Exception。程式錯誤與入口契約錯誤應該可見，只有指定發布失敗需要隔離。
- 提前設上限或串流格式。當時先保留完整輸入與既有 codec，容量以一次正式量測確認，不自行截斷。

## 留下的問題

V1 不自動限大小、截斷或淘汰，使用者負責刪檔。
Buffer 隨 evolve 增長，發布時還需要全份 JSON 與 gzip bytes。
當時要求量 overhead、單檔大小與開關的峰值 RSS 差。
單檔超過 256 MiB 或峰值差超過 2 GiB 時，應帶數字重審上限、截斷或串流，不把小場景結果外推為容量保證。

## 依據

當時的調查與執行工作檔未入庫。本文保留 API、責任、失敗分類與容量回頭門檻，不要求讀者取得工作檔。
公開接縫測試位於 [writer tests](../../tests/simulate/evolution/test_capture_writer.py) 與 [environment tests](../../tests/simulators/test_capture.py)。
數值量測結果由 evolution README 持有，測試不以 pytest duration 代替 overhead。
