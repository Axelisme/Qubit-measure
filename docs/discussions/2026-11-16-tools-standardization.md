---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0068-remote-transport.md
  - ../../tools/README.md
basis:
  repo: cb6acb0c94b1a6b4f314cc9ae1e18df014d9b500
  external: none
---

# Tools 標準化：補工具缺口，不搬目錄、不加硬閘

## 結論

2026-11-16，designer 依使用者的代裁授權採用 B1、E1a、E3a 與 B2 的一部分，留待使用者追問。品質工具補齊 `-h`、`argv` 與 docstring，gate 現況表加只報告的 `docstring N`，README 寫下 CLI 正向規則。不搬目錄，也不新增硬閘。

這批修改已在 main `cb6acb0c9` 落地。工具契約與命令由 [tools README](../../tools/README.md) 擁有，工具定位由 [ADR-0068](../adr/0068-remote-transport.md#分層與連線) 擁有，完成前的操作流程由 [CLAUDE.md](../../CLAUDE.md#5-完成前驗證) 擁有。本篇保存當時的比較，不另立規則。

`family_pareto.py` 與 `family_pareto_scenarios.py` 的行數處置交給已核可的 sim-tool-layering，沒有在這次標準化修改中完成。

## 問題

審查要回答的是，`tools/` 的 27 支 Python 工具及支援模組，如何達到使用者要求的「好用、好讀、通用」。逐支調查與三輪 challenge 後，成因分成兩類。

1. 舊工具留下黑箱文件缺口。黑箱條文於 2026-10-04 入庫，審查時 Ruff D100、D101、D102、D103 共命中 50 筆，分布在 9 支工具。這些定義行的最後修改日全在 09-24 或 09-25，規則之後新增的工具沒有命中。因此這是規則之前的舊債，不能當作正向規則已失效的證據。
2. 新工具逐票累積，剛跨過行數門檻。27 支中有 12 支在 11-11 之後才出現。`family_pareto.py` 曾由 975 行增至 1016 行，`family_pareto_scenarios.py` 也超過 1000 行。審查沒有取得跨線當次的 ratchet 回執，不能斷言檢查沒跑或失效。能確認的是，拆出職責或記錄內聚理由的處置尚未完成。

調查撤回了「tools 沒有任何自動煞車」的說法。共用掃描範圍已包含 `tools/`，Ruff、Pyright、file-size 與 suppression 等檢查並未排除它。當時 ratchet 的 Radon 不掃 tools，import-linter 的 root packages 也不含 tools，這兩項限制不能推論成所有檢查都未涵蓋。

CLI 另有可觀察的缺口。`run_heavy -h` 回傳非零，5 支沒有 argparse 的 check 收到 `-h` 仍直接執行檢查，其中 `check_pytest_collection` 會做完整 collection。即使回傳 0，沒有 usage 也不算提供 help。`check_script_awaits` 收到不存在的 `--root` 還會回報零違規，這違反 Fast Fail。

文件也有分歧。README 稱 `gate.py` 是日常入口，當時 CLAUDE.md 的完成前驗證卻沒有提它。有 caller 的工具仍然缺 README 命令，所以「沒有 CI 或 caller」不足以解釋文件缺口。

## 考慮過的選項與比較

### 修補或搬遷

下表成本是審查時的估計，不是落地 diff 的統計。

| 選項 | 做法與成本 | 取捨 |
| --- | --- | --- |
| A，保留並記債 | 只替內聚的 scenarios 加一行行數理由，另登兩筆待辦。主工具有可拆的 owner，不能直接加標記。 | 改動最少，但 9 支工具的不合格項仍留著，不滿足使用者的「必須兼具」。 |
| B1，補必要缺口 | 修 help、argv、錯誤處理與主要 docstring 缺口。原案還包括抽出 Pareto 結果型別，約 290 行修改、330 行搬移。 | 採用工具修補部分。Pareto 抽取受分層方案影響，改交 sim-tool-layering。 |
| B2，補其餘缺口 | 原案約 250 行，包括選項 help、文件、其餘 docstring、報表改善及兩支工具的補測試。 | 只採用文件與介面說明等部分，沒有整包採用。 |
| S，只搬模擬器組 | 模擬器工具與資料搬到 `tools/simulator/`，估計約 20 檔、150 行 import／命令改寫及 600 行文件搬移。 | 分組會改 CLI 路徑、測試位置及既有文件連結，沒有解決工具本身的缺口。 |
| C，全面重排 | 分成 quality 與 simulator，共用 receipt、統一 root 選項，各組另寫 README。估計約 55 檔，至少兩個 task。 | 同時改動路徑契約與進行中工作的命令，成本大於這次補缺口的範圍。 |

採用的 B1 修補了 `run_heavy` 的 help，讓 `check_file_size`、`check_test_capabilities`、`check_suppressions`、`check_test_path_correspondence`、`check_pytest_collection` 接收 argv 並以 argparse 處理 help。另修不存在 root 的失敗行為，補 `check_pytest_collection` 與 `quality_report` 的 docstring。

B2 採用的部分包括 33 個 `help=`、README 的「工具／命令／exit code／是否在 gate」統一表、命令缺少的 `--directory <worktree>`、其餘 main／class 的 docstring、gate 耗時敘述，以及 `test_doc_relations.py` 的 capability marker。報表字串重組等 Pareto 項目沒有併入這批品質工具修改。

### 守護強度

當時採用的 progressive guardrails 判準是，先修現有缺口並使用最弱的正向引導，有規則寫下後仍失敗的證據才升級。審查據此把原先提議的兩個強制檢查降階。

| 選項 | 機制 | 當時的處置 |
| --- | --- | --- |
| E1a | gate 現況表加 `docstring N`，只報告 tools 的 D100、D101、D102、D103 命中數。 | 採用，不改 exit code，也不加入比較模式的硬步驟。 |
| E1b | gate 增加 docstring 硬步驟。 | 暫不採用，50 筆全是黑箱規則之前的舊債。 |
| E1′ | 全樹啟用 D 規則。 | 暫不採用，tools 以外另有 12,844 筆命中，沒有查明哪些是規則之後的違反。 |
| E2 | ratchet 的 Radon roots 加入 tools。 | 暫不採用，沒有對應的失敗類別，Radon 分數仍作閱讀線索。 |
| E3a | README 寫明，有 CLI 的工具收 argv，`-h` 印 usage 並以 0 結束。 | 採用，配合 B1 修好現有六支。 |
| E3b | 自動列舉工具，在同一 process 呼叫 `main(["-h"])` 驗證 help。 | 暫不採用，CLI help 規則是本次審查才寫下，現有缺口不是規則之後的違反。 |
| E4 | 用 AST 白名單限制 tools 對 tests 的 import，禁止 import 測試本體與 conftest。 | 暫不採用，沒有工具 import 測試本體的實例；白名單保留依賴的前提也已由 Q1 的分層方案取代。 |

E1a 使用 isolated Ruff 規則，只取得缺少 docstring 的數量。規則清單寫在 gate 程式內仍是設定，不是「零新設定」。若日後升成硬閘，必須在 README 說明，不能假定 ratchet 的設定 REVIEW 會偵測這份清單。

### Q1、Q2、Q3 的處置

Q1 問 tools 與 tests 的依賴方向。審查時，tools 有 11 條 import 指向 6 個測試支援模組，`tests/simulate/evolution/contract/guard_sweep_cost.py` 又反向 import `tools.family_pareto`。原 dossier 比較了保留並記錄、加白名單、把六個模組搬進 tools 或 lib，以及把 Pareto 全組搬進 tests。原先建議保留依賴，但這不再是待選方案。

先前已由使用者核可的模擬器分層方案回答了 Q1。tools 與 tests 互不 import，經凍結數值資料與參考解 cache 交接，獨立 solver 留在 tests。正向與反向邊都要移除，不能只改 tools 那一側。實作交給 sim-tool-layering，不為這次標準化另外搬一套 oracle。

Q2 確認 tools 是否容納模擬器量測工具與 `probe_snapshot.py`。使用者已把 benchmark 當作 tools 內的通用開發工具，也選擇把 probe snapshot 工具放在 tools。ADR-0068 原先「tools 只放 repo 品質檢查」出現在 MCP 組合根選址段落，是附帶描述，不是使用者對 tools 範圍的裁決。本次區別這兩者，修正 ADR 與 README 的定位句，沒有搬目錄。現行範圍見 [ADR-0068](../adr/0068-remote-transport.md#分層與連線)。

Q3 問 gate 是否寫進 CLAUDE.md。採用把完成前驗證第 2 步改成執行 `tools/gate.py --base <ref>`。gate 預設先做 import sorting 與 formatter，保留原步驟的效果，再執行其他檢查。這也讓日後可能採用的 E1b 有明確的日常入口。

### 先例與立場

- 遵循使用者 2026-11-14 的要求：「入库就不能只是随便写，你必须要非常好用，非常好读，还有通用性，这三者必须要兼具。」使用者以 `benchmark_simulator.py` 為例，工具可以只服務模擬器，只要不綁死單一 family 或場景。通用不等於每支工具都要跨領域。
- 遵循使用者 2026-11-12 的判斷：「有帮助于开发的工具本来就应该要内建在 repo 本身。」本次沒有把工具移出 repo，也沒有把重排目錄當作達標的必要條件。
- 遵循 [程式碼品質的行數裁決](../code-quality.md#審查裁決)。使用者於 2026-10-29 說：「真內聚的模塊，超過行數就加tolerence好了」、「是否內聚應該是看維護者的視角，改一個功能只要改一個模塊」。先抽出有 owner 的職責，再判斷剩餘模組是否內聚，不為了降數字硬拆。
- 遵循同一份文件的使用方語義邊界，tools 不假設 lab 的欄位名、實驗名或 cfg 形狀。補 CLI 與文件沒有放寬這條邊界。
- 遵循已核可的模擬器分層方案，取代 dossier 原先保留 tools 與 tests 依賴的建議。區別 ADR-0068 的舊附帶句與使用者已選定的工具位置，處置如 Q2。

## 被否決的選項與理由

A 只記債，沒有修掉已確認的工具缺口。S 搬模擬器組主要換來分組，卻改動命令與路徑契約。C 全面重排至少要兩個 task，不能當作補 help 與 docstring 的前置工作，三者都未採用。

E1b、E1′、E2、E3b、E4 沒有足以支持當下控制強度的失敗證據，因此沒有跟著修補一起加入。尤其 `run_heavy` 雖然較新，仍早於這次才明訂的 help 規則，不能用它證成「規則之後又違反」。

沒有為無選項的 check 設 help 例外。那會讓 `-h` 繼續執行檢查，甚至啟動完整 collection。也沒有在這次標準化中替 `check_test_structure` 與 `analyze_routing_telemetry` 新增約 140 行測試，當時沒有相應失敗實例，這不是修補既有介面缺口的必要部分。

`tools/benchmark_simulator_workloads.py` 與 `tools/family_pareto_settings.py` 的合成 board 沒有合併。它們的值不相等，也不是複製；合併會改變 benchmark 版本比較與既有 Pareto 摘要的可比性。`benchmark_simulator*` 已由核可的分層方案安排退休，審查中針對它的標準化項目不再另做。

## 留下的問題

E1b 的升級條件是，docstring 清到 0 後，黑箱規則入庫後新增的檔案又出現命中。E3b 的升級條件是，README 寫下規則後，新增或改動的 CLI 再次出現 help 失靈。兩者都要先取得新失敗的證據，不能因這次已修舊債就直接升級。

若觸發 E3b，測試須自動找出 `tools/*.py` 中定義 main 的工具，缺少 argv 參數也要失敗，不能用固定名單漏掉新工具。它在同一 process 呼叫公開的 `main(["-h"])`，接受回傳 0 或拋出 `SystemExit(0)`，並確認 stdout 有 usage。這是留下的升級方案，不是本次新增的測試。

`family_pareto.py` 超過 1000 行的處置併入 sim-tool-layering 票 02。先抽出有 owner 的結果型別，剩餘部分確實內聚才寫行數理由。`family_pareto_scenarios.py` 的處置併入票 01，因為它會改成場景 producer。`benchmark_simulator*` 由票 03 退休。這些都是後續分層工作的責任，不代表標準化落地時已完成。

行數理由也不是機械豁免。`check_file_size.py` 不讀 `tolerated(file-size)`，加註解不會減少現況列的超標數。驗收必須分開看行數訊號、內聚理由與實際拆出的職責。

## 依據

本文依據 main `cb6acb0c9`，保留 2026-11-14 至 11-16 的工具審查、三輪 challenge、設計者處置，以及 11-16 的代裁與落地結果。原始 dossier、challenge、處置表與任務決策工作檔未入庫；上述問題、數字、比較與理由已寫入正文，讀者不需要那些檔案。

現行可核對的文件是 [tools README](../../tools/README.md)、[ADR-0068](../adr/0068-remote-transport.md#分層與連線)、[程式碼品質](../code-quality.md#審查裁決) 與 [CLAUDE.md](../../CLAUDE.md#5-完成前驗證)。本篇沒有外部來源。
