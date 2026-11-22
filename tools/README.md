# tools/

**Last updated:** 2026-11-23 (curated FTF wall-time scenes)

**讀者：** framework-dev。查閱 repo 開發工具的 CLI、輸出與比較機制。

`tools/` 放 repo 內部開發工具：品質檢查、重運算執行器與模擬器量測工具。review 品質判準由 [程式碼品質](../docs/code-quality.md) 定義，環境與驗證流程依 [AGENTS.md](../AGENTS.md)。`scripts/` 放使用者自己執行的板端 server、GUI 啟動與資料工具。[`mcp_servers/`](../mcp_servers/README.md) 放 agent 的 MCP 組合根。
離線 Migration CLI 位於 `migrations/migrate_storage.py`，以 `python -m migrations.migrate_storage` 執行，不接量測啟動流程。執行 legacy 遷移、處理 pending／resume 或修改 component kind 前，讀 [Agent 離線遷移說明書](../docs/storage-migration.md)。

每支有命令列入口的工具收 `argv`，`-h` 以 0 結束並印 usage。`argv=None` 讀取 `sys.argv`。
品質檢查以純函式處理判準，由回傳 exit code 的 `main(argv)` 把 JSON receipt 輸出到 stdout。
`run_heavy.py` 的 `main()` 管理排隊、子程序與清理，只有子程序啟動後才輸出 JSON receipt。
`check_script_awaits.py` 與 `check_quality.py` 另把人類可讀的違規行輸出到 stderr。
`check_quality.py suppressions` 只計量不判定，在 stderr 列出使用量最多的抑制，最多 20 筆，並固定 exit 0。
`check_ratchet.py` 與
`check_pytest_collection.py` 只在工具本身失敗時寫 stderr。`run_heavy.py` 在 stderr 說明等待原因、啟動錯誤與清理失敗。
`gate.py` 是給人看的入口，輸出純文字。
它們可以被 import，`tools/check_ratchet.py` 就是這樣使用其他幾支的。

`_support.py` 共用走訪樹、讀 attribute chain、透過 import 解析呼叫與載入另一支檢查的機制。
各支檢查自行定義判準。`check_experiment_imports.py` 是 `.importlinter` 註冊的 custom contract，依 `core.py` 自動辨認實驗目錄，檢查實驗間與跨家族 helper 的直接 import，違規時列出來源、目標與行號；它不 import 實驗定義。
共用模組要求 `tools/` 在 `sys.path` 上。
直接執行某支檢查、`load_tool` 載入另一支與 pytest 的 `pythonpath` 都滿足這項要求。

## CLI 一覽

下表命令省略共同前綴 `uv run --directory <worktree> --no-sync -- python`，並在目標 worktree 根目錄執行。
「現況」只在省略 `--base` 時執行。「ratchet」表示比較時使用該 detector，不代表單支工具的絕對數字必須為零。
支援模組沒有命令列入口，由表中工具或 `lint-imports` 呼叫。

| 工具 | 命令 | Exit code | 是否在 gate |
| --- | --- | --- | --- |
| `gate.py` | `tools/gate.py [--base <ref>] [--no-fix]` | 現況 0；比較 0 全通過、1 有失敗、2 base 錯誤 | 入口 |
| `check_ratchet.py` | `tools/check_ratchet.py --base <ref>` | 0 無上升、1 有上升、2 工具失敗 | 比較 |
| `check_quality.py` | `tools/check_quality.py <subcommand>`，見[品質檢查](#品質檢查) | 0／1；`test-structure` 與 `all` 另有 2 | 現況、ratchet |
| `check_script_awaits.py` | `tools/check_script_awaits.py [--root <repo>]` | 0 無違規、1 有違規、2 root／讀取／解析失敗 | 現況、比較 |
| `check_mcp_text_budget.py` | `tools/check_mcp_text_budget.py [--timeout <seconds>]` | 0 預算內、1 超額、2 收集失敗 | 現況、比較 |
| `check_experiment_imports.py` | `lint-imports`，不經 Python 前綴 | 由 lint-imports 回報 0 通過、1 失敗 | import contracts |
| `check_doc_readers.py` | `tools/check_doc_readers.py [--root <repo>]` | 0 宣告完整、1 有缺口 | 否 |
| `doc_relations.py` | `tools/doc_relations.py <base>` 或 `tools/doc_relations.py --check` | diff 0；check 0 連結存在、1 有違規 | 否 |
| `check_pytest_collection.py` | `tools/check_pytest_collection.py` | 0 四組一致、1 收集失敗或不一致 | 否，另跑 |
| `quality_report.py` | `tools/quality_report.py snapshot` 或 `tools/quality_report.py compare <before> <after>` | 0 報表完成、2 無法產生或比較 | 否 |
| `probe_snapshot.py` | `tools/probe_snapshot.py compare A B [--rtol R] [--atol A]` | 0 輸出相同或容差內且輸入相同；1 陣列、PNG 或 case 不同／缺漏，優先於 3；2 用法、讀寫或格式錯誤；3 輸出相同或容差內但輸入不同 | 否，手動比較離線 probe 輸出 |
| `run_heavy.py` | `tools/run_heavy.py --log <file> <timeout> -- <command>` | 子程序狀態；2 啟動前失敗、70 runner 失敗、75 等待逾時、124 執行逾時 | 否，重計算外層 |
| `analyze_routing_telemetry.py` | `tools/analyze_routing_telemetry.py <path> ...` | 0 完成，無法讀檔時非零 | 否 |
| `inspect_simulator_capture.py` | `tools/inspect_simulator_capture.py [path] --telemetry-directory DIRECTORY` | 0 完成，壞資料或讀檔失敗時非零 | 否 |
| `family_pareto.py` | `-m tools.family_pareto --scenario <name>` | 0 完成，輸入或量測失敗時非零 | 否 |
| `paired_benchmark.py` | `-m tools.paired_benchmark run --baseline A --candidate B --scenario NAME --reference-cache DIR --blocks K --exclusive --output FILE` | 0 完成；1 輸入、cache 或 child 失敗；2 用法錯誤 | 否，需外層 exclusive runner |
| `experiment_wall_time.py` | `-m tools.experiment_wall_time`，參數見[實驗 Run 牆鐘](#實驗-run-牆鐘) | 0 所選列均有結果；3 已出報表但仍有未完列；1 工具或環境錯誤；2 用法錯誤 | 否，只記耗時 |

Usage 錯誤由 argparse 以 2 結束；未轉譯的 Git、讀檔或其他執行錯誤仍會傳出例外，不能當作檢查通過。
模擬器量測與重計算的授權、排隊和收據要求見後續各節。表中的命令不提供執行授權。

## 重計算執行器

全樹 pytest、full Pyright 與效能量測經 `run_heavy.py` 執行，讓同一 repo 的所有 worktree 共用排隊與 CPU slot。
先切換到要驗證的 worktree，再執行下列命令，讓子程序沿用該工作目錄：

```bash
uv run --directory <worktree> --no-sync -- \
  python tools/run_heavy.py --log <file> [--wait-limit <seconds>] [--exclusive] <timeout-seconds> -- <command> [args...]
```

`--log` 將命令的 stdout 與 stderr 合併寫入指定檔案，並覆寫原有內容。
先建立 log 與收據檔案的父目錄，避免開檔失敗。
Runner 結束後將一行 JSON 收據印到自己的 stdout，重導 stdout 即可另存收據。
`--wait-limit` 限制等待 slot 與記憶體的秒數，省略時持續等待。
`<timeout-seconds>` 限制命令啟動後的執行時間，不包含排隊時間。

Runner 以 Linux `flock` 鎖住 `<git common dir>/run-heavy/slot-<n>.lock`，讓同一 repo 的 worktrees 共用 2 個 slot。
Runner 會建立鎖目錄，清理 `.agent_state/` 不會移除鎖。
每個 slot 取得 `cpu_count // 2` 個連續 CPU，並將子程序的 CPU affinity 固定在這些核心，CPU 總數為奇數時不使用最後一核。
Runner 依整台機器的 CPU 編號切分 slot，不依呼叫者的 affinity 改變切法。
命令取得空 slot 且 `/proc/meminfo` 的 MemAvailable 至少 8 GB 時才啟動，讀不到 MemAvailable 就回傳 2。
`--exclusive` 依 slot 編號順序取得兩個鎖，取不齊就全部釋放後再等。
獨占命令使用兩個 slot 的 CPU 聯集，總核數為奇數時仍忽略最後一核。
獨占命令持鎖期間，一般與其他獨占請求都要等待。等待不保證先來先執行。
子程序繼承 slot 鎖，runner 結束後，命令或繼承鎖的後代仍持有鎖時不會釋出 slot。
所有持鎖程序結束後，作業系統自動釋出鎖。

Runner 固定設定 `QT_QPA_PLATFORM=offscreen`、`MPLBACKEND=Agg`，讓命令不開啟顯示視窗。
它也將 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 與 `NUMBA_NUM_THREADS` 設為 `1`，限制數值函式庫的執行緒數。
命令超時或結束後仍有同 process group 的存活程序時，runner 先送 SIGTERM，10 秒後仍未停止就送 SIGKILL。
送出 SIGKILL 後最多再等 10 秒，包含直接子程序的回收與 group 殘留檢查。
超過上限就停止等待，在 stderr 說明可能仍有程序存活，並照常輸出收據。
Z 或 X 狀態的程序不算存活程序；無法停止的程序若仍繼承 slot 鎖，會繼續持鎖直到結束。
Runner 收到 SIGINT 或 SIGTERM 時，將訊號轉送給命令的整個 process group。

| 結果 | Runner exit status |
| --- | --- |
| 命令自行結束 | 回傳命令的 exit status，被訊號終止則回傳 `128 + signal` |
| 命令超時 | 回傳 `124` |
| 啟動後的 runner 作業或清理失敗 | 回傳 `70`，在 stderr 說明原因，盡力清理 group 並輸出收據 |
| 超過 `--wait-limit` 仍未獲准啟動 | 回傳 `75`，將原因寫到 stderr，不輸出收據 |
| 啟動前失敗 | 回傳 `2`，將原因寫到 stderr，不輸出收據 |

Runner 遇到用法錯誤、非 Linux、不在已提交的 Git worktree、少於 2 核、log 無法開啟、affinity 無法設定或命令無法執行時，會在啟動前失敗。
只有命令真的開始執行才會印收據，開始後無論成功、失敗、超時或被訊號終止都會印收據。
若收據本身無法序列化或寫入 stdout，runner 會傳出原始例外，不將它當作啟動前失敗。

| 收據欄位 | 記錄內容 |
| --- | --- |
| `command`、`cwd` | 記錄命令的 argv 與執行工作目錄 |
| `head`、`tree`、`dirty` | 排隊前先檢查 Git 是否可讀，取得 slot 後重讀該 worktree 的 HEAD commit、`HEAD^{tree}` 與修改狀態並寫入收據。`git status --porcelain` 含 untracked 且非空時將 `dirty` 設為 `true` |
| `exclusive`、`slot` | 一般模式為 `false` 與取得的 slot 編號；獨占模式為 `true` 與 `null` |
| `cpus`、`env` | 記錄子程序的 CPU affinity 清單與 runner 固定設定的環境變數。獨占模式的 `cpus` 是兩 slot 的聯集 |
| `wait_seconds` | 記錄啟動前的等待秒數 |
| `started_utc`、`finished_utc`、`elapsed_seconds` | Runner 啟動子程序前與清理流程結束後的 UTC 時刻，以及這段期間的 monotonic 秒數 |
| `exit_code`、`timed_out` | 記錄 runner 的 exit status，超時時將 `timed_out` 設為 `true` |
| `residual_killed` | 直接子程序結束後仍找到同 process group 的存活程序時設為 `true`，超時本身的整組清理只記在 `timed_out` |
| `log` | 記錄命令輸出 log 的路徑 |

收據記錄工作樹與執行身分，不記錄 probe 的輸入內容。相同 HEAD 不代表相同輸入。
要核對 cfg、seed 或來源陣列，請比較 [Probe snapshots](#probe-snapshots) 的輸入指紋。

全樹測試沿用 [tests/README.md 的完整選集命令](../tests/README.md#執行測試套件)，保留 marker、rootdir 與 failure log 選項。
Full Pyright 將下方的全樹比較命令包進 runner，讓排隊與收據套用到整次比較：

```bash
uv run --directory <worktree> --no-sync -- \
  python tools/run_heavy.py --log <per-task 目錄>/pyright.log 1800 -- \
  python tools/check_ratchet.py --base <ref> --detector pyright --full-pyright \
  > <per-task 目錄>/pyright-receipt.json
```

## Probe snapshots

| 命令 | Exit code | 是否在 gate |
| --- | --- | --- |
| `python tools/probe_snapshot.py compare A B [--rtol R] [--atol A]` | 0 相同或容差內且輸入相同；1 陣列、PNG 或 case 不同／缺漏；2 用法、讀寫或格式錯誤；3 輸出相同或容差內但輸入不同；`-h` 回 0 | 否，手動比較離線 probe 輸出 |

在自己的 probe script 中 import writer。Caller 選擇要保存的陣列或 dataclass、PNG bytes 與輸入：

```python
from pathlib import Path
from tools.probe_snapshot import write_snapshot

write_snapshot(
    Path("snapshots/before"), "t1.run", result,
    pngs, inputs={"cfg": cfg, "seed": 7, "environment": environment_inputs},
)
```

`inputs` 是必填 keyword-only，可明填 `None`。Caller 提供 cfg、analysis options、seed、
環境設定或來源陣列；writer 不發現實驗或讀取環境私有狀態。
每個 label 寫入新目錄，保存 `arrays.npz`、PNG 與 `inputs.json`。既存 label 拒絕寫入，
重跑時選新 output root。輸入指紋包含陣列的 shape、dtype 與內容，不只記形狀。
指紋只涵蓋 caller 提供的輸入。要核對工作樹與執行身分，請另看[重計算執行器的收據](#重計算執行器)。

同一支 script 在兩個 worktree 各以自己的 `uv run --directory <worktree> --no-sync --`
執行一次，分別寫 A、B，再執行上表 compare 命令。工具不啟動其他 worktree 或儀器。
表格逐 case 列出陣列、圖與輸入的比較結果。預設要求 shape、dtype 與值完全相同，
相同位置的 NaN 視為相等。PNG 比解碼後的 RGBA 像素，不比檔案 bytes 或 metadata。
明示 `--rtol` 或 `--atol` 才容許浮點／複數誤差，未指定的一項為 0。
符合容差但非完全相同時顯示 `WITHIN TOL`，PNG 仍須像素相同。
輸出差異的 exit 1 優先於輸入差異的 exit 3。Exit 3 需要人工判讀，不表示通過或失敗。
比較結果不宣稱是 regression；模擬器或環境改動本來就可能改變結果。仍須逐張看圖驗收。

## 實驗 Run 牆鐘

[`case-inputs.json`](data/experiment_wall_time/case-inputs.json) 決定全表。首版 62 列包含先前手選的 52 個工作量（選集 commit `4a2dea87690bfcb2764886b2e5feb0b2113510c9`），另有 10 個快磁通或共用 coil 的派送驗收場景。每個場景填唯一 `scene_id`、精確 adapter key，以及有界 Form 覆寫的 path 和值。同一 adapter 可對應不同場景。Lab 新增 adapter 不會自動增加列。

入庫審閱時，維護者核對場景與覆寫的 `basis` 理由及 `unpinned.form`／`point` 的來源說明。CLI 不檢查這些註記的有無或內容。查核原則見[程式碼品質〈依判定用途決定查核與記錄〉](../docs/code-quality.md#依判定用途決定查核與記錄)。`ftf_exclusion` 會影響派送。對仍有註冊 adapter 的場景填非空理由時，工具不開表也不派送 Run。報表以該理由將該列標為 `not_runnable`。回查 manifest 解析與派送分支時，搜尋 `read_manifest` 與 `_run_batch`。

省略 `--select` 會選整份 manifest。指定時先比對精確 scene ID，再比對 manifest 中 adapter key 的第一段家族名。例如 `--select twotone/freq` 選一個場景，`--select mux` 選出 manifest 裡的 mux 場景。不同 selector 可以重疊，同一場景只跑一次。重複或找不到的 selector、無效 JSON 或執行輸入、重複 scene ID、同場景重複或重疊的覆寫 path 都是用法錯誤。父子路徑如 `["delay"]` 與 `["delay", "stop"]` 不可同時覆寫；工具在建立批次或派送任何列之前拒絕。省略 `--output` 時，工具在 `.agent_state/benchmark/wall-time/` 建立有時間戳的新目錄。量測指定的輸出目錄必須尚未存在。

使用有 Qt 與 QICK 的 worktree 環境。工具用 `Controller(view=None)` 和 Qt 事件迴圈，不開視窗、遠端服務或硬體連線。每列使用工具自持的 [FTF 設定](data/experiment_wall_time/ftf-settings.yaml)、[board profile](data/experiment_wall_time/board-profile.yaml) 與[乾淨 fixture](data/experiment_wall_time/fixture/)。首版在離線從已驗的標準資料建立，不同步 lab 的 preset 或工作點。工具只從 `zcu_lab.assembly` 的 `lab_definitions` 取得元件和實驗宣告，不讀 lab 模擬檔或 startup。依賴邊界見 [ADR-0072〈使用者空間的依賴原則〉](../docs/adr/0072-module-boundaries.md#使用者空間的依賴原則)。Import 契約名稱是 `lab-assembly-only`。它用 `time_scale=0` 停用模擬等待。

工具將 fixture 複製到每列可寫的 project，經公開 entry Open、use point 選取 `ftf-demo/initial`，再開現行 adapter 的 Form。未覆寫的欄位取自現行 Form 及捕捉的 fixture 參數、模組和 aliases；需要操作者輸入的欄位可以留空。見 [Measure services 的開表 owner](../lib/zcu_tools/gui/app/measure/services/README.md#操作-owners)。Open 後工具以一次 `form.write` 套用該 scene 的覆寫，不會補回 Open 缺少的來源。路徑使用欄位名及從 0 起算的索引，例如 `["delay", "n"]` 是線性掃描點數。表單 Ready 才派送 Run。Run 的 admission、輸入快照與終態見 [ADR-0066〈問題與決策〉](../docs/adr/0066-operation-lifecycle.md#問題與決策)。

場景覆寫只保留操作者本來要選的欄位，或用於限制計時工作量的時域、板型與掃描範圍。[Constructive phase](../zcu_lab/v2/two_qubit/constructive_phase/guide.md#prepare-the-form) 的 Form 自帶可編輯的 −180° 至 180°、25 點相對相位掃描。計時案例只指定其 duration 端點。`singleshot/ac_stark` 的 Q1 pulse source 不能在 Open 提供有效的 mixer 初值；板型不需要 mixer，此案例才直接填 `null`。

Fresh FTF point 的 `qubit.kappa` 是模型線寬，單位是 MHz。用 Q1 的 κ 開 `twotone/freq`，全寬約 0.11723 MHz；這不是計時場景的 87.860 MHz 寬尋峰。要找峰請按用途調整掃窗。不要把有界計時窗當成 Form 初值或校準值。

查掃頻窗的校準來源、可編輯的表單初值與案例覆寫的分工，見[作者指南〈校準初值政策〉](../zcu_lab/v2/measure-authoring.md#校準初值政策)。

查 CZ pulse、slot 與相位配對的來源，從 [Joint readout 到 local Z 的校準路線](../zcu_lab/v2/two_qubit/guide.md#joint-readout-到-local-z-的校準路線)看校準順序。`ftf-demo/initial` 存有 CZ pulse 與 slot，但沒有 `$coupler.ext.cz_phase`。量到 local-Z correction 不等於把配對寫入來源；配對的身分檢查和寫回見 [Local Z〈Preview 與 Apply〉](../zcu_lab/v2/two_qubit/local_z/guide.md#preview-與-apply)。

查序列內的脈衝磁通線設定，見 [FTF `flux_lines` 欄位與限制](../zcu_lab/simulation_envs/FTF/field-guide.md#flux-lines序列內磁通校準與濾波)。查 point 裡的 DC 磁通校準，見 [ADR-0082〈Component 保存 offset 與 signed periods〉](../docs/adr/0082-component-flux-frame.md#component-保存-offset-與-signed-periods)。

`--workers N` 設定每列 FTF 環境的平行 worker 數。`N` 必須是正整數，省略時用 1。若要採 GUI 的 worker 策略，指定 `max(1, slot CPU 數 − 1)`，例如 4 核 slot 用 `--workers 3`。`run_heavy.py` 的收據記錄 payload 參數與 `cpus`，可核對 worker 數和 CPU affinity。

完整表或需要排他的正式比較，在已協調的窗口用 `tools/run_heavy.py --exclusive`。先設定 `output` 為新的絕對路徑，建立其父目錄，並將 `budget_s` 設為整批 payload 的秒數預算。下例省略 `--workers`，使用 1 個 worker。若要套用 GUI 策略，按排他 runner 分配的 CPU 數指定 `--workers`：

```bash
uv run --directory "$PWD" --no-sync -- python tools/run_heavy.py \
  --exclusive --wait-limit 120 --log "$output.log" "$budget_s" -- \
  python -m tools.experiment_wall_time --output "$output" > "$output.receipt.json"
```

非排他 runner 可用於 CPU 診斷或已核准的子集量測，其秒數不能代替排他比較。Linux 上，空的 ICD 目錄停用 OpenCL。以下用 4 核 slot 選一列，並把 cache 留在 worktree：

```bash
quality="$PWD/.agent_state/quality"
mkdir -p "$quality/no-opencl"
OCL_ICD_VENDORS="$quality/no-opencl" \
ZCU_TOOLS_CACHE_DIR="$quality/cache" NUMBA_CACHE_DIR="$quality/numba" \
uv run --directory "$PWD" --no-sync -- python tools/run_heavy.py \
  --wait-limit 60 --log "$output.log" 120 -- \
  python -m tools.experiment_wall_time --workers 3 --select twotone/freq --output "$output" \
  > "$output.receipt.json"
```

`batch.json` 記錄格式版本、本批 scene ID 與各自的 adapter key、Git HEAD 和 manifest、FTF 設定、board profile、fixture 檔案的 SHA-256。工具將 HEAD 和派送前的 dirty 狀態交給 telemetry writer。正式批次從乾淨 checkout 執行，執行期間不得改輸入。雜湊用於辨認輸入，不鎖定檔案。`start.json` 在送出 Run 前記 UTC 與 monotonic 秒數。Run 終止時先寫 `row.json` 再清理資源；逾時則先寫該列，再等 Run 終態與 Controller idle。`report.json` 和 `summary.md` 從 journal 產生，以場景為分母，分列已判定、本批選取與 manifest 場景數。

工具在送出 Run 及收到 Controller 終態事件時各取一次時間。`wall_s` 是兩次 monotonic clock 的差，UTC 時刻供閱讀。環境組裝、Open、覆寫及派送前落盤都不在計時內。工具不呼叫 Analyze 或 Save，也不量它們。每列的 `case_directory` 指向自己的 `project/` 和 `telemetry/`。`project/` 保留該列初始化的 entry、point 與 acquired ledger，不寫回入庫輸入。

工具從 [acquire telemetry](../lib/zcu_tools/simulators/README.md#acquire-telemetry) 加總各階段秒數。`core_s` 是 `compile_s`、`evolve_s`、`readout_s` 和 `materialize_s` 四項之和。`materialize_s` 含 acquire 內產生 raw shots 的時間，也含依 `acquire_id` 配對的延後 materialize。`remainder_s` 從 acquire 牆鐘扣除該牆鐘內的四項工作。延後 materialize 在 acquire 之外，不再從 remainder 扣除。`ShotReducer` 等 host 處理不計入核心。

何時在 acquire 內計時、何時由 getter 回報耗時，見 [Program simulation 的計時說明](../lib/zcu_tools/program/v2/simulation/README.md#raw-materialization-timing)。Raw shots 的取樣契約見 [ADC〈Integrated raw shots〉](../lib/zcu_tools/simulate/adc/README.md#integrated-raw-shots)，實作可搜尋 `ADCRawShots`。

Run 在期限內完成，且至少有一筆 acquire、每筆都有 acquire 牆鐘和四個分項、每筆補充紀錄都有主紀錄配對時，才填 `core_s`。缺 materialize 的舊 telemetry 或未走到階段的 `null` 都不補 0。配不到主紀錄的補充紀錄不計秒數或 acquire 次數，並使分項不完整。成功但分項不全的 `missing_breakdown` 列保留 Run 秒數，四個分項與 remainder 寫 `null`。分項不全或逾時時，`core_s` 寫 `null`，`core_s_lower_bound` 只加總分項齊全的主紀錄與配對到它們的補充紀錄。沒有分項齊全的主紀錄時，下限也寫 `null`。

完整只表示現存 telemetry 的分項齊全。Writer 採盡力寫入；工具無法察覺整筆主紀錄或補充紀錄遺失。若該 Run 期間出現保留上限刪檔或寫入失敗的警告，不要用該列的 `core_s` 斷言達標。

摘要的 Core 欄在不完整列顯示 `≥` 加下限，沒有下限則顯示「—」。3 分鐘判定使用未四捨五入的秒數。完整核心嚴格大於 180 秒才列入 `Core >180 s`。下限嚴格大於 180 秒列入 `Core 已知至少超標`，其餘沒有完整核心的列列入 `Core 未判定`。Run 秒數和期限都不參與核心判定。

`--row-timeout SECONDS` 從送出 Run 起算，預設 360 秒。只接受有限且介於 180 到 2,147,483.647 秒的值，兩端都可用。到期時先把所有 Python 執行緒的 stack 寫到該列的 `timeout-stack.txt`，再請 Controller 取消。`timeout` 的精確 `wall_s` 是 `null`，`wall_s_lower_bound` 表示 Run 大於期限。能加總的 acquire 分項標「部分」，沒有可用分項時標「分項缺」。請求取消後，工具先依當下的 telemetry 寫 `row.json`。取消寬限的 30 秒內，工具若觀察到 Run 終態且 Controller idle，就重新讀取 telemetry、更新逾時列，再繼續派送。寬限內未觀察到兩者時，工具保留先寫的列並停止派送，報表將尚未開始的列記為 `not_run`。

Controller 在期限內回報 Run `failed` 時，工具先寫 `run-failure.json`，再落盤 `run_failed` 列。該列記 Run 起點、收到失敗事件的時刻，以及 Controller 提供的錯誤類型與訊息。`failure_record` 是相對於該列 `case_directory` 的診斷檔路徑。失敗列不填 Run 或核心秒數。工具寫完該列才檢查 Controller 是否 idle；若仍有活動中的操作，就停止派送。摘要分別計數成功計時、`timeout` 與 `run_failed`。`Run failures` 按完全相同的 Controller 診斷分組，每組列出 scene ID 與診斷檔路徑。工具不推斷根因。

若 Open 成功，但表單仍缺欄位或 slot，工具記 `missing_defaults`，不派送 Run。乾淨 fixture 不附讀出 classifier。需要帶 radius 的 circle classifier 時，先依各自 guide 在相同讀出條件下做 GE 與 GE Post。這是上游校準前置，不是 FTF 不支援。Constructive phase 在 Form 驗證時檢查 classifier，Singleshot AC Stark 在 Run 時檢查。

`not_runnable` 只適用於場景明列的排除。已選的 `two_qubit/rb`、`two_qubit/rb_heralded` 仍缺 `$coupler.ext.cz_phase` 的 Q1／Q2 有序 Local Z 配對。工具記原因，不開 project，也不派送 Open 或 Run。未納入的 `two_qubit/cz_optimize` 仍有相同前置；JPA 也不在這份清單內。

已選場景的精確 adapter 不再註冊時，工具寫下 `missing_adapter`、scene ID、adapter key 和維護者待辦。不開 project、不派送 Run、不記耗時，也不算已判定。其他場景繼續，續跑不沿用該未量列。這不同於無效 selector，後者不開批次。

工具例外、未知 Run outcome、環境組裝、journal 與 telemetry 解析錯誤會中止指令，不轉成場景狀態。重建與續跑拒絕 scene ID 或 adapter 不符的結果列，以及沒有有效 Run 秒數的 `ok` 或 `missing_breakdown` 列。

中斷時，有 `start.json` 而沒有 `row.json` 的列記為 `incomplete`。其 `started_at` 是派送前記下的時刻，不證明 Run 已派送。兩份紀錄都沒有時，列為 `not_run`。兩者不填精確秒數或下限，也不算已判定。180 秒判定不涵蓋未入選的實驗。

只有[場景選定結論](../docs/discussions/2026-11-23-curated-wall-time-scenes.md#結論)指定的 10 個派送驗收場景，才要求觀察 Controller 發出的 `RunStarted` 事件及其 admission token（實作搜尋 `RunStartedPayload`）。

量測和 `--report-only` 使用同一套 exit status：

| Exit status | 意義 |
| --- | --- |
| `0` | 報表已寫，所選列均有結果。可以含核心未判定的列，例如 `timeout` 或 `run_failed`。 |
| `3` | 報表已寫，但含 `incomplete`、`not_run` 或 `missing_adapter`。收到 SIGINT 或 SIGTERM 後若能寫出部分報表，也適用此規則。 |
| `1` | 工具或環境錯誤，包含讀寫或解析 journal 失敗。 |
| `2` | 用法錯誤，例如無效的 manifest、選擇、期限或不相符的續跑來源。 |

這些是牆鐘工具的 exit status。`run_heavy.py` 強制結束 payload 時，外層收據仍依 runner 的規則回報。

若程序被強制結束，兩份報表可能沒有寫完。以下指令只讀原目錄的 journal，不讀 manifest、fixture 或 lab，也不派送 Run：

```bash
uv run --directory "$PWD" --no-sync -- python -m tools.experiment_wall_time \
  --report-only --output "$output"
```

舊 adapter-key 格式不能用新版續跑或重建。請在 `batch.json` 所記的原 commit 執行 `--report-only`。批次起迄與外層逾時看 `run_heavy.py` 的收據，不用來推算 Run 耗時。要補未派送或未完成的列，使用下方的續跑方式。

續跑會派送 Run，仍由 `run_heavy.py` 配置 CPU slot、寫 log 和收據。以 `--resume-from` 指向前一份輸出。Commit 和每份輸入的 SHA-256 必須相同，否則用法錯誤，不派送 Run。`Sources.hashes` 用整份 manifest 原始 bytes 算 SHA-256，只改審閱註記也會使雜湊不同。呼叫端也要沿用來源批次的 worker 數。工具不把 worker 數寫入 journal，也不檢查是否一致。請查來源收據的 `command`：有 `--workers 3` 就沿用 3，沒有 `--workers` 就用預設的 1。

新報表併入來源中已落盤的判定列。`--select` 只派送所選範圍內尚未判定的列。`timeout` 和 `run_failed` 都是已判定列，不會自動重跑。只留用現有 scene ID 和已註冊 adapter key 都與來源列相符的結果。`missing_adapter` 不留用，選中時會再次核對註冊。留用列標 `reused`，其 `case_directory` 指向原始產物。新程序第一個送出的 Run 標 `cold start`。

下例續跑來源使用 3 個 workers 的非排他批次。保留 `output` 為來源目錄，另設定新的絕對路徑 `continued_output` 及本次 payload 的秒數預算 `budget_s`。先建立 `continued_output` 的父目錄，並確認新輸出、log 和收據路徑未被使用。若來源未指定 worker 數，改用 `workers=1`；若來源停用了 OpenCL，沿用上例的 ICD 與 cache 設定。若來源是排他的批次，在協調的窗口為 runner 加上 `--exclusive`。

```bash
workers=3
uv run --directory "$PWD" --no-sync -- python tools/run_heavy.py \
  --wait-limit 60 --log "$continued_output.log" "$budget_s" -- \
  python -m tools.experiment_wall_time --resume-from "$output" \
    --workers "$workers" --select mux --output "$continued_output" \
  > "$continued_output.receipt.json"
```

若要重跑已判定的失敗列，用 `--select` 指定該 scene ID，另開新的 `--output`，不要帶入包含該列的 `--resume-from`。若已換 commit，使用新 commit 開新批次，不留用舊列。

工具不掃描 guide、不比精度，也不執行 kernel 比較。固定 optimizer benchmark 使用獨立的數值場景和 `Engine.evolve`；這張表仍派送現行完整 Run。變更場景或 fixture（包括 projection floor）會改變工作量，不能把時間差全歸因於 simulator。見 [ADR-0078](../docs/adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)。

## Simulator benchmark

模擬器量測按層分開，不再提供 program → acquire 的端到端 benchmark。
Kernel 量測在 `Engine.evolve` 上用 `families=` 固定候選，輸入只有一段。
整體耗時改用多段 `TimelineBatch` 在 `Engine.evolve` 的序列層量測，不經 program compilation 或 ADC sampling。
現有的 family 誤差與耗時量表見下節。

量測場景的選法、速度驗收的順序與 10% 回報規則見 [模擬器維護規則](../lib/zcu_tools/simulate/maintenance.md#4-量測)。

## Shared simulator measurement

`tools.simulator_measurement` owns the session modes and wall/process clocks.
Callers provide a context factory yielding a prepared `Session(invoke, inspect,
clear_history, on_sample=None)`. `invoke(telemetry)` performs one
`Engine.evolve`. `inspect(result, telemetry)` returns caller-defined diagnostics.
The context releases its Engine and source on normal exit and on failure.
The owner does not select scenes, load references or construct models.

For serial requests, use `SampleRunner(open_prepared_session,
mode="warm_clear_history", warmups=2)` as a context manager, then call
`runner.sample(telemetry=True)`. The factory must open a fresh prepared session
on each call. Mode `first` opens and closes one session per sample without
warmup. Both warm modes retain their session and discard the initial warmups.
`warm_clear_history` clears history before each sample.
`warm_retain_history` keeps it. A request with `empty_history=True` clears once
for a diagnostic without rewarming or changing later requests' policy.
`close()` releases retained resources; a later sample opens and warms anew.

`measure_sample` times one already-prepared session without opening, warming,
clearing or closing it. It reads each clock immediately around one invocation.
Setup, warmup, history clearing, reference I/O, inspection, telemetry conversion
and sample observers stay outside both timers.
`Sample.wall_seconds` and `Sample.process_seconds` are finite, non-negative
elapsed seconds. Zero is a valid reading when a clock does not advance within
its resolution. Negative values, NaN and infinity raise Pydantic ValidationError,
a ValueError subclass. The owner records zero unchanged, without epsilon,
dropping or resampling. Whole-process CPU may exceed wall time.
These readings do not establish speed differences below clock resolution.
Family retention rests on the more-than-10% rule and on a person reviewing the data (see [ADR-0078](../docs/adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)), not on this owner.
`Sample.observation` contains the untimed inspection result.

`measure_modes` collects the family table's five samples per mode, including
its slow-family exception, and returns `ModeMeasurements` with samples and an
optional unmeasured reason. `family_pareto.measure_family` adapts
`EngineSession` to this owner, requires a `ReferenceResult`, and converts
observations to the existing report schema. Selection, whole-batch reference
preflight and the check-only exit remain in the consumer.
Neither API resets the process or Numba disk cache.
Cross-worktree comparisons require the same committed measurement tools on
both sides. Cross-worktree coordination and ABBA statistics belong to
[`paired_benchmark`](#paired-engine-benchmark), not this owner.

## Paired Engine benchmark

`tools.paired_benchmark` compares one indexed frozen scenario across two sides.
Use two worktrees to compare revisions, or the same worktree to compare Engine
backends. Both sides must contain committed copies of the common measurement tools and
codecs. Each side needs its own bootstrapped uv environment. The tool does not
copy uncommitted scripts into older commits or synchronize dependencies.
Use a common committed tool baseline for both versions being compared.
`run` is the only comparison entry point. `child` is its internal JSON service.
There is no acquire adapter or program-to-acquire timing path.

Freeze the scenario, mode, worker width, budget and K in the execution plan.
Obtain compute approval before running. The following example uses K=3 for
both calibration and timing; it is not an execution grant.
Set A, B, REF and OUT to absolute paths, then create the output directory.

```bash
mkdir -p "$OUT"
uv run --directory "$A" --no-sync -- python tools/run_heavy.py \
  --exclusive --wait-limit 120 --log "$OUT/controller.log" 600 -- \
  uv run --directory "$A" --no-sync -- python -m tools.paired_benchmark run \
    --baseline "$A" --candidate "$B" --scenario idle_coherence \
    --reference-cache "$REF" --mode warm_clear_history --workers 1 \
    --blocks 3 --exclusive --output "$OUT/report.json"
```

Only the outer runner takes resource locks. The CLI's `--exclusive` records
that caller-owned admission; it does not acquire another lock or verify it.
Every child uses its own `uv run --directory <side> --no-sync` and inherits the
outer process group and inheritable slot descriptors. Requests run serially.
A failed child stops the comparison without retry. The CLI prints the cause
and returns 1. It publishes JSON only after success, using an atomic replace,
so a failed comparison does not overwrite an earlier report.

`--index PATH` chooses one index for both sides. Without it, each side uses its
tracked default indexes. `--scene-cache DIR` supplies an absolute numeric-cache
override; otherwise the tools storage owner resolves the repository-shared
cache. `select_inputs` supplies the original `SelectedInput` identity.
Reference preflight uses the public solver-free codec. A miss fails before any
Engine construction or evolve. This command never builds a scenario, solves a
reference, fills a missing numeric cache or rewrites a reference key.

The handshake compares `PROTOCOL_VERSION`, `FROZEN_PHYSICAL_VERSION`,
`SCENE_CACHE_VERSION` and the ordered per-record identity of name, physical hash,
oracle hash and metadata hash. The protocol version covers the wire format and
tools declaration interpretation.
The record identity includes the metadata hash because companions and companions_by_target decide the observation whitelist and expected_routes decides attribution.
The physical version covers decoding fields,
units, dtypes, defaults and Engine-input reconstruction. The cache version
covers index JSON and the cache-directory revision. `SEGMENT_NUMERIC_VERSION`
is provenance only, so numerical optimizations can change it. Every fresh
child must match that side's first handshake before its first sample.

`--families NAME ...` fixes the Engine candidate whitelist exactly as supplied.
Omitting it keeps default routing. The tool does not infer a different physical
workload or append companions. `--budget` defaults to 1e-3 and `--workers` to 1.
`--baseline-backend` and `--candidate-backend` each accept `auto` (default),
`cpu` or `opencl`. To measure the automatic GPU route against the CPU on the
same checkout, give both `--baseline` and `--candidate` that path and set
`--baseline-backend cpu --candidate-backend auto`. Keep the scenario, mode,
workers, families and budget identical. Worker width and backend change execution,
not the reference key. Different point counts or dimensions require separately
frozen records and comparisons.

Modes describe process state as well as session history:

- `warm_clear_history` keeps one child per side, performs two untimed warmups
  when its session opens, and clears history before each timed sample.
- `warm_retain_history` keeps one child per side with the same warmup count,
  then retains history between samples. Calibration groups and final timing
  open fresh sessions within that child.
- `first` primes each side's Numba namespace in a discarded same-workload child.
  Every measured sample then opens a new child and a new session. It measures
  the first evolve in that process with disk-cache loading, not process import.
  Each first child returns one sample regardless of its duration. Pareto's
  slow-family extra warm sample applies only to the table protocol.
- `cold` does not prime. Each measured child gets a new empty `NUMBA_CACHE_DIR`
  parent. The controller deletes that parent after the child exits, including
  on failure. It never deletes an existing user cache.

All modes delegate session transitions and wall/process clocks to
`simulator_measurement`. Only one `Engine.evolve` is timed per sample.
Process startup, imports, source/Engine construction, reference I/O, warmup,
history reset, density errors and telemetry conversion stay outside the timers.
Disk-cache priming and owner warmups are separate operations. For an OpenCL
comparison that excludes cold builds, use a warm mode. Measure cold builds
separately with fresh OpenCL build and detection caches. Record whether the
OpenCL driver had its own persistent cache enabled. `cold` refers only to
Numba and does not reset OpenCL caches.
The timed call includes Engine dispatch, scheduling and worker management;
it is not bare-kernel timing.

Each cross-side block completes A1, B1, B2, A2. A is baseline and B is candidate.
For wall and process CPU separately, its ratio is `(B1+B2)/(A1+A2)`.
A ratio below 1 means B is faster. Zero elapsed readings remain raw samples.
If either side's block sum is nonpositive, statistics raise ValueError naming
the block, clock and both totals.
A zero baseline sum has no ratio, and a zero candidate sum is a clock-resolution artifact. Ratios feed the median, IQR and telemetry decision, so the statistics fail instead of substituting or skipping a block. The sample owner only records readings and does not make this check.
No epsilon, resampling or dropped block hides
clock-resolution limits. Summaries report the median and linear Q1/Q3 at
indices `(K-1)*0.25` and `(K-1)*0.75`.

Each side first runs off/off noise-floor blocks and off/on overhead blocks.
Calibration uses the same ratio, with A=off and B=on. Only when both sides'
wall and process overhead medians lie in their own off/off IQR closed intervals
does final timing enable telemetry on both sides.
K must exceed 2: `run_comparison` raises ValueError otherwise, because a two-block IQR is too weak to choose a mode. When any of the four medians lies outside its interval, both timing sides disable telemetry and each adds one empty-history telemetry-on diagnostic in the same execution. Those diagnostics never enter timing ratios.
At K=3, calibration plus comparison requests 60 samples before optional extra
diagnostics. Warm sessions also run 12 untimed warmups in total.

`BenchmarkReport` schema 1 preserves both sides' HEAD/tree/dirty facts,
interpreter and effective Numba cache, all raw blocks, calibration decisions,
worker/budget/family/backend coordinates, observed CPU affinity/native-thread settings
and the caller's exclusive declaration. Every observation includes ordered
points, the complete complex density, same-coordinate oracle error, reference
key and convergence error. `density_errors` compares complete cross-side
matrices per block, never only populations or a truncated block. A same-dimension
oracle does not establish truncation convergence.

Available family routing reports retain aggregate `prepare_cpu_s`,
`apply_cpu_s`, logical `rows` and `cache_hits`. The CPU fields measure the
calling thread, not Engine wall or process CPU. `dop853.opencl_rows` and
`dop853.opencl_detection_error` identify GPU execution and auto fallback.
`dop853.dp54_rows` and `dop853.dop853_rows` count physical endpoints by
tableau across CPU and GPU. Missing accounting remains unavailable, not zero.
Across different frozen N coordinates, time differences estimate marginal cost.
Keep nonlinear points separate. `apply_cpu_s/rows` is a logical-row amortization,
not the cost of one apply call. Diagnostics from an extra sample are not a
breakdown of a telemetry-disabled timing sample.

## Engine family Pareto table

`family_pareto.py` compares every family on the same complete physical inputs.
It times **Engine.evolve**, not `program.acquire`. The program-to-acquire
benchmark is retired; sequence-level measurements exclude compilation and ADC sampling.

The producer declares fixed carrier, gain, length, idle and gate-sequence scans,
including Gaussian gates, biased inputs, overlapping tones and a joint FTF scene.
The synthetic inventory contains the three-level `idle_coherence` scene.
Accuracy/guard grids and amplified test workloads stay in tests.

`select_inputs(index=None, scenario=None, *, scene_cache=None)` is the single public source and
report-identity seam. It reads numeric records through the simulator codec,
without importing the scene producer or rebuilding device models. Each returned
`SelectedInput` contains an `Input`, its `FrozenRecord` and the metadata digest.
`Input.build()` creates a fresh source that the caller closes.

By default it reads `tools/data/family_pareto/frozen-synthetic.index.json` and
`frozen-real.index.json`. With neither `--index` nor `--scenario`, it requires
and selects the 49 names in `PARETO_DEFAULT_NAMES` beside `select_inputs`.
It rejects missing default names after checking all indexes for duplicates,
before reading any payload or reference. Other rows may share those indexes;
the default selection does not read their payloads or references.
`--index PATH` without a scenario selects every row in that index.
`--scenario NAME` selects exactly one row from the supplied index or both default
indexes, including rows outside the Pareto set. Adding a Pareto scene requires
both its index row and an explicit addition to the name set. Profile rows do
not provide evidence for family retention or removal. Numeric records live
in the repository-shared `<git common dir parent>/.agent_state/benchmark/family_pareto/scenes/v1`.
`--scene-cache DIR` requires an absolute override. Tools resolve Git locations;
simulate only receives explicit paths. No caller should infer scene identity
from an index location or file hash. Cross-index duplicate names always fail.
Reference loading is cache-only for every selection.

### Recorded scenario summaries

The first thirteen scenarios originally ran at clean commit
`3545520dd8d5e8784be642246605f263654ce72b`, tree
`c9f484253fae2cc0db5a7d80dc1dd21bbe808d77`. Each table ran once with exclusive
access, all 17 pinned families, default routing and its fixed observed union.
The A gain-sweep record now uses the remeasurement described after the table.
The generated summaries retain over-limit errors and refusal reasons.
JSON is the numeric source; the paired Markdown lists the same outcomes and provenance.
These records contain no dominance conclusions. The
[kernel evidence table](../lib/zcu_tools/simulate/evolution/README.md#recorded-real-workload-evidence)
lists the qualified candidates' non-dominated workloads and evidence limits.
Compare times within one table, not across scenarios or measurement runs.

| Scenario | Generated summary | Numeric source |
| --- | --- | --- |
| F carrier sweep | [Markdown](data/family_pareto/summaries/carrier_sweep.md) | [JSON](data/family_pareto/summaries/carrier_sweep.json) |
| A gain sweep | [Markdown](data/family_pareto/summaries/gain_sweep.md) | [JSON](data/family_pareto/summaries/gain_sweep.json) |
| L length sweep | [Markdown](data/family_pareto/summaries/length_sweep.md) | [JSON](data/family_pareto/summaries/length_sweep.json) |
| I idle sweep | [Markdown](data/family_pareto/summaries/idle_sweep.md) | [JSON](data/family_pareto/summaries/idle_sweep.json) |
| N gate sequence | [Markdown](data/family_pareto/summaries/gate_sequence.md) | [JSON](data/family_pareto/summaries/gate_sequence.json) |
| M simultaneous drive | [Markdown](data/family_pareto/summaries/simultaneous_drive.md) | [JSON](data/family_pareto/summaries/simultaneous_drive.json) |
| S shaped gate sequence | [Markdown](data/family_pareto/summaries/shaped_gate_sequence.md) | [JSON](data/family_pareto/summaries/shaped_gate_sequence.json) |
| O1 overlapping constant tones | [Markdown](data/family_pareto/summaries/overlapping_constant_tones.md) | [JSON](data/family_pareto/summaries/overlapping_constant_tones.json) |
| O2 overlapping shaped tones | [Markdown](data/family_pareto/summaries/overlapping_shaped_tones.md) | [JSON](data/family_pareto/summaries/overlapping_shaped_tones.json) |
| N-long 50/100/200 gates | [Markdown](data/family_pareto/summaries/long_gate_sequence.md) | [JSON](data/family_pareto/summaries/long_gate_sequence.json) |
| S-long 50/100/200 shaped gates | [Markdown](data/family_pareto/summaries/long_shaped_gate_sequence.md) | [JSON](data/family_pareto/summaries/long_shaped_gate_sequence.json) |
| F carrier sweep at 0.45 Φ0 | [Markdown](data/family_pareto/summaries/carrier_sweep_bias045.md) | [JSON](data/family_pareto/summaries/carrier_sweep_bias045.json) |
| N gate sequence at 0.45 Φ0 | [Markdown](data/family_pareto/summaries/gate_sequence_bias045.md) | [JSON](data/family_pareto/summaries/gate_sequence_bias045.json) |

The eight-point random constant-gate table uses clean commit
`4e64d9abfcaf04f066c8d8f82672eaa2378c8e0f`, tree
`5add8d093d14ae99097a7d839faad6fda57fa1aa`. It ran once with exclusive
access and the same 17 pinned families, default routing and its fixed observed
union. It records `single_tone_rwa`, default routing and the fixed union as
`over_limit`.

The A gain-sweep and eight-point random shaped-gate tables each ran once again
with exclusive access at clean commit
`7743f24a78ced9a632f7984d521f10e840155cba`, tree
`977db41f067d4b092c0eb6bd0bc44746f6401658`. Both remeasurements use the same
complete policy coverage and cached references. `dop853` qualifies at
`4.14019e-6` on A and `2.88700e-5` on random shaped gates. These replace its
earlier `over_limit` records, which reflected unevolved densities. Other
policies retain their statuses and final errors.

| Scenario | Generated summary | Numeric source |
| --- | --- | --- |
| Random constant-gate sequence | [Markdown](data/family_pareto/summaries/random_gate_sequence.md) | [JSON](data/family_pareto/summaries/random_gate_sequence.json) |
| Random shaped-gate sequence | [Markdown](data/family_pareto/summaries/random_shaped_gate_sequence.md) | [JSON](data/family_pareto/summaries/random_shaped_gate_sequence.json) |

References remain in the shared local `references/` directory described below.
Complete matrices, diagnostic reports and journals remain under the corresponding
local scenario directories. They survive task cleanup and are not tracked.
The tracked summaries can be read without those local files. Remeasurement needs
the reference cache; clearing it or moving to another machine requires reference
preparation after cost approval.

### Single-scenario remeasurement

`--scenario NAME` requires exactly one matching input in the selected records.
An unknown name fails with legal names; duplicate names fail with their source
declarations. Selection failure starts no policy or Engine. The option can be
combined with `--index`. Selection reads indexed numeric records, not builders.
Missing scene files list the name, hashes and scene-producer command, then fail
before policy or Engine work. Selection never rebuilds scenes or solves references.

Every measurement first loads all references from `--reference-cache DIR`.
A miss lists every missing input name and oracle key, then fails the whole batch.
No already-cached input starts timing before that preflight succeeds.
`--check-references` uses the same loader and writes a JSON report to `--output`.
Each entry records `name`, `physical_sha256`, `oracle_sha256`, `metadata_sha256`,
`key`, `convergence_error`, `wall_seconds` and `cpu_seconds`. Success returns 0 without measurement; a miss returns nonzero.
The checker neither solves references nor fills the cache.

Reference production belongs to the tests-side entry point described under
[Freeze numeric scenes](#freeze-numeric-scenes). Solver work needs separate
cost approval. The consumer has no `--prepare-references`, `--freeze-inputs`,
`--include-synthetic` or `--reference-cache-only` compatibility flags.
Cache-only loading is unconditional, not an option that callers can disable.
`run_table` has no solver injection.

For a supplied scenario, existing output-path defaults, summary publication and
comparison remain unchanged. This transition does not define new cache roots,
sharing rules, rebuilding or invalidation policy. The runner's `--exclusive`
acquires the lock; the inner `--exclusive` only records the caller's declaration.
The tool records UTC start time, HEAD, tree and dirty state without checking the
lock or reading the runner receipt. These annotations do not decide accuracy.

`--summary-output <path>.json` writes one current-run summary and a generated sibling
`.md`, creating parents as needed. Without it, the tool writes `summary.json` beside
the complete matrix. To update the tracked record, point it at
`tools/data/family_pareto/summaries/<Input.name>.json` and review the whole replacement.
There is no exclusive, clean-tree or all-family write gate, and no policy-by-policy
merge of different runs. `--families` still produces a partial measurement.

`family_pareto_summary.py` owns the summary schema, builder, JSON/Markdown output and
comparison. JSON is the numeric source. Each pinned policy, default policy and
fixed observed family group retains its status, final density error, three history
modes' wall/process CPU medians, sample counts and unmeasured reasons. Missing values
remain `null`. Refused policies have zero samples in every mode and the refusal as
the unmeasured reason. Whitelists, companions and attribution identify group cost;
these are not isolated target timings. Provenance includes HEAD/tree/dirty, tool UTC
start time and the exclusive annotation. Manifest/physical/oracle hashes and
`measured_families` identify inputs and coverage. The summary stores no dominance.

`FamilyResult.status` owns the five outcomes and the final-error limit. They are
`qualified` for error ≤ 2e-3, `over_limit`, `guard_or_precondition`, `not_candidate`
and `companion_blocked`. Companion refusal takes precedence over target refusal.
Reports and summaries read this property. Missing public refusal facts fail instead
of guessing a classification from text.

The command compares against the tracked scenario JSON if available.
`--compare-summary <path>.json` selects another baseline. It reads the baseline
before writing, including when both paths are the same. Missing or invalid baselines
print a diagnostic but do not block the new summary. Different input fingerprints
mark the runs not comparable. Partial runs compare only common measured families,
plus separately present default/fixed groups, and list families missing from either
run. Output lists status/error pairs, mode values and numerical time deltas, plus
version, dirty, exclusive and coverage changes. These differences do not establish
cross-run speed or dominance.

Historical full matrices are not converted into this summary schema. New records
require new measurements. Preparing references or running the example commands still
requires the corresponding cost approval.

### Inventory-wide manifest workflow

The command sequence is check first, measure second. Omit `--index` and
`--scenario` to select the Pareto default set from both tracked indexes.
Select one index and an absolute numeric cache override when comparing every
row of a separately prepared scene set.
From the worktree root, after obtaining the relevant cost approval:

```bash
uv run --directory <worktree> --no-sync -- \
  python tools/run_heavy.py --log <results>/check.log <check-timeout-seconds> -- \
  python -m tools.family_pareto --index <selected-index> --scene-cache <absolute-scene-cache> \
  --reference-cache <reference-cache> --check-references \
  --output <results>/reference-check.json > <results>/check-receipt.json
uv run --directory <worktree> --no-sync -- \
  python tools/run_heavy.py --exclusive --log <results>/run.log <timing-timeout-seconds> -- \
  python -m tools.family_pareto --index <selected-index> --scene-cache <absolute-scene-cache> \
  --reference-cache <reference-cache> --partial-output <results>/attempt.jsonl \
  --output <results>/matrix.json --summary <results>/summary.md \
  > <results>/receipt.json
```

Measurement repeats the complete preflight; the earlier check is not permission
to skip cache loading. Neither command solves a miss. Use the separately approved
tests-side producer to prepare references. The commands grant no compute budget.

`--families single_tone_rwa,floquet_period` selects pinned policies while retaining
default routing and its fixed observed family group. Omitting it measures all 17
families. Unknown names fail with legal names. The option changes neither physics
nor reference keys. It cannot be combined with `--check-references`.
The matrix records `measured_families`; partial reports identify omitted families
as unmeasured, not unavailable. Dominance compares only measured pinned policies.
A partial table does not justify deleting a family.

`family_pareto_inputs.py` owns the producer's named `Scene` and `Input`
records. `Input.build()` returns a fresh source that the caller closes.
The synthetic inventory contains only the explicitly declared `idle_coherence`.
The 115 other borrowed accuracy, guard, sweep, history and amplified-workload
inputs are no longer measurement scenes. Their tests-side owners remain intact.
Oracle caching excludes only the model's nonphysical identity namespace.
Reports store each selected record's `{name, physical_sha256, oracle_sha256,
metadata_sha256}` plus HEAD/tree/dirty provenance, not a copy of numeric records.
The public codec computes both physics hashes and `digest(record.metadata)`.
`SelectedInput.identity` exposes those same fields to paired consumers, so source
migration does not change the handshake. Full argv remains execution provenance,
not scene identity. The unchanged summary schema retains its `manifest_sha256`
field name; its value is now the digest of the ordered record identities.

Each pinned family and the separately listed default policy has three modes.
Normal modes keep five samples. If a family's first empty-history evolve
exceeds 60 wall seconds, it is the only first-evolve sample. No separate probe
or warmup runs. Clear-history is unmeasured with n=0, an explicit reason and
None times. Retain-history immediately runs once in that same session, n=1.
The JSON retains every sample. Every mode lists its count, including zero.
Dominance and timing equivalence use only shared measured modes.

The modes are:

- `first_evolve` creates a fresh source/history/Engine for each of five samples.
- `warm_clear_history` creates another session, runs two untimed warmups, then
  clears history before each of five samples.
- `warm_retain_history` creates another session, runs the same two warmups, then
  retains history for five samples.

Construction, cleanup, oracle solves, warmups, history clearing and result
inspection are outside the wall and process-CPU timers. All samples run in one
process with one Engine worker. Disk caches remain unchanged. The exclusive runner
sets native thread counts to one and supplies the HEAD/tree/dirty receipt.
The matrix also records those Git facts, libraries, exact argv, input hashes,
all raw samples, per-sample paths, density errors and medians.

The oracle is the independent full-Lindblad timeline solve at relative/absolute
controls `2e-10/2e-12` and `2e-12/2e-14`. Its two density results must differ by at
most `2e-6`. Every solver's error is measured, including exact families. Values
above `2e-3` remain FAIL. Existing strict xfails are not removed.

Only policies whose final density error is at most `2e-3` participate in
dominance. This error is the maximum after each complete pulse sequence across
all retained modes, not a per-kernel error or segment budget. Qualified policies
are equally accurate. A competitor dominates only when its median time is
**less than `0.9 * target_time`**. Exactly 10% is time-equivalent.
`t4`, `adiabatic_map`, `flux_dop853` and `slow_envelope` have exclusive domains;
`t6` and `dop853` are their domains' last candidates. These six families are
directly retained and are neither dominators nor dominated targets. Refused,
companion-blocked and over-limit policies provide no dominance evidence.
Time-equivalent peers have neither direction faster by more than 10%, regardless
of error. Wall/CPU and all three modes remain separate; there is no weighted score.
Default routing is shown but never treated as a pinned-family competitor.
The summary lists each candidate family's non-dominated input/mode/metric cells
and the full measured matrix with dominators and time-equivalent peers. An
all-dominated conclusion applies only to measured qualified cells, not to
unmeasured physics. Before deleting a more accurate but slower candidate, its
faster dominator must also qualify on the same type of long sequence, with far
more than 24 gates. Neither these results nor long-sequence qualification
authorize deletion without user approval.

Only `NoValidKernel` from the first timed evolve becomes `unavailable`.
The matrix preserves its complete message and leaves error/timing absent.
The JSON `refusal` records the first failing group's domain, caller points,
segment rows and final-round `attempted` families. For each whitelisted family,
`family_reasons` is `guard_or_precondition` if it appears in `attempted`, or
`not_candidate` if it does not. This includes named companions. The Markdown
cell shows the same classification. No later segment group is classified.
Candidate offer order is not necessarily first-attempt order. Unrestricted
default routing preserves the group facts without per-family classification.
A timed result within `1e-9` of its initial density at any point stops the
command with RuntimeError if the reference moved more than `2e-3` at that
point. This check runs after the clocks stop. An unchanged reference allows
an unchanged result.
Other RuntimeErrors, wrong pinned paths, numerical failures and output failures
stop the command. Measurement failures leave earlier outputs untouched. Output
write failures return nonzero and may leave a partial destination file.
`measure_family`, `measure_policy`, `run_table`, `dominance`, `timing_equivalents`
and `main(argv)` are importable. Comparators pair modes by name, not list order;
every available pinned row must have all three unique modes and measured sample
errors. Missing mode/error information raises ValueError. Unavailable and default
rows are not competitors. `measure_family` delegates the Engine-only timing protocol
to `simulator_measurement.measure_modes`. It requires a real reference.
Tests use controlled clocks and small
collaborators; the complete matrix runs outside pytest through the exclusive runner.

### Dedicated DOP853 segments

`dop853_segment.dop853_segment_input(Dop853Coordinates(...))` builds
simultaneous two-tone Engine inputs. It does not use acquire, append readout,
change `single_model_input`, or join the default family-table inventory.
The factory validates settings before `SingleFluxoniumSource.model_facts`
rebuilds the retained model and Gibbs initial density. It uses the fixed single
model with bias .45, two charge ports and no fast flux. Levels above the fixed
native cutoff raise `ValueError`; no prebuilt matrix is sliced.

```python
from contextlib import closing
from zcu_tools.simulate.evolution import Engine
from tools.dop853_segment import Dop853Coordinates, dop853_segment_input

item = dop853_segment_input(Dop853Coordinates(envelope="sampled", repeats=2))
with closing((case := item.build()).source), closing(Engine(workers=1, families=("dop853",))) as engine:
    result = engine.evolve(case.source, (case.batch,), telemetry=True)
```

Run numerical measurements through `run_heavy.py` under the applicable compute
grant. Importing or constructing inputs does not solve references or evolve.
Coordinates select `gaussian` or non-Gaussian `sampled` envelopes, support in ns,
retained `levels`, sample `knots` including both endpoints, `repeats` in
1/2/20/100, and independent `points`. Both tones have the same support. Every
sample is a linear-interpolation knot. Gaussian centers differ within that
support. Flat-top plateaus are outside this input. Each repeated pair restarts
its pulse-local carrier phase, with no gaps or companion family. End time is
N times the support. N does not multiply the point count. For multiple points,
only Pump gain varies; Engine worker count is a separate execution setting.
Change one coordinate at a time to isolate duration, dimension or knot density.
The controls are benchmark inputs, not calibrated STIRAP transfer.

```python
from tools.family_pareto_inputs import freeze_records

records = freeze_records([item])
```

`freeze_records(inputs: list[Input]) -> tuple[FrozenRecord, ...]` snapshots the
ordered declarations into numeric records with tool-owned metadata, including
the versioned coordinates. It closes each built source without evolving; source
arrays remain in the returned records. Builder and serialization failures
propagate. The simulator manifest writer calculates the physical/oracle hashes
and owns the container; this call does not write a manifest or solve references.
`zcu_tools.simulate.evolution.reference_key(case.source, case.batch)` returns the
persistent reference key without disk access or integration.
`Input.name` and its built `Scene.name` identify all coordinates:
`dop853_segment-<envelope>-d<duration_ns>-l<levels>-k<knots>-r<repeats>-p<points>`.
Integer durations omit the decimal part; 20.5 ns becomes `d20p5`. Names do not
enter the physical, oracle or metadata hashes, or the reference key.

`from tools.dop853_segment import profile_inputs` exposes the declared profile
catalogue. Calling `profile_inputs()` builds seven Gaussian and seven sampled
inputs. Each envelope has the d20/l8/k9/r1/p1 baseline, then single-axis variants
for duration 40 ns, levels 20, knots 17, and repeats 2/20/100. The final input is
`dop853_segment-gaussian-d20-l8-k9-r1-p64`, a 64-point workers candidate.
This call neither times the inputs nor prepares their references. The measured
Gaussian worker results and their limits are recorded in the
[DOP853 route summary](../docs/discussions/2026-11-17-dop853-speed-route-summary.md#多-worker-的配對結果).
These research declarations remain outside the default family-table inventory;
the measured sample does not change production defaults.

`profile_inputs(scenario=NAME)` selects one declared profile by its complete
coordinate-derived name before constructing any model. Unknown names raise
`ValueError` with legal profile names and construct no models. Omitting the
argument retains the complete catalogue in the order above.

The scene producer's `make_real_inputs()` includes these profiles after the fixed
scans. Its existing `--scenario NAME` route constructs only the selected Input;
fixed scan names keep their direct routes. `main(argv, inputs=[item])` still
accepts caller-owned declarations. Neither this wiring nor `profile_inputs()`
prepares scene files, reference entries or real-index rows. Each consumer must
prepare its selected scene and reference caches explicitly; the recorded Gaussian
results do not imply that every consumer's caches are ready.

After separate compute approval, freeze a selected profile through the existing
producer with `--update-index --scenario NAME`, choosing the shared real index
and approved scene-cache destination. Until those explicit preparations finish,
a complete prepare fails the exact-name-set check against indexes lacking the
profile rows. No profile row is added to the Pareto default selection.
Independent reference preparation belongs to the tests-side producer described
below. Its interaction-picture oracle stops at knots and checks two tolerances.
During measurement, use the public `load_reference(cache_dir, case.source,
case.batch)`; misses raise instead of solving. Worker count does not enter the key.
N and points do.
Same-dimension reference agreement does not establish retained-level convergence.

Measure wall/process only around `Engine.evolve`, keeping construction,
reference loading and density comparison outside. These are whole-Engine times,
not bare kernel costs. Report public `prepare_cpu_s`, `apply_cpu_s`, `rows` and
`cache_hits` separately. `apply_cpu_s / rows` is a logical-row amortization,
not one apply call's cost. Missing diagnostics are errors, not zeros.

### Fixed scenarios, companions and reference cache

`family_pareto_scenarios.py` owns curated core pulse sequences and the scene
producer CLI. `family_pareto_scene_builders.py` owns shared single/integer model
construction, public program compilation, source detachment and support-based
route declarations. The compiled FTF scene retains its own joint-model composition.
`family_pareto_gate_scenes.py` owns numeric FTF gate templates and contiguous
sequences, separately from program compilation. Its `numeric_gate_sequence`
accepts a source, named ordered DriveTrack tuples, playback order and a constant
flux level. It preserves absolute carrier phase and pulse-local envelope times.
`make_gate_inputs` supplies fixed-model repeated pairs and gate sequences;
the scene producer CLI selects these through `make_real_inputs`. Source
snapshots reuse `numeric_builder`; cache/index handling stays in the existing
producer and storage owners.
`family_pareto_storage.py` owns benchmark and scene-cache locations, plus
cross-index selection policy. Its `local_data_directory()` queries Git from the
process working directory and creates the shared benchmark directory. Git and
filesystem errors propagate. `scene_cache_directory()` instead queries Git from
the tool's worktree, accepts an absolute override, and creates no directory.
`family_pareto.py` imports the benchmark resolver directly from this owner.
`single_model_input(name, emit, manifest, families)` in the builders module is
the public single-model construction entry point.
`family_pareto_settings.py` owns fixed single-fluxonium, integer-fluxonium, FTF and board inputs,
copied from commit `303074e166d1a563addfab4fd94717d958a7dd56`. The manifest
records their source commit and fixed values. Runtime construction does not import
lab experiments, read lab YAML/profiles, execute experiments, or use replay.
Model/template changes require a deliberate update here. The fixed FTF values
do not track a user's working-copy edits. The tool overrides only FTF
`retained_levels` from 16 to 32. Local levels remain 4×4×4; all other physical
and board parameters retain that source commit's values. At the specified CZ
working point, the conditional-phase comparison needed at least 25 joint states
to reach a difference within 0.5°. Retaining 32 leaves margin. This is not a
convergence claim for arbitrary CZ pulses or the full native/local model.
FTF references and timings must be rebuilt at 32 states; 16-state results do not
apply.

The single-fluxonium scans use the full 21-level model. Their carrier center is
DC E1−E0 from the public model at each scan's fixed bias. Gains, phases and durations are illustrative
benchmark settings, not fitted calibrations or calibrated pi/pi/2 rotations.
Every shot ends in readout at the fixed bare cavity frequency, gain .1, with a
1 us pulse/window and fixed flight time. The Engine batch ends at readout start,
not at the end of that window.

| Scan | Core sequence | Requested axis (24 points unless stated) | Envelope and pulse lengths | Compiled logical spans per point | Reason |
| --- | --- | --- | --- | --- | --- |
| F `carrier_sweep` | One drive, terminal readout | DC E1−E0 ±100 MHz | Constant, 1000 ns, gain .1 | 2 | Frequency reuse without exact duplicate carriers. |
| F-bias045 `carrier_sweep_bias045` | Same as F at 0.45 Φ0 | Biased DC E1−E0 ±100 MHz | Same as F | Same as F | Compares the same carrier offsets at a second DC bias. |
| A `gain_sweep` | One drive, terminal readout | Gain .01 to .1 | Gaussian, 100 ns, sigma 20 ns | 2 | Shaped-family input at fixed envelope and carrier. |
| L `length_sweep` | One drive, terminal readout | Length 20 to 400 ns | Constant, gain .1 | 1 or 2 | Changes support without also resampling a Gaussian. |
| I `idle_sweep` | X-phase gate, idle, Y-phase gate, terminal readout | Idle 20 to 400 ns | Two constant 20 ns gates, gain .1 | 4 | Shared initial gate with changing gap and final absolute phase. |
| N `gate_sequence` | Prefix of repeated X,Y,-X,-Y phase alphabet, terminal readout | Gate count 1 to 24 | Constant 20 ns gates, gain .1 | 2 to 35 | Ragged sequences with a reusable alphabet, without random seeds. |
| N-bias045 `gate_sequence_bias045` | Same as N at 0.45 Φ0 | Gate count 1 to 24, carrier at biased DC E1−E0 | Same as N | Same as N | Compares the same compiled gate prefixes at a second DC bias. |
| M `simultaneous_drive` | Synchronous F1/F2 drives, terminal readout | Shared gain .01 to .1 | Two Gaussians, 100 ns, sigma 20 ns | 2 | Independent joint-model drive operators under overlapping shaped controls. |
| S `shaped_gate_sequence` | Prefix of repeated Gaussian X,Y,-X,-Y gates, terminal readout | Gate count 1 to 24 | Gaussian 100 ns gates, sigma 20 ns, gain .1 | 2 to 47 | Reuses N's alphabet and ragged prefixes with A's shaped-gate support and width, separating shape from sequence reuse. |
| `random_gate_sequence` | Fixed random primitive phase prefixes and recovery phases, terminal readout | 8 fixed samples from 20 seeds; selected depths [0,1,2,4,8,16,32,128], 8/20 seeds | Same constant gates as N | Public support partition, including one zero-gate point | Measures phase-dependent sequence error without replacing N's cyclic alphabet. |
| `random_shaped_gate_sequence` | Same fixed phases as random_gate_sequence, with Gaussian gates | Same 8 seed/depth points | Same Gaussian gates as S | Public support partition, including one zero-gate point | Separates shaped-gate reuse from the fixed cyclic phase order. |
| N-long `long_gate_sequence` | N's constant X,Y,-X,-Y prefixes, terminal readout | Gate counts 50, 100, 200 (three points) | Same constant 20 ns gates and gain .1 as N | Public compiled support partition, including alignment gaps and readout lead | Checks whole-sequence accuracy beyond 24 gates without replacing N. |
| S-long `long_shaped_gate_sequence` | S's Gaussian X,Y,-X,-Y prefixes, terminal readout | Gate counts 50, 100, 200 (three points) | Same Gaussian 100 ns gates, sigma 20 ns and gain .1 as S | Public compiled support partition, including alignment gaps and readout lead | Checks shaped-gate reuse at long sequence lengths without replacing S. |
| O1 `overlapping_constant_tones` | Fully overlapping constant Pump/Stokes pair, terminal readout | Pump fixed at DC E1−E0; Stokes scans that center ±50 MHz | Both 1000 ns; gains .02/.002 | 2 | A carrier slice of the dual-tone reset scan on one qubit, without a cavity drive. |
| O2 `overlapping_shaped_tones` | Stokes-first Gaussian pair on one qubit, terminal readout | Stokes lead .3 to 0 us | Both supports .3 us, sigma .05 us; Pump 7241.848921606473 MHz/gain .030001831166453; Stokes 3111.1686681315896 MHz/gain .100006103888177 | Public support partition includes single-tone, overlap and idle spans | Uses the STIRAP delay sequence without artificial inventory repeats. |

The numeric FTF gate declarations use the same fixed model at flux
F1/coupler/F2 = .48/.20/.49 Phi0, local4x4x4 and 32 retained joint states.
Gaussian, DRAG, sampled and flat-top X90 pairs last 100 ns.
`multitone_gate_{gaussian,sampled,drag,flat_top}_n{1,21,100}` repeats one aligned
pair, for 12 kernel scenes. A sequence tail uses the same role's pulse.
`multitone_depth500_{gaussian,sampled}_{min_pairs,max_pairs}` mixes aligned pairs,
alternating-role single-tone tails and 200 ns dual-charge-line CZ representatives
with a fixed seed0 permutation. The two inventories contain respectively
537/1232/1436 and 661/988/932 pair/tail/CZ supports. They represent the minimum
and maximum aligned-pair counts from a three-seed, two-arm simultaneous RB
inventory, rather than replaying a lab experiment.

The gate producer records all four envelope recipes and role-specific gains
in its source. DRAG uses 33 knots, sigma25 ns and Q=-dG/dt/alpha with role-local
anharmonicities, zero Q endpoints and complex maxabs normalization, without a
2*pi conversion. The kernel flat-top uses 33 knots, 25 ns ramps with sigma6.25 ns,
a 50 ns platform and forced zero endpoints, then peak normalization without
baseline subtraction. It does not use the CZ's 257-knot envelope.
The gains target single-tone dressed population .5 from joint ground with loss
off. The actual scenes retain joint Gibbs density and fixed baths/dephasing.
The original spike working files were not committed. The CZ scale comes from
the public drive operators' coherent matrix element and the flat-top area. This
is not a calibrated CZ and does not establish conditional phase or fidelity.
The real index and default Pareto name set include all 16 gate scenes. Their
metadata omits the acquisition-only `numerics.debug_capture` setting. The index
pins identities, not local cache availability, reference convergence or
whole-sequence speed. Reference production requires its own compute approval.

`carrier_sweep_bias045_input()` and `gate_sequence_bias045_input()` keep the
fixed single-fluxonium physical and board values from the source commit above.
They override only the public `DCBias.offset_phi0` from .5 to .45. The DC source
still starts at zero, so `offset_phi0 + flux_per_unit * live_value` is .45 Φ0.
Each builder obtains that bias's model through `snapshot()` and `model_facts()`.
Its E1−E0 sets the requested carrier center before public program compilation.
The original F and N builders stay at .5 Φ0. Each pair shares its core sequence,
so the new bias retains F's 24 offsets and N's 1..24 prefixes, constant pulses,
phase alphabet, compiler alignment gaps and terminal readout lead.
The manifest records the effective settings and compiled carriers. These are
separate inputs, not replacements or calibrated rotations. Builder availability
does not establish reference convergence, family admission or timing at .45 Φ0.

M uses public timeline fork/join to start both drives together and place readout
after both finish. Its two carriers are the uncoupled local DC E1−E0 frequencies
of F1 and F2, not fitted dressed-transition calibrations. Both ports act on the
same coupled FTF density. Readout uses the fixed bare r1 cavity frequency, gain
.1, a 1 us pulse/window and fixed flight time. This benchmark sequence is not a
calibrated CZ gate. The CZ comparison above explains only the retained-state
choice, not a fidelity claim for this scan.

S uses A's 100 ns support and 20 ns sigma to compare sequence reuse at the same
Gaussian width. The support spans five sigma. These are fixed benchmark gates,
not calibrated pi rotations. The lab's generic `pi_pulse` creation template
seeds a constant pulse, so it supplies no calibrated Gaussian duration.
S adds no random gates or extra repetitions to make a family admissible.

The two random-phase scans share
[`random_gate_sequences.json`](data/family_pareto/random_gate_sequences.json).
Its phases were generated offline from the user-authored Clifford prefix and
inverse-recovery tables at commit `46a4291cfdf1b1617bc1fb03499813450b16ed7e`,
`zcu_lab/v2/twotone/rb/sequence.py`. The file records that source's SHA-256,
parent seed 42, all 20 spawned seeds, the nine depths and selected indices.
The parent axis is the seed-major Cartesian product of those 20 seeds and nine
depths. The retained parent indices are [0,15,23,85,101,147,163,179], a fixed subset
of the initial `np.linspace(0,179,24,dtype=int)` selection. These eight points use
eight seeds and depths [0,1,2,4,8,16,32,128], with 243 physical gates in total.
They keep both axis endpoints and depth128, but omit depth64 to reduce reference
cost without shrinking the model. The selection favors shorter sequences at
other depths and does not represent RB seed statistics.
Each point retains its primitive phase prefix and recovery phases separately.
Virtual Z changes the copied phase frame; identity gates and recovery padding
emit no pulse. The selected depth-zero point emits no drive. Its Engine batch
ends at readout start with zero duration.
The builders compile those phases through public Pulse/TimelineBuilder components.
They do not import lab code, run Clifford algebra or read lab YAML.
Model, carrier, gain, envelope, pulse length, initial density and companions follow
N or S. These uncalibrated gates need not return the state to identity.
The inputs measure evolution error on an RB-type phase order, not RB fidelity.
Changes to the fixed controls require review. New seed/depth sequences require offline generation.

`long_gate_sequence_input()` and `long_shaped_gate_sequence_input()` compile
N-long and S-long through the same public program and single-fluxonium model
path as N and S. The compiled-scan inventory includes both three-point scans, both bias045 scans
and both random-phase scans alongside the original nine scenarios, for fifteen
compiled scans. The numeric gate scenes are additional inputs.
Only the requested gate counts change for the long scans.
The model and board values retain source commit
`303074e166d1a563addfab4fd94717d958a7dd56`; DC bias, initial density, carrier,
phase cycle, pulse parameters and terminal readout follow N or S respectively.
Each branch emits its complete prefix without idealizing away compiler timing.
The frozen manifest declares the gate counts and complete target/`t4` rows/ns
from public compiled supports. Long-sequence reference reuse requires checking
the full physical input, point order, basis, both tolerances and convergence
contract; matching gate counts or pulse durations alone does not permit reuse.
Accuracy-only family observations are not exclusive timing samples.

O2 copies the integer model's two charge lines, Pump on generator 1 and Stokes
on generator 2. Both couple to the full charge operator at 1 GHz/gain.
The source commit's historical synthetic STIRAP controls supply the carriers.
Their former envelope gains .015/.05 convert by 65536/32766 to the gains above.
The current STIRAP delay sequence supplies the .05 us sigma/.3 us support scale.
These choices are benchmark controls, not calibrated population transfer or
new acquired-IQ validation. Readout uses generator 0/ADC 0 at 7460 MHz,
gain .1, a 1 us pulse/window and zero flight time.
The lead uses a table sweep so each requested value is quantized independently.
The endpoints remain separated and fully overlapping. Each shot has one pair;
changing the relative lead changes overlap support, rather than repeating a
canonical gate to increase reuse. Default routing amortizes the cold endpoint
and Fourier build over eligible uses in the current evolve, then adds per-row
bookkeeping and playback costs. Pinning `multitone_gate` bypasses pricing,
not structural eligibility. Once selected, the gate builds and runs without
an admission probe.

O1 takes its overlap sequence from `twotone/reset/dual_tone/freq` and its
±50 MHz scan range from that experiment's Form. The Form's actual reset
frequencies and gains come from caller-owned component ext values, not fixed
calibrations. The generic `two_pulse_reset` template supplies constant 1 us
supports and gain .2. O1 uses DC E1−E0 as an explicit uncalibrated carrier center.
The generic single-model charge line has coefficient .1 GHz/gain; the integer
line has 1 GHz/gain. Pump gain .02 preserves the .02 GHz charge coefficient
of the template's .2 gain on the single line. Stokes gain .002 gives a tenfold
weaker probe on the same operator. These coefficients are not Rabi rates.
Both pulses start together on generators 1 and 2, use equal support, and end
before the generator 0/ADC 0 readout described for O2. There is no cavity drive
before readout, so this is not an AC Stark experiment or a calibrated reset.
The small drive coefficients and nearby positive carriers give `common_frame`
and `floquet` a potential admission point. Actual numerical guards still decide
availability; the scan adds no repetitions, guard overrides or runtime fallback.
The fixed target declarations assign all driven rows/ns to the target and the
idle tail to `t4`, using public compiled supports before any Engine measurement.
Default routing and its observed-union group keep their existing protocol.

`integer_fluxonium_input(name, emit, manifest, families)` in
`family_pareto_scene_builders.py` owns the shared fixed integer settings, board,
environment, compilation and numeric-source detachment.
Its `emit(builder, dc_model)` callback receives a public `TimelineBuilder` and
DC `SecularModel`. It can emit constant or Gaussian pulses on either charge line
and use the model's GHz energies to choose carriers. The builder appends terminal
readout. The returned `Input.build()` creates a fresh source that callers close.
Callers supply `manifest["pulse_sequence"]` and requested control provenance.
The default declaration assigns driven spans to each target and idle to `t4`.
Mixed-domain callers replace that declaration, as O2 does. The same builder can
therefore compile a two-constant-charge-tone scan without another model copy.

Continuous requested axes are equally spaced and include both endpoints. Native
hardware quantization changes the realized values. Length uses a public table
sweep so each requested length is quantized as its own scalar pulse, rather than
accumulating a truncated increment. Its realized supports with this fixed board
are 20, 36, 54, 70, 86, 102, 118, 136, 152, 168, 186, 202, 218, 236, 252, 268,
284, 300, 318, 334, 350, 368, 384 and 400 ns. Other hardware values also follow
the public compiler conversion. The manifest preserves requested ranges and full
physical arrays, not a claim that every requested value is exactly representable.

Span counts include compiler alignment gaps and the no-drive readout lead. N can
have short gaps between gates even though its core requests no extra idle.
The manifest records every point's actual span count, compiled carriers,
model/board inputs, snapshot, exact tick ratio, calibration method, expected
routes and full physical hashes. Logical spans are not unique cached preparations.

A pinned policy may include named companions, such as `t4` for the idle tail.
`Input.companions_by_target` replaces the shared `companions` tuple for listed
targets. Unlisted targets keep that shared tuple; an explicit empty tuple pins
the target alone. `companions_for(target)` resolves this declaration.
Only inputs with a nonempty mapping add `companions_by_target` to the frozen
manifest. Existing inputs keep their manifest keys and physical/oracle hashes.
O2's `single_tone_gate`, `gate_cache` and `shaped_sambe` targets use
`common_frame` plus `t4`. Its `period_table` target uses `dop853` plus `t4`.
Its `common_frame`, `multitone_gate` and `multitone_rwa` targets use
`shaped_sambe` plus `t4`. `dop853` uses only `t4`.
The original `multitone_rwa` overlap companion and `single_tone_gate` single-tone
companion refused their respective spans on this fixed input. A subsequent
`multitone_gate` overlap companion also refused, so the four shaped targets
next tried `common_frame`. Overlap targets refused their own multiple-tone spans with
`shaped_sambe` as companion; that is not a reason to replace the companion.
Companion changes are input declarations, never runtime fallback choices.
After both overlap companions refused, the two named targets `shaped_sambe` and
`period_table` each tried a `dop853` companion. In the `shaped_sambe` probe,
`dop853` took 13 single-tone rows/2894 ns from the target. That group keeps its
`common_frame` companion and remains companion-blocked.
The `period_table` group requires its target to keep all 46 single-tone rows/7264 ns,
`dop853` to handle exactly 23 overlap rows/3574 ns, and `t4` to handle
25 idle rows/43.696428571428555 ns. Every sample must match the full declaration.
A mismatch stops the attempt; that group remains companion-blocked rather than
changing the expected route to the observed one. No other companion can compete
for the target's type of span.
O2 counts each span's active tone supports using public `segment_spans`, then
sums rows and ns for idle, single-tone and overlap work. It does not reproduce
the dispatch candidate table or add a fallback.
The input declares `{family: (logical segment count, cumulative ns)}` for each
target's complete group. Both quantities must match public `FamilyRouting`
reports in every sample. A count mismatch or duration mismatch stops measurement.
The duration comparison allows only floating accumulation noise
(`rel_tol=1e-12`, `abs_tol=1e-9 ns`).
Cumulative ns cannot distinguish equal-length spans. It is not per-span
observation and does not prove disjoint domain ownership.

All timing samples enable routing telemetry. The paired nonexclusive carrier
comparison found on/off wall ratios 0.984, 1.147 and 1.004 for first, clear and
retain modes. CPU ratios were 0.983, 1.147 and 1.005. Clear-history differed by
about 2 ms; this single pair does not isolate instrumentation overhead from
shared-machine variation. Telemetry remains on for all policies to collect CPU
and logical work consistently, without adding a third evolve for slow families.
Reported times are telemetry-enabled group latency; near-10% comparisons need
caution given that observation.

`tools.family_pareto_report.render_saved_summary(matrix_json)` renders the JSON
text of an existing matrix without solving references or timing evolves. It
restores the recorded result types and uses the same summary renderer. Save the
returned Markdown separately from the measured summary and record the renderer
version separately from the matrix's measurement version. Re-rendering recomputes
dominance from recorded errors and times using the current qualification rule.
It does not modify the matrix, its provenance or any raw measurement.

Each sample preserves `prepare_cpu_s`, `apply_cpu_s` and their sum separately
for every reported family. These are calling-thread CPU from `FamilyRouting`,
including preparation for rejected guards. The report shows target and companion
rows separately. Only whitelisted non-target families are companions; other
reports in a pinned policy are preparation diagnostics, including zero-row guard
CPU. All reported prepare+apply CPU remains in each share's denominator.
Default and fixed-group reports retain group-member attribution.
They are not whole-process CPU, may exclude routing/geometry
overhead, and need not add up to the process CPU timer.
Pareto dominance continues to use whole-`Engine.evolve` wall and process CPU
separately. A target plus named companions is one timed group, never isolated
target performance. Default routing is also compared with its observed family
union fixed. That extra group has no expected-route declaration or route check;
this applies to every scenario, including O2. It stays outside dominance.
Differences over 2x are findings, not routing changes.

A refusal with an attempted companion but no attempted target is
`companion_blocked`, with no `unavailable` value for the target.
The original message and first failing group's `attempted` facts remain intact.
This does not classify later groups or claim that the target's guard refused.
If a whole class of targets cannot produce a table because of the same blocked
companion, stop and request a companion decision. Do not substitute a fallback.
Each dominance relation records whether the two companion sets match.
Different-companion relations are group observations only; they cannot alone
justify deleting a target. Family conclusions compare equal-companion groups.
The summary reports each family's prepare/apply thread CPU and its fraction of
summed reported thread CPU. These fractions are not wall or process-CPU shares.

`family_pareto_reference.load_references` loads the complete selected batch with
the simulator's public `reference_key` and `load_reference`. Equal keys share one
reference object, including inputs whose model identity differs. It checks loaded
convergence errors before policy execution and lists all missing names/keys.
`run_table` and `--check-references` use this same preflight.

`--reference-cache DIR` selects an existing cache. Loading does not create the
directory or write NPZ files. Keys include complete identity-free physics, both
tolerance levels and the convergence contract. Files retain density and convergence
error without pickle. Keep the cache outside Git. Reference load totals remain
separate from Engine time. This change does not choose storage or sharing policy.

### Freeze numeric scenes

The scene producer snapshots complete numeric physics through
`freeze_records`, which returns public `simulate.evolution.FrozenRecord` values.
Scene reason, targets, companions and provenance are opaque tools metadata.
Evolution owns the index and single-record gzip codec. See
[Frozen numeric inputs and references](../lib/zcu_tools/simulate/evolution/README.md#frozen-numeric-inputs-and-references)
for the physical record version, scene-cache version and reference key.
Every frozen scene needs a tracked builder in this producer. A scene built only by an untracked prototype cannot enter an index.

Git contains only `frozen-synthetic.index.json` and `frozen-real.index.json`.
Each ordered row pins `name`, `physical_sha256`, `oracle_sha256` and
`metadata_sha256`. The last digest uses `simulate.evolution.digest(metadata)`.
The 19 existing payloads were migrated without changing identities, point order
or reference keys. Numeric gzip files are no longer tracked at the tip.
The old gzip files are not reachable from the landed history. The migration did not validate a complete cold builder rebuild.

Without `--scene-cache`, [`scene_cache_directory`](family_pareto_storage.py)
resolves `.agent_state/benchmark/family_pareto/scenes/v<SCENE_CACHE_VERSION>`
relative to the parent of Git's common directory. An explicit override must be
absolute.

Run model construction only with approved compute and the current runner/lock
policy. This command rebuilds and checks the pinned inventory, outside measurement
preflight:

```bash
uv run --directory "$PWD" --no-sync -- python -m tools.family_pareto_scenarios \
  --index-dir tools/data/family_pareto --scene-cache /absolute/scene-cache
```

`main(argv, *, inputs=...)` retains explicit producer declarations. Its default
inventory includes the [DOP853 profile catalogue](#dedicated-dop853-segments).
The staged `multitone_depth50_{gaussian,sampled,drag,flat_top}_max_pairs` prefixes
and `multitone_depth500_{drag,flat_top}_max_pairs` inventories require explicit
`--scenario` selection and a separate index and scene-cache directory. The four
prefixes take the first 258 complete gates of the same seed-0 permutation of
2,581 gates, not 50 circuit layers. The flat-top X90 uses the complete sampled
pulse, including both ramps. Adding these declarations does not populate the
tracked indexes or shared cache.
`prepare_scenes(inputs, indexes, scene_cache, *, scenario=None, update_index=False)`
owns preparation. `indexes` maps categories to explicit index paths.
The producer fixes and logs OMP, OpenBLAS, MKL and Numba native thread settings
at one. Complete preparation checks each category's exact name set before
snapshotting, even with `--update-index`. `--scenario NAME` processes one row.
For a new name already declared in the inventory but absent from both indexes,
pass `--update-index --scenario NEW`. The producer builds and caches that scene,
then appends its identity to the tail of its declared category's index.
Without `--update-index`, a missing row fails. Unknown declarations, duplicate
selections and category moves fail before building. For example, after adding
the tracked builder declaration and obtaining compute approval:

```bash
uv run --directory "$PWD" --no-sync -- python -m tools.family_pareto_scenarios \
  --index-dir tools/data/family_pareto --scene-cache /absolute/scene-cache \
  --update-index --scenario NEW
```

Diagnostics distinguish unindexed names, existing files, missing files, rebuild checks and hash
mismatches. A mismatch fails without replacing the old pinned file or index.
Missing files are filled only after the rebuilt identity matches the pin.

Only `--update-index` changes selected identities. Every selected builder must
succeed before any index is published. Unselected rows retain their order and
identity. Earlier numeric files can remain if a later builder fails.
Preparation never infers deletion or category moves from missing declarations.
For those changes, explicitly edit the tracked declaration and index together.
Changed physics may also require separately approved reference production.
No cache clear, model rebuild or reference solve is implied by this document.

The single-support `kernel_single_constant`, `kernel_single_sampled` and
`kernel_multitone_sampled` scenes use public `SingleFluxoniumModel` directly.
They are separate from the complete F/L/I/N sequences, which retain 24 points
and 21 levels each. See the [data README](data/family_pareto/README.md).
`family_pareto_multitone_scenes.py` declares fourteen constant-pair price
controls: three INTEGER_MODEL C2b projections, a nine-point fixed-M synthetic
feature star and consecutive N=20/100 base supports. Their names join the
Pareto default set. Each new row is appended with `--update-index --scenario`;
only hashes enter Git. See the [data README](data/family_pareto/README.md) for
coordinates and the distinction between repeated supports and exact-key reuse.

The shared cache is not task scratch. Task cleanup must not delete it.
Manual cleanup is confined to the scene-cache prefix.

Independent reference production stays in tests. See
[Reference solver limits](../lib/zcu_tools/simulate/evolution/README.md#reference-solver-limits)
for the bath inputs both solvers use. After separate solver-cost approval,
supply explicit index-to-solver mappings and a scene cache:

```bash
uv run --directory "$PWD" --no-sync -- python -m tests.simulate.evolution.contract.family_pareto_reference_producer \
  --converged tools/data/family_pareto/frozen-synthetic.index.json \
  --interaction tools/data/family_pareto/frozen-real.index.json \
  --scene-cache /absolute/scene-cache --cache /path/to/approved/reference-cache
```

Add `--scenario NAME` to solve only that row. The reference producer requires
exactly one match across the explicitly supplied indexes before reading any
payload. Zero or multiple matches fail before the solver. Other rows need no
payload for this operation. Without a scenario, it retains batch preparation.
`prepare_references(..., scenario=NAME)` offers the same selection for one index.
For bounded interaction-reference runs, the tests-side `interaction_density`
solves one tolerance at a time. Save the full DC-basis density and absolute end
of each chunk separately for coarse and fine. On resume, use that density as
`initial_densities` and that end as `start_ns`, without rebasing pulse start
or carrier phase. Compare the two final matrices on the original timeline;
only a difference of at most 2e-6 qualifies the fine density for its original
reference key. An intermediate checkpoint is not a reference-cache entry.
The reference producer reads names and physics; metadata never selects a solver.
All selected numeric files must load before any solver runs. Missing scene files
list the name, hashes and scene-producer command, then fail without rebuilding.
Existing reference keys are read cache-only. Missing reference keys are solved
only by that separately approved producer. Measurement never writes references
or copies selected records. Its `main(argv)` accepts only CLI arguments.
Small tests write temporary indexes and scene files through the public codec.
`read_inputs(indexes, scene_cache)` also accepts explicit paths and returns
declarations whose fresh sources belong to the caller.

### Partial attempts and failure handling

Use a new `--partial-output` JSONL path for each timing attempt. Existing journals
are rejected, and the journal cannot be either complete output path.
It starts with the run ID, manifest hash, Git state and partial status.
Each completed sample is appended outside timing, before the next evolve.
Skipped modes and completed policies are also retained. A later ordinary failure
adds its original error and returns nonzero without replacing earlier complete
reports. A timeout can leave the final line incomplete; only prior complete JSON
lines are usable. No `run_complete` event means the attempt is incomplete, even
if older matrix/summary files remain.

The bounded carrier run permits one 9000-second exclusive attempt. Stop on
timeout, preserve its receipt and journal, and do not restart automatically.
Use the run ID and receipt to distinguish this attempt from older complete files.

## Markdown reader declarations

```bash
uv run --directory <worktree> --no-sync -- python tools/check_doc_readers.py
```

`check_doc_readers.py` 以 `git ls-files` 選取 tracked `*.md`，不檢查 untracked 文件。
它依 [讀者宣告規範](../CLAUDE.md#讀者宣告)，略過 YAML front matter 後檢查前 8 行。
這是固定行數窗口，不解析 Markdown 段落。
ADR 與討論快照豁免，但 `docs/adr/README.md` 與 `docs/discussions/INDEX.md` 仍受檢查。
`lib/` 與 `tests/` 下的 `README.md` 另檢查 `**Last updated:** YYYY-MM-DD`。
日期只檢查格式，不判定日期新舊。

stdout 的 JSON 包含 `status`、`violation_count` 與 `violations`，每筆違規有 `path` 與 `reason`。
stderr 同時列出路徑與原因。無違規回傳 0，有違規回傳 1；Git 或檔案讀取失敗會拋出錯誤。
`--root <repo>` 可指定另一個 checkout，預設檢查工具所在的 checkout。
`check_sources(mapping)` 接受 repo-relative Markdown 路徑到文字的 mapping，回傳依路徑排序的
`Violation(path, reason)` tuple，不讀檔、不修改輸入。合成來源測試位於
`tests/tools/test_check_doc_readers.py`，不把 repo 文件內容固定在 pytest 裡。

## Document relations

```bash
uv run --directory <worktree> --no-sync -- python tools/doc_relations.py <base> [--term <literal> ...]
uv run --directory <worktree> --no-sync -- python tools/doc_relations.py --check
```

`doc_relations.py <base>` 比較 commit `<base>` 與工作目錄，列出這次改動可能讓哪些 tracked 文件過時。
輸出兩種行：

- `related: <path>  (<reason>)`：由引用關係找到的文件。包含每個改動路徑上各層的 `README.md`、
  這些 README 與改動檔案以 `ADR-NNNN` 或 `adr/NNNN-` 連結引用的 ADR，以及引用同一批 ADR 的其他文件。
  被超過 8 份其他文件引用的 ADR 只列它本身，原因欄註明略過的份數。
  Changed 檔若不是有效 UTF-8，內容不提供 ADR 引用，路徑仍參與 README chain。
  Markdown、Python 檔的解碼錯誤，以及 Git 與 I/O 錯誤仍會使命令失敗。
- `hit: <path>:<line>: <term> (<source>): <text>`：文件中出現搜尋詞的行。搜尋詞來自改動的 `.py` 檔：
  被刪除或改名的公開頂層函式、類別與大寫常數（`removed name`），定義內容有變的（`changed name`），
  以及檔案路徑的最後兩段（`changed path`）。`--term` 可重複，補上文件實際使用的稱呼（`given`）。
  名稱以完整單詞比對，路徑以子字串比對。方法名稱不納入，因為會命中不相干的敘述。

單一搜尋詞的命中超過 `--max-hits`（預設 40）時，只印一行 `too many:`，請換更精確的詞。
討論快照與 `docs/adr/retired/` 是凍結文件，兩種行都不列。這個模式一律回傳 0。

`doc_relations.py --check` 檢查所有 tracked Markdown：相對連結必須指到 tracked 檔案或含有 tracked 檔案的目錄，
不得用 `.py#L<n>` 釘行號；`ADR-NNNN` 必須對應 `docs/adr/` 或 `docs/adr/retired/` 下的檔案。
連結檢查略過 fenced code block 的示例，包括 fence 行本身。Fence 由行首最多三個空白後、
至少三個連續 backtick 或 tilde 開啟；backtick fence 的附加文字不能含 backtick。
結束行使用同一種字元，長度至少等於開啟行，後面只能有空白。未結束的 block 延續到該文件結尾。
正文連結仍依原文件行號回報，`ADR-NNNN` 在 fenced code 內也照常檢查。
每筆違規印一行 `<path>:<line>: <reason>`，有違規回傳 1。

`--root <repo>` 指定另一個 checkout。`related_documents`、`search_terms`、`find_hits` 與 `check_references`
接受路徑到文字的 mapping，不讀檔；合成來源測試位於 `tests/tools/test_doc_relations.py`。

## Routing telemetry analysis

`uv run --directory <worktree> --no-sync -- python tools/analyze_routing_telemetry.py .sim_telemetry/`
reads local routing JSONL files and prints aggregate family rows, usage shares,
cache hits, rejections and measured prepare/apply CPU seconds. It reports CPU
seconds per relative scheduling cost unit. For records with seconds estimates,
it also reports weighted `(actual - estimated) / estimated`; unknown estimates
remain excluded. Malformed lines produce file/line warnings and a skipped count.
Files and directories can be mixed; nothing is rewritten. All inputs merge into
one report. The analyzer never infers simulator identity from paths, filenames,
families or backend versions.

`wall_by_complete` groups `complete=true` as `normal_return`, `false` as `raised`,
and missing or null as `unknown`. Normal return includes a partial Stop result;
it does not mean every requested round ran. Only observed groups appear.
Each group reports `included`, `excluded`, and sums of `wall_s`, `compile_wall_s`,
`evolve_wall_s`, `readout_wall_s` and `remainder_s`. The remainder is total wall
minus the three stages, including environment preparation and pacing outside
those stages. The analyzer ignores supplemental `event: "materialize"` records
when counting acquires and aggregating these three stages.

A row contributes wall totals only when all four wall fields are JSON numbers.
Missing or null values in those four fields exclude the row from wall totals,
including in the unknown group. A failed acquire may not have reached a stage.
A null `materialize_wall_s` alone does not exclude it: that stage is not part of
this breakdown. The analyzer does not filter by `commit` or `dirty`. Included plus excluded across groups equals `records`.
All valid main rows still count toward family statistics. Supplemental records
do not contribute family statistics. Non-null `complete` accepts
only JSON booleans. Non-null wall fields accept only JSON numbers, not booleans or
numeric strings. Wrong types skip the whole row, including family work, and emit
the same file/line warning as other malformed records.

FTF, single and six default to the same project-root `.sim_telemetry` directory;
the rows contain no simulator identity. To inspect them separately, set distinct
`numerics.telemetry.directory` values and run once per directory:

```bash
uv run --directory <worktree> --no-sync -- python tools/analyze_routing_telemetry.py <ftf-directory>
uv run --directory <worktree> --no-sync -- python tools/analyze_routing_telemetry.py <single-directory>
uv run --directory <worktree> --no-sync -- python tools/analyze_routing_telemetry.py <six-directory>
```

One invocation with all three directories instead merges their rows.
The [shared telemetry contract](../lib/zcu_tools/simulators/README.md#acquire-telemetry)
defines fields, extras prefixes, failure behavior and the writer's file budget.

## Simulator capture inspection

```bash
uv run --directory <worktree> --no-sync -- python tools/inspect_simulator_capture.py
uv run --directory <worktree> --no-sync -- python tools/inspect_simulator_capture.py <capture-file-or-directory> --telemetry-directory <routing-directory>
```

The inspector prints JSON from recorded manifests and telemetry JSONL. It does
not construct models, run evolution, inspect dispatch or evaluate guards.
Omitting `path` calls `simulate.evolution.capture_directory()`. A directory
selects only its immediate `capture-*.json.gz` files, sorted by filename. An
explicit file selects that file alone. The default telemetry directory is
`.sim_telemetry` under the tool's own worktree root, regardless of the caller's
working directory. No path or family implies a simulator identity.

Each batch retains its capture ID, zero-based batch index, complete flag and
actual routes. Caller and worker rounds stay separate. Candidates retain their
recorded order and guard decisions, including `untested_after_short_circuit`.
A caller row remains pending, reported as `停在deferred`, only when its candidates
include a recorded `deferred` decision and no worker result resolves it. All-rejected
caller rows retain their candidates and null family without being marked pending.
Prefix representatives retain their actual routes. Adopters, including adopters
in another batch of the same evolve, report `沿用第k個prefix run` with a one-based
run number within the representative batch. The inspector does not invent
adopter classifications or repeated prefix evolution.

The inspector matches telemetry only when `capture_file` equals the manifest's
basename. Missing and null keys do not match. A unique match adds acquire
`complete`, total wall, compile, evolve, readout and remainder seconds. Remainder
is total minus those three stages. It does not change each evolve's recorded
`complete`. Without a match, the report says `未找到同次 telemetry 行`, omits timing
and does not guess why. Multiple matching rows raise an error naming the
manifest. Empty manifests report `沒有evolve被錄到`.

Compare optional conditions without changing the stored observations:

```bash
uv run --directory <worktree> --no-sync -- python tools/inspect_simulator_capture.py <capture-file> --budget 1e-7 --families '["direct"]'
```

`--budget` takes a float, `--source-identity` and `--segment-numeric-version`
take strings, and `--frozen-physical-version` takes an integer. `--families`
takes a JSON string array or `null` for unrestricted candidates. Array order and
duplicates do not change the restriction. An empty array means no candidates.
Omitted options retain each batch's recorded condition. Any difference lists
its metadata key in `condition_differences` and labels `新條件下重新評估`. The
report still shows the recorded conditions and routes, explicitly marked
`下列仍是錄製時選路，未重算`. This label is not a rerun or a reproduction claim.

Unrecorded routing reasons are `未知`. Numerical error is `未提供` because these
captures have no independent reference solution. Guard acceptance is not an
error estimate. The tool neither generates references nor reads numerical
caches. Missing required capture fields and malformed matched telemetry fail
instead of supplying default values.

## 怎麼跑

```bash
uv run --directory <worktree> --no-sync -- python tools/gate.py --base <lane 起點>
```

`gate.py` 是日常入口。它依序執行：受影響檔案的 ruff import 排序與 formatter、
`lint-imports`、lab script await guardrail、measure MCP 文字預算、然後 ratchet。
部分檢查會掃描全樹或啟動子程序，耗時不只由改動檔案數決定。Pyright 現況檢查另以 `--with-pyright` 啟用，不保證固定秒數。

省略 `--base` 時回報現況，列出全樹的絕對數字。
`docstring` 列執行 `ruff check --isolated --select D100,D101,D102,D103 --output-format json tools`，只報告缺少 docstring 的命中數，不影響 exit code，也不加入比較模式的硬性步驟：

```bash
uv run --directory <worktree> --no-sync -- python tools/gate.py                  # 現況
uv run --directory <worktree> --no-sync -- python tools/gate.py --with-pyright   # 加上 pyright
```

「這棵樹現在欠多少」和「這次改動有沒有加重」是兩個問題，所以給兩個答案。相對比較必須明填 base。
在長命分支上，自動選用 `merge-base HEAD main` 可能掃入大量無關變更。

```text
--base <ref>     判定基準。省略則改報現況
--no-fix         不改寫檔案，只判定。CI 或想先看再決定要不要格式化時用
--with-pyright   現況模式加上 pyright
```

`--no-fix` 會對同一批受影響檔案執行 import 排序檢查與 `ruff format --check`，不寫檔；
只有沒有存續的 Python 變更檔案時才跳過這兩項。

`--no-fix` 之外，預設會實際執行 import 排序與 formatter，也就是說 **`gate.py` 會改你的檔案**。
那是 `CLAUDE.md` §5 要求的步驟，放進來是為了不必記兩次。

### 失敗輸出

```text
ok   ruff import sort
ok   ruff format
ok   import contracts
ok   script awaits
ok   mcp text budget
FAIL ratchet
    suppressions: <path> suppression:noqa 0 -> 1
```

每行 regression 的形式是 `<detector>: <檔案> <規則> <base 計數> -> <candidate 計數>`。
想看某一項的完整輸出就直接跑那個 detector：

```bash
uv run --directory <worktree> --no-sync -- python tools/check_ratchet.py --base <ref> --detector suppressions
```

`gate.py` 在失敗時印診斷細節。品質設定有變時，即使計數未增加，也會列出 `REVIEW` 與
before/after；這是人工審查提示，不是自動判定設定變弱，也不代表已獲豁免。
範圍包含 Ruff、Pyright、pytest 設定與 `.importlinter`。刪除規則、降低 severity 或縮小
掃描範圍不能只靠違規計數看見，因此這些差異獨立列出。

格式化排在最前面是有原因的：它會改寫程式碼，先量測再格式化的話，讀到結果時那個狀態已經不存在了。

共用 scanner 與 ratchet 的變更檔選集包含 `mcp_servers/`，和使用者入口 `scripts/` 一樣納入檢查。Pyright 使用同名 root 的執行環境，test path checker 將 `tests/mcp_servers/` 對應 repo root 的組合根。

較耗時的兩項**刻意留在外面**；快速 gate 不替代 collection 與行為測試：

```bash
uv run --directory <worktree> --no-sync -- python tools/check_pytest_collection.py
```

全樹 pytest 使用 [tests/README.md 的完整選集命令](../tests/README.md#執行測試套件)，由該處維護 marker 選擇、rootdir、failure log 與收據重導。

`--dist=worksteal` 維持 command-level，因 parity intentionally disables pytest plugin autoload；
不設定 pytest 全域預設，因此未帶旗標的 `pytest -n auto` 語意不變。

## Lab script await guardrail

`check_script_awaits.py` 全掃 `zcu_lab/` 所有 async def，不 import 或執行 lab。
`check_sources(sources)` 接受 repo-relative path 到 source string 的 mapping，
回具名 Violation(path, line, target)，line 從 1 起算。
`main(argv)` 可用 `--root` 指定合成 fixture repo，stdout 回 violation_count／violations，
stderr 列 path／line／target；無違規 exit 0，違規 exit 1，讀取或語法失敗 exit 2。
Quality gate 以絕對檢查執行，不是可容忍新增違規的計數比較。
Status 模式也列目前違規數。

所有 recipe、workflow 與 helper 共用一條規則。
Await 目標必須是 env／scope／iteration／tab／session 呼叫，或解析到 lab 內定義的函式呼叫。
目標函式可用 def 或 async def 定義，工具不推導回傳型別。
工具解析直接的絕對／相對 import，不因外部函式名稱像 helper 就放行。
唯一開放的 lib await helper 是 `zcu_tools.experiment.run_core.run_core`，直接 import alias 也以完整路徑核對。其他 lib helper 仍拒絕。
Lab 模組只以自己的 import 轉匯出的名稱（例如 `from x import f as f`）會沿 import 追到定義；追到 lab 以外或循環則拒絕。
Helper 的 async def 本文也受檢查，未被呼叫的 async def 亦在選集，不識別 decorator 或做 reachability。
在 async def 內，明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接 Call 不論有無 await 都拒絕，
包含未 await 的 create_task 與 async with TaskGroup()。
其他 async with／async for 不另禁。

這是 ADR-0075 的漸進式 AST guardrail，不是 type checker 或 sandbox。
框架 receiver 以 env／scope／iteration／tab／session 字面名稱辨認，不驗證其 runtime 型別。
Assignment aliases、動態分派與包裝不追蹤，可能造成漏報或拒絕合法 helper。
出現實際失敗類別再擴充，不藉此新增作者排程 API。
同 process 合成 source／fixture 驗證在 `tests/tools/test_check_script_awaits.py`。

## Measure MCP text budget

```bash
uv run --directory <worktree> --no-sync -- python tools/check_mcp_text_budget.py
```

`check_mcp_text_budget.py` 用目前 interpreter 啟動 `mcp_servers/measure.py`，經 stdio
送出 `initialize`、`notifications/initialized` 與 `tools/list`，讀完回覆後以 EOF 關閉
server。它不啟動或連接 GUI，也不呼叫量測工具或硬體。`--timeout SECONDS` 限制 stdio
子程序的總時間，預設 60 秒，逾時會終止並回收子程序。

這個 collector 只對同 repo 的已知 stdio server 固定送三則只讀訊息，不是通用 MCP
client。這個用途不引入完整 async MCP SDK 依賴，也不改寫為 async。若未來需要互動
工具呼叫、支援異質 server 或通知路由，再評估使用 SDK。

檢查以解碼後的字元數計量，不計 UTF-8 bytes 或 JSON escape 字元。上限包含等號：

| Wire 文字 | 上限 |
| --- | ---: |
| Server instructions | 2,000 |
| Recipe tool description | 200 |
| 其他 tool description | 400 |
| Schema property description | 100 |

Recipe 身分來自與 server 相同的 `zcu_lab.assembly.lab_definitions().recipes`，以
`RecipeDeclaration.name` 精確比對 tool name，不寫死名稱，也不從前綴推測。
Property 檢查包含巢狀 object、array、union 及 `$defs` 中的欄位。不重複展開 `$ref`，
也不把 schema 頂層／型別說明或 `default`、`examples` 內的資料當成 property 說明。

stdout 的 JSON receipt 包含 `instructions_length`、`tool_count`、`overage_count`，以及
每筆超額的 `item`、`length`、`limit`。stderr 同時列出每筆超額的 wire 路徑、字元數與上限。
無超額回傳 0，有超額回傳 1；stdio 啟動、逾時或回覆解碼失敗回傳 2。
必要回覆結構、文字型別或被走訪的 schema 結構不合法也回傳 2，不用空值掩蓋錯誤。
省略或 null 的 descriptions 計為零；boolean subschemas 合法。只驗證計量所走訪的
schema 節點，不驗證 `default`、`examples`、`const`、`enum` 中的 instance data，
也不作通用 JSON Schema 驗證。
Lab 宣告載入錯誤直接拋出。`gate.py --base <ref>` 每次都執行此檢查，沒有 Python
檔案變更也不跳過。超額或執行失敗都使 gate 回傳 1。無 `--base` 的現況表以
`mcp text budget` 列出 `overage_count`。現況模式仍回傳 0，不判定變更是否通過。

`check_text_budget(instructions, tools, recipe_names)` 可直接 import，回傳
`BudgetReport`，其 `exit_code` 與 `overages` 分別提供 pass/fail 和全部超額項目。
`collect_text_budget(recipe_names, *, timeout=60.0)` 收集 wire 後回傳同一種 report，
不載入 lab；`main` 負責取得 lab 的 recipe names 與轉譯 collection errors。
`tests/tools/test_check_mcp_text_budget.py` 使用合成 wire inputs 驗證計量契約，並在外部
subprocess 邊界替換回覆以驗證 collection errors，不載入 lab 或真 server。

## 品質快照與熱點報表

三個主要入口各有責任：`gate.py` 串接日常檢查，`check_ratchet.py` 判定相對 base 的
regression，`quality_report.py` 保存現況並解讀債務分布。新報表不取代 gate，也不重做 detector
規則。`check_pytest_collection.py` 與 pytest 仍另外執行，快照不代表行為測試通過。

```bash
uv run --directory <worktree> --no-sync -- python tools/quality_report.py snapshot --with-pyright --output .agent_state/quality/before.json
uv run --directory <worktree> --no-sync -- python tools/quality_report.py snapshot --with-pyright --output .agent_state/quality/after.json
uv run --directory <worktree> --no-sync -- python tools/quality_report.py compare .agent_state/quality/before.json .agent_state/quality/after.json --output .agent_state/quality/comparison.json
```

沒有 `--output` 時 JSON 寫 stdout；指定檔案必須不存在，避免覆寫證據。snapshot 的 stderr
列出完整計數與各類 top 10，`--top N` 可調整顯示筆數。完整 finding、來源提供的 line/message
及額外量值保留在 JSON；可 import `summarize(snapshot, parent="lib/zcu_tools/gui")` 取得完整
分布或限定父目錄。分組為 production、tests、scripts、tools、configuration、other，以及
所屬目錄、規則與檔案。不把跨 detector 的總數當品質分數。

Pyright 為 opt-in，未指定時記錄 skipped；啟用時只計 errors，但保留非 error 診斷為 uncounted。
測試結構的 advisory 也保留但不計入違規。零筆只能是 completed 的觀察；detector 失敗記錄
error 與原因，不能轉成零。Import contracts 獨立記錄 pass/fail/error，不混入診斷數。

Snapshot 保存 commit、HEAD tree、tracked/untracked status、被量測 Python source digest、
Python/platform、已安裝 distribution 版本、工具版本、repo 設定與工具實作檔案指紋。
掃描前後來源或方法有變即拒絕輸出，避免把兩棵樹混成一次觀察。它是目前 worktree 的快照，
不是重建任意歷史 commit 的工具；外部 SDK、硬體、未安裝 runtime profile 與外部設定不在涵蓋內。
同版本套件的原地修改不由 distribution 版本指紋辨識；detector 自身的靜態分析限制仍適用。

Compare 要求相同方法與 detector selection/state；任一 error 或方法不同會拒絕比較，
列出理由。不同 commit/source 是允許的。設定、工具或環境更新後應以同一方法重量兩側，
或建立新的 baseline，沒有 force-comparable 開關。

比較按精確 `(detector, path, rule)` 列出 before/after、introduced_count、resolved_count、net，
並彙整到 scope/module/rule。introduced/resolved 指計數變化，不是逐條診斷的身分追蹤。
搬檔會出現舊路徑減少與新路徑增加，報表不把它解讀為修復；正式 rename 判定仍由 ratchet
擁有。既有 suppression 的設定項目歸 configuration，不假裝有來源行號。

Exit 0 只表示報表成功，不表示品質合格；即使有診斷或 import gate fail 也可成功產生報表。
工具錯誤、無效 receipt 或不可比條件 exit 2。規則沒有新門檻，既有 gate 輸出與 exit 語意不變。

### Radon CC advisory

`quality_report.py --with-radon` 是 opt-in 分數報表，保留 advisory 計量，不取代 Ruff `C901`。
ratchet 另外預設啟用 `radon` detector，只比較 CC > 20 的函式數，不比較分數漲跌。
快速 `gate.py` 透過 ratchet 執行這項比較。兩者共用 Radon 解析，但掃描範圍不同。
先安裝 locked quality 環境，再產生快照：

```bash
uv sync --directory <worktree> --locked --group quality
uv run --directory <worktree> --no-sync -- python tools/quality_report.py snapshot --with-radon --output .agent_state/quality/radon.json
```

`--with-radon` 可與 `--with-pyright` 同時使用；`--top N` 控制 stderr 熱點數量。
報表在 worktree Python 內使用 Radon 分析函式，不呼叫 CLI 或 `uv tool install` 提供的全域 executable。
選集由 repo 掃描器決定，固定 `no_assert=False`；不採用 `RADONCFG`、個人或 repo 的 Radon CLI
設定，因此 `cc_min`、`exclude`、`no_assert` 等 CLI 設定不會悄悄改變報表。
未選用時明列 skipped；選用但未安裝、解析失敗或輸出無效時記 error，exit 2，不當作零筆。
高複雜度本身不改變 exit code，也不判定品質合格與否。

報表沿用來源掃描排除規則，只分析 `lib/` 與 `tools/`，不含 tests、scripts 或 Notebook。
保留 Radon JSON 提供的函式、方法及 closure，排除 class aggregate；巢狀 block 以 qualified name
顯示。Radon 未提供的 block 不另自行推導，例如函式內定義的 class 可能不在其輸出中。
計算包含 assert，不按 rank 過濾；CC 與 Ruff 的演算法不同，不能直接共用 12 的門檻。

JSON 的 `detectors.radon.findings` 保存 path、line，以及 details 中的 name、complexity、rank；
全部 `counted=false`，不混入違規計數。stderr 分開顯示 production/tools 的等級分布與 CC 熱點；
可 import `complexity_summary(snapshot, top=10)` 取得同樣資料。
A 為 CC 1–5、B 為 6–10、C 為 11–20、D 為 21–30、E 為 31–40、F 為 41 以上。
這些是閱讀與重構線索，不是拆函式指令，不引入 MI 品質總分。

Snapshot 使用 schema 2；schema 1 需以目前方法重新量測。
Radon distribution 版本隨既有 method.distributions 保存；方法或選用狀態不同仍拒絕比較。
`compare` 保持違規計數比較，**不比較 CC 分數升降，也不配對函式改名或搬移**。

## Ratchet 比較與 exit code

`check_ratchet.py` 把八項檢查對照 base，逐個檔案及規則列出計數上升。
detector 名稱為 `ruff`、`pyright`、`radon`、`file-size`、`path-correspondence`、
`test-structure`、`capabilities` 與 `suppressions`。
交付時逐項列出上升的 path、detector／rule、before／after、分類及理由。
實質問題、搬檔診斷與既有超標項的審查判準見
[程式碼品質](../docs/code-quality.md#審查裁決)。`lint-imports` 與 pytest collection
各自保留硬性檢查結果，行為測試另行執行。

File-size 對每個檔案只計是否超過 1000 行。超過計 1，否則計 0，不比較 raw 行數增長。
1135 → 1137 是 1 → 1，不新增 regression；998 → 1003 與新檔 1001 行都是 0 → 1。
恰好 1000 行未超標。`check_quality.py file-size` 仍列出實際行數，既有 oversize 檔案增長是
非實質的大小變化，不會增加 file-size 計數。各檔獨立比較，刪除其他檔案不能抵銷新超標。

Radon 對 `lib/` 與 `zcu_lab/` 全部 Python 來源計數，包括 recipes／workflows，
不含 tests、tools 或 scripts。沿用 repo 掃描器的排除規則，計入 assert，排除 class aggregate。
每個 CC > 20 的函式、方法或 closure 計 1，rule 為 `cyclomatic-complexity`。
CC 20 → 21 是 0 → 1；21 → 40 與 40 → 21 都是 1 → 1；21 → 20 是 1 → 0。
各檔案只比超標函式總數，不追蹤函式身分，也不比較 raw CC 分數。
未安裝 Radon、讀檔失敗或解析錯誤時 exit 2，不當作零筆。

一般診斷及設定抑制逐個檔案及規則比較。Git 以 20% 相似度確認的 rename 沿用原檔來源；
其餘新檔從零比較。拆檔只有一個新檔可成為 Git rename，其餘路徑的診斷上升要逐項判讀。
工具沒有持久 baseline 檔，也不替 reviewer 判定診斷是否實質新增。

CLI 保留機械狀態：沒有計數上升時回報 `PASS`／exit 0，有上升時回報 `FAIL`／exit 1，
工具無法完成比較時 exit 2。`FAIL` 要求逐項審查，不等於自動拒絕交付；不能把接受的上升
改寫成機械 `PASS`。`gate.py` 仍呈現這些訊號，硬性檢查失敗則必須處理。

`type-ignore` 與 `pyright-ignore` 另外按來源位置及 diagnostic scope 比較。
`tools/suppression_comparison.py` 的 pure `compare_ignores` 以唯一 AST statement、
owner／branch 與 line role 識別位置，保留 comment 增刪與 formatter 拆行後的對應。
同位置的 blanket type-ignore 改成非空 explicit pyright-ignore code 集合可以通過；
同 kind 也可以維持或縮小原範圍。新增位置、擴大 code 集合或改回 blanket 會回報上升，
其他位置或種類的刪除不能抵銷。位置有歧義、語法無法證明或 code list 無效時，
工具不會自動核准來源配對，必須保留診斷並交審查判讀。配對不跨 function／class／branch。

這類 regression 的 rule 包含 kind、reason 與 candidate line。before 0／after 1 表示
工具回報一個 escape-site 變化，不是該檔的 raw marker 總數。
`check_quality.py suppressions` 與品質 snapshot 仍提供完整 usage counts，不清零抑制或換 baseline。
`check_ratchet.suppression_regressions` 負責 Git path／rename 與 source-pair 接線，
其他 escape families 及 configuration 仍分別比較計數。

```bash
--base <ref>        比較基準，預設 git merge-base HEAD main
--detector <name>   只跑一項，可重複；查看上升細節時用
```

兩側都用 candidate 的規則量測。base tree 透過 `git archive` 展開後拿到 candidate 的
`pyproject.toml`，避免新啟用的規則只出現在 candidate。base 同時保留原設定於
`pyproject.base.toml`，供設定層抑制比對，不讓新增 per-file-ignore 隱形。

預設 Pyright 只檢查變更檔案，不能保證未修改 caller 仍符合修改後的 interface。
合併前另跑慢速全樹比較，讓未修改 caller 的診斷保持可見。
使用[重計算執行器段落的 Full Pyright 命令](#重計算執行器)，保留 log 與 runner 收據。

`--full-pyright` 對兩側使用 candidate 設定、目前 worktree interpreter 與已安裝依賴，
不建立 base 的另一個環境。receipt 的 `pyright_scope` 記錄範圍，成本與全樹相關，
因此不放進快速 gate。`gate.py --with-pyright` 是現況報告，exit 0 不代表本次變更已驗收。

`check_ratchet.py` 與 `gate.py --with-pyright` 以 `_support.PYRIGHT_THREADS`（4 個執行緒）呼叫 Pyright。2026-11-02 在 8 核主機上量測全樹 strict 檢查：單執行緒 152 s、4 執行緒 64 s、8 執行緒 85 s（同時有 pytest 佔用 CPU），三者的診斷完全相同。

Full 模式會重用 base 的 Pyright counts。快取位於主 checkout 的
`.agent_state/ratchet/cache/`，透過 Git common dir 定位，所有 worktree 共用。
Key 包含 base commit SHA、candidate `pyproject.toml` 與 `uv.lock` 的 SHA-256、
Pyright 版本，以及目前 Python 的 implementation 與版本。任一成分改變即重新計算。
快取以原子替換寫入，損壞或讀取失敗的檔案視為未命中並重新計算。
刪除該目錄即可清除全部快取。

Full receipt 的 `pyright_base` 值為 `cache` 或 `computed`。命中時跳過 base 的
Pyright；若沒有其他 detector 或設定差異需要 base，也跳過 base tree 展開。
Candidate 仍每次執行 Pyright。Changed-files 模式不使用快取，行為不變。
快取沿用 `uv.lock` 描述的環境，不偵測同一 lock 下已安裝依賴的原地修改。
這類環境修改後需刪除快取再跑。

設定差異與診斷分開判讀。合併前必須審閱 `configuration_changes`，不能只看 `PASS`。
暫存 base tree 放在 invoking worktree 的 `.agent_state/ratchet/`，比較結束即移除該次目錄。

## 檢查項目

`lint-imports` 與 `check_pytest_collection.py` 是獨立的硬性檢查；前者由 `gate.py` 呼叫，後者需另外執行。Ruff、Pyright、Radon、file size、test path／structure／capabilities 與 suppressions 等計量項由 `check_ratchet.py` 對照 base 判斷。單獨執行 detector 顯示現況，不代表 regression 驗收。pytest 行為選集仍需另跑。

實際規則和掃描範圍見 `pyproject.toml`、`.importlinter` 與各 detector。新增 lint 規則與 missing-import 檢查仍由 ratchet 判定；optional profile 未安裝時的 import 診斷保留可見，不為消除診斷而全域關閉 import 檢查。

## 品質檢查

`check_quality.py` 必須指定子命令。頂層與各子命令的 `-h` 列出用法，不掃描來源。
下表命令沿用 CLI 一覽的 Python 前綴。

| 子命令 | 命令 | Exit code |
| --- | --- | --- |
| `file-size` | `tools/check_quality.py file-size` | 0 未超標、1 有超標 |
| `test-paths` | `tools/check_quality.py test-paths` | 0 對應存在、1 有缺口 |
| `test-structure` | `tools/check_quality.py test-structure [--root <repo>]` | 0 無 blocking、1 有 blocking、2 root／讀取／解析失敗 |
| `test-capabilities` | `tools/check_quality.py test-capabilities` | 0 已宣告、1 有未宣告能力 |
| `suppressions` | `tools/check_quality.py suppressions` | 0，僅計量 |
| `all` | `tools/check_quality.py all` | 五項 exit code 的最大值 |

每個子命令的 stdout 是自己的 JSON receipt。`all` 依表列順序執行五項檢查，
stdout 是以子命令名為鍵、receipt 為值的 JSON 物件。`test-structure` 失敗而沒有 receipt 時，
該值是空物件。stderr 依序保留各項診斷。`gate.py` 現況表保留五列，
但只呼叫一次 `all`，各列讀取自己的計數。

### 測試結構

歸屬、命名、fixture 與搬遷 policy 見 [tests/README.md](../tests/README.md)。先看完整診斷，
再用 ratchet 判斷本次是否新增 blocking findings：

```bash
uv run --directory <worktree> --no-sync -- python tools/check_quality.py test-structure
uv run --directory <worktree> --no-sync -- python tools/check_ratchet.py --base <ref> --detector test-structure
```

獨立 checker 可帶 `--root <repo>`。JSON 的 `findings` 包含 path、line、rule、message、severity；
`violation_count` 只計 blocking，`advisory_count` 單獨列出。沒有 blocking 時 exit 0，有則 exit 1；
讀檔或語法錯誤 exit 2，不視為通過。它不匯入或執行被檢查的測試。

Blocking 規則：

- `test-module-import`：tests 內 Python 檔明確 import 另一個存在的 `tests/**/test_*.py`。
- `conftest-import`：明確 import 存在的 `tests/**/conftest.py` 作為 library。
- `duplicate-test-definition`：同 module 或 Test class 的直接敘述重複定義 test function / Test class。
  不同 class 的同名測試不衝突；條件分支與 nested function 不推論為收集時覆蓋。

Import 解析限 repo-root-qualified 絕對路徑與明確相對路徑，不追 dynamic import、re-export、
pytest import mode 或自訂 `sys.path`。例如 bare `import conftest` 不在目前可確定解析範圍。
`from package import name` 的 name 可能是套件屬性，不因同名檔案存在就判定匯入子模組；
只檢查明示的 package/module 部分。`from .test_peer import helper` 仍檢查 test_peer。
同名 `.py` 與含 `__init__.py` 的套件共存時，優先解析套件。這些限制不是 policy 豁免。重複定義檢查依預設 `test_` / `Test` 命名，不模擬 decorator、繼承、
assignment 或 pytest plugins 的收集行為。

`test-file-name` 對 ticket、phase、part 加編號、歷史 B/C 編號及 misc 類名稱提醒人工審閱；
不阻擋，也不把 `T1` / `T2` 當 ticket。命中不代表已確認歸屬錯誤。
Fixture 依賴深度、setup 語意重複與責任混合仍由 review 判斷，不設 LOC 比或案例數硬門檻。
`check_quality.py file-size` 提供大小訊號，無須另設一套測試檔行數門檻。

快速 gate 的 ratchet 只比較 blocking 的每檔每規則計數，兩側用 candidate checker 掃全 tests tree，
讓新增目標檔造成既有 import 可解析的情況也被看見。無法由 Git 確認來源的新增檔仍從零計數，
需審閱是否有拆檔繼承的診斷。命名提醒留在獨立 checker；`gate.py` 現況表只顯示 blocking 數。

## Opt-in diagnostics

以下命令不進快速 gate，也不自動修復依賴。先在要檢查的 worktree 安裝 locked 環境：

```bash
uv sync --directory <worktree> --locked --group quality
```

一般 `pytest` 預設停用 randomly plugin，即使已安裝 quality group 也不改變 collection 順序。
診斷測試污染時，明確啟用 plugin 與 seed；先單 worker，再按需要檢查 parallel 排程：

```bash
uv run --directory <worktree> --no-sync -- pytest <affected-tests> -n 0 -p randomly --randomly-seed=12345
```

同 seed 不保證重現 thread／xdist 排程；randomly 也會重設隨機狀態。保留 seed、選集與執行模式，
把順序造成的差異視為缺陷，不以偶然通過結案。

Branch coverage 已由 dev group 提供，針對修改的 module 找未驗證分支，不設全域百分比門檻：

```bash
uv run --directory <worktree> --no-sync -- pytest <affected-tests> --cov=<affected-module> --cov-branch --cov-report=term-missing
```

依賴宣告與安全稽核適合獨立診斷或 release 前執行，非 code-health regression 判定：

```bash
uv run --directory <worktree> --no-sync -- deptry .
uv run --directory <worktree> --no-sync -- pip-audit --format json
```

Deptry findings 需對照 `lib/` layout、套件／module 名稱、Notebook 使用與 optional extras；
不要只因 unused 診斷而移除依賴。Pip-audit 需要漏洞服務，檢查的是目前環境，不代表未安裝的
Python 3.12 design 或其他 profiles 也通過；Git／未識別套件是涵蓋限制，需列出而不是略過。
每個受支援 profile 用自己的環境檢查，不為了掃描而混裝不相容 extras，也不使用 `--fix`。

Ruff 候選規則調查使用 `--extend-select` 保留原規則；`RUF100` 的結果取決於目前啟用哪些規則。
抑制清理前確認適用的完整規則集合。Broad catch 的 logging 可以通過 lint，但是否能恢復、
失敗狀態是否保留，仍由 code-quality policy 的人工審查判斷。

## Suppression 計量

`check_quality.py suppressions` 計量 `# noqa`、`# type: ignore`、`# pyright: ignore`、
`cast()`、per-file-ignore，以及 `pyproject.toml` 中停用的規則。它固定 exit 0，
判定由 ratchet 與人工審查完成。例如 `# noqa: C901` 會降低 Ruff 診斷數，
suppression 計數則讓這項改動保持可見。

保留 suppression 的條件、指定 rule 與原地理由見
[既有容忍項](../docs/code-quality.md#既有容忍項先嘗試修除理由留在原地)。
`check_quality.py test-capabilities` 的 marker 記錄測試需要的能力，適用契約見
[tests README](../tests/README.md)。

計量涵蓋可辨識的註解、設定與 import。Reviewer 仍需審閱改名與拆檔的目的，
例如把 `_foo` 改名以避開 `reportUnusedFunction`，或把 5000 行拆成五個 999 行的檔案。

## 判讀一次測試執行

每次驗證記錄候選 commit、環境、測試選集、xdist 排程、exit status 與原始 log。collection 相同或一次通過不保證沒有順序依賴及 Qt 資源生命週期問題；worker 中止、間歇失敗與警告需要在相同選集重新調查，不能因個別重跑通過就宣告原失敗解除。歷史候選的通過數與失敗案例不是現況基準，也不是豁免。
