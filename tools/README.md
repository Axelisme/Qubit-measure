# tools/

**Last updated:** 2026-09-24 — check-only、全樹型別比較與設定 review

`tools/` 放 repo 內部的品質檢查。`script/` 放使用者入口——板端 server、GUI 啟動、資料工具。
兩者的讀者不同，不混用。

本目錄的檢查是同一個形狀：純函式加上一個回傳 exit code 的 `main()`，把 JSON receipt 輸出到
stdout。`check_file_size.py`、`check_test_capabilities.py`、`check_test_path_correspondence.py`
另把人類可讀的違規行輸出到 stderr；`check_suppressions.py` 只計量不判定，在 stderr 列出使用量最多的
抑制（最多 20 筆）並固定 exit 0；`check_ratchet.py` 與
`check_pytest_collection.py` 只在工具本身失敗時寫 stderr。`gate.py` 是給人看的入口，輸出純文字。
它們可以被 import，`tools/check_ratchet.py` 就是這樣使用其他幾支的。

`_support.py` 放它們共用的機制——走訪樹、讀 attribute chain、透過 import 解析呼叫、載入另一支
檢查。各支檢查**判斷什麼**仍各自獨立，共用的只有這些。它假設 `tools/` 在 `sys.path` 上，這在三種
情況下都成立：直接執行某支檢查、`load_tool` 載入另一支、以及 pytest 的 `pythonpath`。

## 怎麼跑

```bash
uv run --no-sync -- python tools/gate.py --base <lane 起點>
```

`gate.py` 是日常入口，約 2 秒。它依序執行：受影響檔案的 ruff import 排序與 formatter、
`lint-imports`、然後 ratchet。

**不給 `--base` 時它改報現況**——全樹的絕對數字，約 6 秒：

```bash
uv run --no-sync -- python tools/gate.py                  # 現況
uv run --no-sync -- python tools/gate.py --with-pyright   # 加上 pyright，多約 40 秒
```

「這棵樹現在欠多少」和「這次改動有沒有加重」是兩個問題，所以給兩個答案。沒有 base 時不會
默默退回 `merge-base HEAD main`——在長命分支上那是九百多個變更檔與數分鐘的 pyright。

```text
--base <ref>     判定基準。省略則改報現況
--no-fix         不改寫檔案，只判定。CI 或想先看再決定要不要格式化時用
--with-pyright   現況模式加上 pyright（約 +40 秒）
```

`--no-fix` 會對同一批受影響檔案執行 import 排序檢查與 `ruff format --check`，不寫檔；
只有沒有存續的 Python 變更檔案時才跳過這兩項。

`--no-fix` 之外，預設會實際執行 import 排序與 formatter，也就是說 **`gate.py` 會改你的檔案**。
那是 `CLAUDE.md` §5 要求的步驟，放進來是為了不必記兩次。

### 失敗長什麼樣

```text
ok   ruff import sort
ok   ruff format
ok   import contracts
FAIL ratchet
2 changed Python file(s), base f1c774476
    suppressions: tools/_p.py suppression:noqa 0 -> 1

Run separately: python tools/check_pytest_collection.py; pytest -n auto --dist=worksteal
```

每行 regression 的形式是 `<detector>: <檔案> <規則> <base 計數> -> <candidate 計數>`。
想看某一項的完整輸出就直接跑那個 detector：

```bash
uv run --no-sync -- python tools/check_ratchet.py --base <ref> --detector suppressions
```

`gate.py` 在失敗時印診斷細節。品質設定有變時，即使計數未增加，也會列出 `REVIEW` 與
before/after；這是人工審查提示，不是自動判定設定變弱，也不代表已獲豁免。
範圍包含 Ruff、Pyright、pytest 設定與 `.importlinter`。刪除規則、降低 severity 或縮小
掃描範圍不能只靠違規計數看見，因此這些差異獨立列出。

格式化排在最前面是有原因的：它會改寫程式碼，先量測再格式化的話，讀到結果時那個狀態已經不存在了。

慢的兩項**刻意留在外面**，把它們折進來會讓一個兩秒的指令變成兩分鐘，然後所有人都學會跳過它：

```bash
uv run --no-sync -- python tools/check_pytest_collection.py   # 約 30 秒
uv run --no-sync -- pytest -n auto --dist=worksteal           # 約 2 分鐘
```

`--dist=worksteal` 維持 command-level，因 parity intentionally disables pytest plugin autoload；
不設定 pytest 全域預設，因此未帶旗標的 `pytest -n auto` 語意不變。

## ratchet 是判準，不是另一個檢查

`check_ratchet.py` 把六項檢查對照 base 判讀：逐 (檔案, 規則) 比較 base tree 與 candidate，
**只有計數上升才失敗**。既有債務不擋工作，往上加才擋。

```bash
--base <ref>        判定基準，預設 git merge-base HEAD main
--detector <name>   只跑一項，可重複；ratchet 報了 regression 想看細節時用
```

兩個設計決定值得知道：

**沒有 baseline 檔。** 以路徑為 key 的 baseline 在檔案改名時失效——那正是大重構最需要 gate
讓路的時候；而且它會變成第二個需要治理所有權的檔案。git 已經知道什麼變了。

**兩側都用 candidate 的規則量測。** base tree 透過 `git archive` 展開後會拿到 candidate 的
`pyproject.toml`，否則新啟用一條規則會讓它的所有發現看起來都是新的，ratchet 會擋下「把檢查
打開」這個改動本身。但 base tree 同時保留自己那份於 `pyproject.base.toml`，供設定層的抑制
比對使用——否則新增一條 per-file-ignore 會對負責看見它的檢查隱形。

預設 Pyright 只檢查變更檔案，不能保證未修改 caller 仍符合修改後的 interface。
合併前使用獨立的慢速全樹比較，讓既存債務保持可見而不阻擋本次變更：

```bash
uv run --no-sync -- python tools/check_ratchet.py --base <ref> --detector pyright --full-pyright
```

`--full-pyright` 對 base 與 candidate 都使用 candidate 設定、目前 worktree interpreter 與
已安裝依賴，不建立 base 的另一個環境；receipt 的 `pyright_scope` 記錄此次範圍。
此模式包含未修改 caller，成本與全樹相關，刻意不加入快速 gate。
`gate.py --with-pyright` 則仍是現況報告，exit 0 不代表 regression 驗收通過。

設定差異與診斷分開判讀：兩側使用 candidate 設定仍會受規則弱化影響，合併前必須審阅
`configuration_changes`，不能只看 `status: PASS`。暫存 base tree 放在 invoking worktree
的 `.agent_state/ratchet/`，比較結束即移除該次目錄。

## 九項檢查

| 檢查 | 守什麼 | 判讀 | 現況（2026-09-24） |
| --- | --- | --- | --- |
| `lint-imports` | 模組間的依賴方向，契約在 `.importlinter`（目前只有 C1：experiment 不依賴 gui） | 硬性 | 綠 |
| `check_pytest_collection.py` | 四種 pytest entrypoint／並行組合收集到相同的測試集 | 硬性 | 綠 |
| `ruff check` | 結構：函式複雜度、分支數、statement 數、參數數、死碼、boolean trap | ratchet | 1492 |
| `pyright` | 型別，以及跨模組存取 private 造成的封裝破口 | ratchet | 3871 |
| `check_file_size.py` | 單檔行數上限 1000 | ratchet | 41 |
| `check_test_path_correspondence.py` | 每個測試目錄的路徑對應一個實際存在的模組 | ratchet | 20 |
| `check_test_capabilities.py` | 測試模組宣告它用的 socket／subprocess／sleep | ratchet | 34 |
| `check_suppressions.py` | 逃生口計量 | ratchet | 2349 |
| `pytest -n auto` | 行為 | 硬性 | **紅**，見下 |

前兩項已全綠，紅了就是這次改動造成的。

中間六項對全 repo 回報數百至數千筆既有債務。**單獨執行它們只會看到既有狀態**，判定一律經由
ratchet。

`ruff` 與 `pyright` 的規則選擇記在 `pyproject.toml` 的註解裡，包含刻意排除哪些規則與原因。

## 為什麼要計量逃生口

其他檢查都只數違規，所以「消音」比「修好」便宜：`# noqa: C901` 讓違規歸零且不留痕跡，
ratchet 會把它讀成進步。

`check_suppressions.py` 數的是消音動作本身——`# noqa`、`# type: ignore`、`# pyright: ignore`、
`cast()`、per-file-ignore，以及在 `pyproject.toml` 裡被關掉的規則。經過 ratchet，消音變成拿一個
計數換另一個計數，總數不落。它自己永遠 exit 0，不判定任何事。

**它不禁止抑制。** 抑制有時是對的：`pyproject.toml` 關掉的 `reportUnknown*` 規則，
診斷幾乎都來自無型別的第三方套件（qick、pyvisa、scqubits、qtpy），不是本 repo 的程式。禁止只會把人推去寫
不需要抑制的更差的程式碼。計量的作用是讓那筆交易在 review 時看得見。

同樣的理由，`check_test_capabilities.py` 的 marker 命名的是**能力**而不是「我可以慢」的許可：
標錯了測試會變成錯的，而不是被豁免。時間門檻量的是症狀且可以用 marker 交易掉；能力量的是原因。

這套東西的邊界也在這裡：**機械 gate 無法防逃逸，只能讓逃逸留下可數的痕跡。** 堵得住的洞，
都是因為規避動作本身留下了東西（一個註解、一個設定條目、一個 import）。留不下痕跡的規避
——把 `_foo` 改名成 `foo` 讓 `reportUnusedFunction` 不發作、把 5000 行拆成五個 999 行——
只能靠 reviewer。

## pytest 目前是紅的

在 main＋Load 起點（`18f22e46a`）與加入這套工具後的分支上，`pytest -n auto` 都有同樣三類既存失敗：

- `tests/gui/ui/test_writeback_widget.py` 的三個 layout 測試每次失敗。
- 每次執行約有一次 xdist worker 以 `Fatal Python error: Aborted`／`node down` 終止，發生在 Qt 物件的 GC
  期間；被記為失敗的是當時在該 worker 上的舊 UI 測試，依排程而異，單獨重跑會通過。
- `tests/gui/plotting/test_plotting.py::test_registry_evicts_gc_collected_figure` 間歇失敗：它比較全域
  `WeakKeyDictionary` 的絕對筆數，同一 worker 上其他測試的 figure 在期間被 GC 就會改變計數。

在它們修好之前，判讀看的是「有沒有多出這三類以外的失敗」。
