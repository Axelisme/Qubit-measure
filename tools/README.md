# tools/

`tools/` 放 repo 內部的品質檢查，以及永久保留的離線 `migrate_storage.py`。後者由明確歷史 evidence 與 lab mapping 轉換舊 result／Database，不控制硬體、不接正常 runtime fallback。review 品質判準由 [程式碼品質](../docs/code-quality.md) 定義，環境與驗證流程依 [AGENTS.md](../AGENTS.md)。`scripts/` 放使用者入口——板端 server、GUI 啟動、資料工具。
品質檢查與使用者啟動腳本的讀者不同。Migration CLI 是永久離線工具，保留在 tools，不接量測啟動流程。

除離線 migration CLI 外，本目錄的檢查是同一個形狀：純函式加上一個回傳 exit code 的 `main()`，把 JSON receipt 輸出到
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
uv run --directory <worktree> --no-sync -- python tools/gate.py --base <lane 起點>
```

`gate.py` 是日常入口。它依序執行：受影響檔案的 ruff import 排序與 formatter、
`lint-imports`、然後 ratchet。

**不給 `--base` 時它改報現況**，列出全樹的絕對數字：

```bash
uv run --directory <worktree> --no-sync -- python tools/gate.py                  # 現況
uv run --directory <worktree> --no-sync -- python tools/gate.py --with-pyright   # 加上 pyright
```

「這棵樹現在欠多少」和「這次改動有沒有加重」是兩個問題，所以給兩個答案。沒有 base 時不會
默默退回 `merge-base HEAD main`。在長命分支上，自動選用該基準可能掃入大量無關變更。

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
FAIL ratchet
    suppressions: <path> suppression:noqa 0 -> 1
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

較耗時的兩項**刻意留在外面**；快速 gate 不替代 collection 與行為測試：

```bash
uv run --no-sync -- python tools/check_pytest_collection.py
uv run --no-sync -- pytest -n auto --dist=worksteal
```

`--dist=worksteal` 維持 command-level，因 parity intentionally disables pytest plugin autoload；
不設定 pytest 全域預設，因此未帶旗標的 `pytest -n auto` 語意不變。

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

Radon 是 opt-in 報表，不加入 `gate.py` 或 ratchet，也不取代 Ruff `C901`。
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

新增 detector 後 snapshot schema 為 2，舊 schema 1 需以目前方法重新量測，不自動補零。
Radon distribution 版本隨既有 method.distributions 保存；方法或選用狀態不同仍拒絕比較。
`compare` 保持違規計數比較，**不比較 CC 分數升降，也不配對函式改名或搬移**。

## ratchet 是判準，不是另一個檢查

`check_ratchet.py` 把七項檢查對照 base 判讀。一般診斷及設定抑制逐 (檔案, 規則)
比較，計數上升才失敗。Git 以 20% 相似度確認的搬檔沿用原檔來源；其餘新檔從零比較。

`type-ignore` 與 `pyright-ignore` 另外按來源位置及 diagnostic scope 比較。
`tools/suppression_comparison.py` 的 pure `compare_ignores` 以唯一 AST statement、
owner/branch 與 line role 識別同一位置，保留 comment 增刪與 formatter 拆行後的對應。
同位置的 blanket type-ignore 改成非空 explicit pyright-ignore code 集合可以通過。
同 kind 也可以維持或縮小原範圍。新增位置、擴大 code 集合、改回 blanket 都失敗，
其他位置或種類的刪除不能抵銷。位置有歧義、語法無法證明或 code list 無效時不核准遷移。
完全相同的 source 保留既有 debt。位置配對不跨 function/class/branch。

這類 regression 的 rule 包含 kind、reason 與 candidate line，before 0 / after 1 表示
新增一個不允許的 escape-site 變化，不是該檔的 raw marker 總數。
`check_suppressions.py` 與品質 snapshot 仍提供完整 usage counts，沒有把抑制清零或換 baseline。
`check_ratchet.suppression_regressions` 擁有 Git path/rename 與 source-pair 的接線，
其他 escape families 及 configuration 仍分別比較計數。

```bash
--base <ref>        判定基準，預設 git merge-base HEAD main
--detector <name>   只跑一項，可重複；ratchet 報了 regression 想看細節時用
```

兩個設計決定值得知道：

**沒有 baseline 檔。** Git diff 將可辨識的舊路徑對應到新路徑，再比較兩側檢查結果。
拆檔時只有一個新檔可成為 Git rename；其餘新檔仍須處理新增診斷。

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

設定差異與診斷分開判讀：兩側使用 candidate 設定仍會受規則弱化影響，合併前必須審閱
`configuration_changes`，不能只看 `status: PASS`。暫存 base tree 放在 invoking worktree
的 `.agent_state/ratchet/`，比較結束即移除該次目錄。

## 檢查項目

`lint-imports` 與 `check_pytest_collection.py` 是獨立的硬性檢查；前者由 `gate.py` 呼叫，後者需另外執行。Ruff、Pyright、file size、test path／structure／capabilities 與 suppressions 等計量項由 `check_ratchet.py` 對照 base 判斷。單獨執行 detector 顯示現況，不代表 regression 驗收。pytest 行為選集仍需另跑。

實際規則和掃描範圍見 `pyproject.toml`、`.importlinter` 與各 detector。新增 lint 規則與 missing-import 檢查仍由 ratchet 判定；optional profile 未安裝時的 import 診斷保留可見，不為消除診斷而全域關閉 import 檢查。

## 測試結構

歸屬、命名、fixture 與搬遷 policy 見 [tests/README.md](../tests/README.md)。先看完整診斷，
再用 ratchet 判斷本次是否新增 blocking findings：

```bash
uv run --directory <worktree> --no-sync -- python tools/check_test_structure.py
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
既有 `check_file_size.py` 提供大小訊號，無須另設一套測試檔行數門檻。

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

## 判讀一次測試執行

每次驗證記錄候選 commit、環境、測試選集、xdist 排程、exit status 與原始 log。collection 相同或一次通過不保證沒有順序依賴及 Qt 資源生命週期問題；worker 中止、間歇失敗與警告需要在相同選集重新調查，不能因個別重跑通過就宣告原失敗解除。歷史候選的通過數與失敗案例不是現況基準，也不是豁免。
