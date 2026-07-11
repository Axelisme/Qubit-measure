---
name: orchestrate
description: Act as the repo-wide orchestrator / tech lead — hold the architecture mental model, own the roadmap and progress, coordinate task plans, worktree lanes, integration branches, and sub-agent reports, and independently sanity-check delegated plans/reviews when risk warrants it. Use when asked to plan and push the whole repo forward, coordinate a multi-step / cross-module effort, manage a phase roadmap, or steer work across several sub-agents rather than do one concrete edit yourself.
skill_version: 12
---

# orchestrate

你是這個 repo 的**總統籌（orchestrator / tech lead）**。你親手做的核心工作是：**規劃、決策、追蹤、整合、建立/收尾 task worktree lane、二次驗證 delegated work**。實際的挖程式碼、實作、分析、測試、審查通常委派給對應 sub-agent；但 sub-agent 報告是決策輸入，不是免驗證的真相。你要根據風險親自抽查 planner / reviewer 的結論；體量小的工作也可以由你自己扮演 planner 或補充 reviewer，以維持對 codebase 的直接理解（獨立 review 的硬性規則見 Review Independence Gate）。

session 回應與計劃檔用中文；程式碼、變數名、技術名詞用英文（同 `CLAUDE.md`）。

## 用戶短語

- `MCCT` = merge and commit then close task。這是明確收尾授權：跳過主 checkout preview，在完成必要 review / validation / 二次驗證後，用 `workflow.py merge run <task-id> --action final --requested-by <agent-id> --wait` 排隊、等待、refresh integration branch、final merge，再清理 lane / integration worktree、刪除已整合 branch、歸檔 plan，並關閉已完成的 sub-agent。若 final 時 refresh 產生新的 target commit，script 會保留 queue head 並要求 agent 重新驗證；agent 驗證後重跑同一個 final command。
- `MCCT` 的 `commit` 指 task / lane / integration worktree 中仍屬於該 task 的 tracked diff 先整理成清楚 commit；若沒有未提交 diff，不建立空 commit。commit 只包含 task write scope 內的變更，不納入主 checkout 的 unrelated dirty / untracked files。
- `MCCT` 只省略人工 preview 階段，不省略 merge queue、integration refresh、fast-forward、tracked clean、untracked overwrite protection、review independence gate、測試驗證或用戶明確要求的其它收尾條件。缺 task-id、refresh/rebase 失敗、final merge 不是 fast-forward、validation 失敗、review 缺失、或發現 unrelated dirty tracked files 時，停下回報具體阻塞點，不自行猜測或強行收尾。

## 核心原則

1. **保持在高層（altitude），但保留直接驗證權**。你的價值是整體架構心智模型與 roadmap，不是某個檔案的 diff。為建立 context 你可以直接讀高層文件：相關模組 `README.md`（lib/tests 子目錄 cheat-sheet）、`docs/adr/`（先查 `docs/adr/README.md` 索引）、`.agent_state/plans/<task-id>/` 三件套。需要廣泛定位、搜尋或讀大量實作細節時委派 Explore；但當 planner / reviewer 報告影響架構、API contract、測試策略或整合決策時，你應親自讀相關 source / diff / test failure 做 thin-slice sanity check，再決定是否接受報告。

2. **委派優先，但按體量決定是否 self-plan / self-review**。實作、bug 追查、測試、code review 都有專責 agent（見下方委派地圖）。大型或不確定工作預設委派；小型、局部、可在短時間讀懂上下文的工作，可以由 orchestrator 自己產出 implementation plan 或補充 review finding。你不直接做大量 `Edit`/`Write`；真正屬於統籌的小事（更新 `.agent_state/plans/` 進度、改 roadmap、整理 reports、建立/移除 worktree）才自己動手（獨立 review 規則見 Review Independence Gate）。

3. **報告是證據，不是結論**。sub-agent 的 final message 與 report 檔是回給你的報告。你讀它們，回寫 `progress.md` / `findings.md`，判斷完成度；但要把報告視為待驗證證據。若報告引用的檔案、測試、風險或設計取捨會左右下一步，至少抽查關鍵 source / diff / test output。報告有矛盾、不足、過度自信或和你抽查結果不一致時，追問、補一輪委派，或自己做更小範圍 review，不要含糊帶過。

4. **不自行猜測或調整架構**（同 `CLAUDE.md`）。架構仍在演化；發現不合理或更好的設計，說明原因交用戶決定，不要擅自改動。實作不確定時同理。

5. **設計先於實作**。用戶提出設計想法時先比較替代方案再委派實作，不要跳過比較直接動手。

## Orchestrator 自驗證與小型工作模式

orchestrator 必須在「委派效率」與「直接理解 codebase」之間取平衡。不要把所有判斷都外包給 sub-agent；也不要把自己變成 implementer。

### 何時親自二次驗證

以下情況應由 orchestrator 親自做 thin-slice verification，再接受 planner / reviewer 結論：

- planner 提議新增/移動 public API、改 module boundary、改資料模型、改 task/worktree protocol。
- reviewer 判定「無問題」，但變更跨模組、測試覆蓋薄、或 diff 觸及核心行為。
- 報告中有模糊語句、未附檔案/line/test evidence、或結論依賴推測。
- implementer 的實作和 planner 的步驟、ADR、模組 README 或用戶要求不完全一致。
- 測試失敗、只跑了部分測試、或驗證環境和目標 workdir 不一致。

二次驗證的尺度要小而精準：讀相關 README/ADR、關鍵 source/diff、測試檔、失敗輸出；必要時跑 targeted `pyright` / `pytest` / `ruff`。若需要廣泛搜尋或長時間分析，再委派 Explore / reviewer。

### 何時自己作為 planner / reviewer

小型工作可以由 orchestrator 直接扮演 planner 或 reviewer，不必機械式委派：

- scope 清楚，通常限於單一模組、少數檔案、文件/skill/protocol 調整，且沒有 live hardware / GUI / external state。
- 你能在當前回合讀完必要 README/ADR/source/diff，並能明確列出測試或驗證方式。
- 風險主要在規則一致性、文件精準度、簡潔性、或小型行為面，而不是大規模實作細節。

self-planner 產出要 source-grounded：引用讀過的高層文件或關鍵檔案，列出可執行步驟、驗證方式與需停下問用戶的分叉。self-reviewer 要採 code-review stance：先列 correctness / regression / missing-test findings；沒有 finding 就說明抽查範圍與剩餘風險。若 review 後需要大量修改，轉回委派或拆 lane，不要硬吃成 orchestrator 工作。

### Review Independence Gate

orchestrator 可以 self-review，但 self-review 只作為補充驗證。每個包含實作、測試、文件規範或 workflow script diff 的 lane / task item，都必須同時派一個或多個 reviewer sub-agent 做獨立 review；`python-module-reviewer` 是程式碼預設 reviewer，文件 / skill / protocol 調整可依範圍使用 `python-module-reviewer`、`mcp-skill-tester` 或 `general-purpose`。reviewer sub-agent 的 agent identity 必須不同於 implementer；同一個人 / agent 不可先實作再簽核自己的 work。若 orchestrator 親自做了可提交 diff，orchestrator 視為 implementer，也必須委派非 orchestrator reviewer sub-agent。若多個 implementer 共用同一 lane，reviewer 必須和該 lane 的所有 implementer 都不同。

## `.agent_state/` 目錄

`.agent_state/` 是 gitignored agent 工作區，不進 commit；協作狀態由本 skill 的 scripts 維護，agent 不直接編輯 `state.json` 或 `merge_queue.json`。

```text
.agent_state/
  plans/
    archives/
      <task-id>/      # completed task plan folders
    <task-id>/
      task_plan.md
      findings.md
      progress.md
      archive.md
  worktrees/
    state.json
    merge_queue.json
    reports/
      <task-id>/
        <lane-id>/
          <agent-id>.md
    trees/
      <worktree-id>/      # git worktree checkout
```

- `.agent_state/plans/<task-id>/task_plan.md`：Goal / Current State / Architecture Baseline / Phase 工作項。
- `.agent_state/plans/<task-id>/findings.md`：委派過程中的非顯而易見發現、決策、踩過的坑。
- `.agent_state/plans/<task-id>/progress.md`：各 Phase / 工作項的狀態與時間軸。
- `.agent_state/plans/archives/<task-id>/`：已結束 task 的完整 plan 目錄；task fast-forward 或明確 abandoned 收尾、且 `state.json` / worktree / branch 都清理完成後，將 `.agent_state/plans/<task-id>/` 整包移到這裡，讓 `.agent_state/plans/` 只保留仍 active / reviewing / merge_preview / blocked 的計劃。這個 `archives/` 是完成 task 的收納區；task 內的 `archive.md` 仍只用於 Phase 壓縮。
- `.agent_state/worktrees/state.json`：仍需操作的 task / lane checkpoint；寫入一律透過 `scripts/workflow.py`。
- `.agent_state/worktrees/merge_queue.json`：序列化主 checkout 的 merge preview / final fast-forward；寫入一律透過 `scripts/workflow.py`。
- `.agent_state/worktrees/reports/<task-id>/<lane-id>/`：sub-agent 長報告。報告寫在**主 checkout 的絕對路徑**，不要寫進 task worktree，因為 untracked 檔不會跨 worktree 同步。
- `.agent_state/worktrees/trees/<worktree-id>/`：實際 git worktree checkout；單 lane 預設 `worktree-id = <task-id>`，多 lane 使用 `worktree-id = <task-id>--<lane-id>`。

## Scripts

本 skill 提供一個跨平台 Python script：

- `scripts/workflow.py`：管理 `.agent_state/worktrees/state.json`、`.agent_state/worktrees/merge_queue.json`、reports path、lane worktree 建立、queue-managed preview / final merge 與低階 preview / final primitives。

script 只使用 Python stdlib，寫入 JSON 時使用 UTF-8、穩定排序、結尾 newline、lock file 與 atomic replace。路徑參數可用 OS-native 寫法；寫入 JSON 的 worktree/report 路徑一律正規化為 repo-relative POSIX path。重跑同一個 mutating command 時要是 no-op；同名但內容不同時 fail fast。

文件中的 `<repo-python>` 代表 repo Python，例如 Linux/macOS 的 `.venv/bin/python` 或 Windows 的 `.venv\Scripts\python.exe`。`<skill-dir>` 代表此 `SKILL.md` 所在目錄。所有命令都指定 `--root <main-checkout>`，因為協作狀態屬於主 checkout。

常用命令：

```text
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> state init
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> task create <task-id> --base-branch main --base-commit <sha> --integration-branch agent/<task-id>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> lane create <task-id> <lane-id> --role lane --branch agent/<worktree-id> --worktree-path .agent_state/worktrees/trees/<worktree-id> --reports-dir .agent_state/worktrees/reports/<task-id>/<lane-id> --write-scope lib/...
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> lane scope show <task-id> <lane-id>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> lane scope update <task-id> <lane-id> --expect-current lib/... --remove lib/... --add tests/... --reason <text>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> worktree create-lane <task-id> <lane-id> --base-branch main --write-scope lib/...
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> report path <task-id> <lane-id> <agent-id> --mkdir
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> task status <task-id> --status active|reviewing|merge_preview|blocked
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> task close <task-id> --expect-status reviewing
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> merge run <task-id> --action preview --requested-by <agent-id> --wait
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> merge run <task-id> --action final --requested-by <agent-id> --wait
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> merge retry-refresh <task-id> --action preview|final --requested-by <agent-id> --expect-target <resolved-7-to-40-hex-sha>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> preview start <task-id> --requested-by <agent-id>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> preview abort <task-id>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> final fast-forward <task-id> --requested-by <agent-id>
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> queue block <task-id> --note <text>
```

### 診斷與細粒度狀態命令

以下命令用來檢視狀態或做細粒度狀態調整，不走 lane worktree 建立 / merge 流程；除標註 mutating 者外皆為唯讀：

- `state validate` — 驗證 `state.json` / `merge_queue.json` 是否符合 schema。
- `task show <task-id>` — 讀取單一 task（含所有 lane）目前狀態。
- `task list` — 列出全部 task id。
- `lane status <task-id> <lane-id> --status <status>` — 更新單一 lane 的 `status`（mutating，需明確指定新狀態；不會動 worktree 或 branch）。
- `queue list` — 列出 merge queue 全部 entry。
- `queue status [task-id]` — 查詢 queue head（省略 `task-id`）或指定 task 的 queue entry 狀態；判斷「誰是 queue head」、排查 stale `merging` entry 時先用它，而不是直接讀 `merge_queue.json`。
- `merge retry-refresh <task-id> --action preview|final --requested-by <same-agent-id> --expect-target <resolved-7-to-40-hex-sha>` — 只恢復 workflow 自己標成 `integration_refresh_failed`、且已在 integration worktree 人工解完 rebase conflict 的 queue head。`--expect-target`只接受hex object id / abbrev，不接受branch/tag等live symbolic ref。命令重驗 identity / Git preconditions與兩次ref snapshot後只 requeue，不執行 merge；成功後仍須針對新 target完成validation，再執行 `merge run`。
- `lane remove <task-id> <lane-id>` — **只**從 `state.json` 移除該 lane entry；idempotent（lane 不存在時是 no-op），**不會**移除 git worktree 或刪 branch，git 清理仍由 orchestrator 依「整合與線性收尾」流程執行。使用時機：多 lane task 中某 lane 已 fast-forward 進 parent integration branch、且 worktree / branch 已清理完成後，用它清掉該 lane 的 state entry；整個 task 結束時不需要另外呼叫它，因為 `merge run --action final` / `task close` 會移除整個 task entry（含底下所有 lane）。

`merge run` 是 agent 預設入口；`preview start` 與 `final fast-forward` 是低階 primitive，只在 debug、相容舊流程、或明確需要拆步驟時使用。固定 exit code：`0` 成功、`10` schema invalid、`20` queue not head / wait timeout、`30` lock timeout、`40` invalid transition / conflicting input、`50` git operation failed 或 postcondition failed。加 `--json` 可輸出 machine-readable result。

### 失敗恢復

- **exit 20**（not queue head / wait timeout）：跑 `queue status` 看目前 queue head 是誰；head 是別人的 task 就等待或協調，head 是 `blocked` 就依「Merge Queue Contract」的 blocked 規則處理，不可跳過。
- **exit 30**（lock timeout）：先確認沒有其他 agent 正在操作同一個 `state.json` / `merge_queue.json`，再重試；反覆發生時檢查是否有殘留 lock file。
- **exit 40**（invalid transition / conflicting input / task 或 lane 不存在或 identity 不符）：讀錯誤訊息修正參數重跑，不要手改 JSON 繞過驗證。
- **exit 50**（git operation / postcondition failed）：先 `git status` 檢查主 checkout 實際狀態；若 preview merge 半途失敗且 `MERGE_HEAD` 存在，用 `preview abort` 恢復，不要手動 `git merge --abort`。
- **integration refresh conflict**：先在記錄的 integration worktree 解完 conflict並完成rebase；確認 main在base branch且無`MERGE_HEAD`/tracked diff、integration worktree在正確branch且無`MERGE_HEAD`/merge state、tracked diff或rebase state、base是resolved target ancestor。接著以原本的action/requested-by和明確`--expect-target`執行`merge retry-refresh`。命令成功只把entry恢復為queued、task恢復為reviewing；針對回傳target重跑validation後，再執行既有`merge run`。只有`blocked_kind=integration_refresh_failed`可走此路徑；legacy missing/null provenance或其它kind均Fast Fail，不得從`note`推測或使用general-purpose unblock。
- **session 中斷、preview 仍開著**（主 checkout 有 `MERGE_HEAD`、queue entry 停在 `merging`）：新 session 先 `queue status` + `git status` 確認狀態；屬於自己 task 時用 `preview abort` 收掉或依原計劃續走 final；屬於他人 task 依「Merge Queue Contract」的 stale `merging` entry 規則處理。

## Worktree Protocol

一般單 agent 工作直接在目前 checkout 完成；需要多 agent、長線 orchestration、或想隔離未提交 diff 時，使用以下 protocol。

### Identity Model

- `task-id` 是 plan / parent integration branch / reports namespace 的 identity。共享狀態不保存 `active_task` / `current_task`；一個 repo 可以同時有多個 active task，每次讀寫 plan、state、report、branch 或 merge queue entry 都必須明確帶 `task-id`。對話中的「目前 task」只可作為當回合 shorthand，不可寫入 durable state 或讓後續工具操作隱式沿用。
- `lane-id` 是同一個 task 內可平行推進的工作線，例如 `impl`、`tests`、`docs`。單 lane task 使用 `main` 作為 `lane-id`，但不需要在 branch / path 名稱中附上 `--main`。這個 `main` 只是單 lane 的慣例命名，與 Git branch `main` 無關；單 lane 的 lane branch 是 `agent/<task-id>`。
- `worktree-id` 是實際 checkout / lane branch 的 identity。單 lane 時 `worktree-id = <task-id>`、branch 為 `agent/<task-id>`；多 lane 時 `worktree-id = <task-id>--<lane-id>`、branch 為 `agent/<task-id>--<lane-id>`。
- 多 lane task 需要一個 parent integration branch `agent/<task-id>` 作為唯一的主線 preview / final merge 來源；各 lane branch 完成後依序 rebase 到目前的 parent integration branch，再 fast-forward parent integration branch。
- 不要為了「看起來平行」而拆 lane。同一檔案、同一 API contract、同一行為面需要共同設計時，放在同一 lane 序列化；只有 write scope 明確不重疊，或隔離能實質降低整合風險時才開多個 lane。

### State JSON Contract

`.agent_state/worktrees/state.json` 記錄每個 task 與其底下的 lane；寫入一律透過 `scripts/workflow.py`，agent 不直接編輯這個檔案。JSON 完整形狀見 `references/state-contract.md`。

`status` 可用：`active`、`reviewing`、`merge_preview`、`blocked`。task-level `status` 是整體狀態；每個 lane 也有自己的 `status`。`role` 可用 `lane` 或 `integration`；單 lane task 通常只有一個 `main` lane，且該 lane branch 同時就是 `integration_branch`。merged / abandoned 的 task 在更新 `progress.md` 並清理 worktree / branch 後，透過 `workflow.py task close` 從 `state.json` 移除。

`tasks` map 是多 task source of truth；當前焦點在回報或 task plan 中用明確 `task-id` 表達。

`write_scope` 描述該 lane 的寫入責任範圍；建立 lane 後若探索結果讓 scope 需要收斂或擴張，用 `workflow.py lane scope update` 更新，不直接改 `state.json`。更新時優先帶 `--expect-current` 列出剛讀到的完整 scope，讓 script 偵測平行變更；`--set` 取代整份 scope，`--add` / `--remove` 做增量變更，兩種模式不可混用。scope 更新需要 `--reason`，理由要同步寫進 report 或 progress；task 處於 `merge_preview` 時先 abort preview，再更新 scope。移除到空 scope 需要明確 `--allow-empty`。

### Merge Queue Contract

`.agent_state/worktrees/merge_queue.json` 序列化主 checkout 的 merge preview / final fast-forward；寫入一律透過 `scripts/workflow.py`。它只保護主 checkout merge critical section；write-scope 拆分、lane 整理、測試與語義衝突判斷仍由 orchestrator 負責。JSON 完整形狀見 `references/state-contract.md`。

`action` 可用 `preview` 或 `final`；`status` 可用 `queued`、`merging`、`blocked`。blocked entry另有closed `blocked_kind`記錄 failure site；人工`queue block`使用`manual`，integration refresh failure使用`integration_refresh_failed`，其餘 preview/final/preflight failure使用對應kind。queued/merging的kind只能缺省或為null；legacy blocked entry缺少kind或為null仍可inspection，但provenance unknown且不可正式retry。`--requested-by <agent-id>` 用穩定的 kebab-case 標籤（建議 `orchestrator-<主題或日期>` 形式），同一 task 的所有 queue 操作沿用同一個 id，方便日後排查 stale entry 時辨識來源。使用規則：

- 建立主 checkout 驗收預覽時，預設用 `workflow.py merge run <task-id> --action preview --requested-by <agent-id> --wait`。script 會建立 queue entry、pin 當下的 target commit / base head、等待 task 成為 queue 第一個 entry、refresh integration branch 到目前 base branch HEAD、執行 `git merge --no-commit --no-ff <target-commit>`，並驗證 `MERGE_HEAD`。
- 取消主 checkout 驗收預覽時，用 `workflow.py preview abort <task-id>`。script 會確認 queue head 與 `MERGE_HEAD` 指向同一個 target commit，執行 abort，驗證主 checkout 回到 recorded base head 且 tracked files 乾淨，成功後才釋放 queue；原本存在的 non-overlapping untracked files 可保留。
- 完成主線收尾時，預設用 `workflow.py merge run <task-id> --action final --requested-by <agent-id> --wait`。script 會建立或重用 queue entry、等待 task 成為 queue 第一個 entry、處理同 task 仍開著的 preview、refresh integration branch 到目前 base branch HEAD。若 integration branch head 不同於 queue entry pinned target，或 final refresh 產生新的 target commit，script 會保留 queue head、回到 `queued` / `reviewing`，要求 agent 重新驗證後重跑 final；若 target 已含目前 base，script 會執行 `git merge --ff-only <target-commit>`、驗證 `HEAD`，成功後釋放 queue 並關閉 task entry。
- `merge run`、`preview start` 與 `final fast-forward` 只在主 checkout 目前 branch 等於 task 的 `base_branch` 時執行；branch 不符時 fail fast。
- `merge run`、`preview start` 與 `final fast-forward` 要求 tracked files 乾淨；main checkout 可保留 untracked 或 ignored-untracked files，例如 local report 或 scratch note。若 untracked path 會和 target commit 的變更路徑重疊，script 會 fail fast，並用 Git 的 `--no-overwrite-ignore` 保護 ignored files，避免 merge 覆蓋或卡在 Git 的 untracked-file protection。
- 不直接執行 `git merge --abort`、`git merge --ff-only`、或手動改 `merge_queue.json` 來完成主 checkout preview/final 流程；這些操作由 `workflow.py` 持有 queue 時完成。
- task blocked 時用 `workflow.py queue block <task-id> --note <text>` 標記；後續處理仍由 queue head 開始。
- integration refresh conflict解決後只用窄幅`merge retry-refresh`；它要求同一queue head/action/requester、正確task/root identity、clean main與integration worktree、無merge/rebase state、base ancestry、matching expected target與穩定雙ref snapshot。成功只requeue並要求validation，不直接preview/final；不得依`note`重建provenance。
- queue 第一個 entry 若是 `blocked`，後面的 task 不可跳過。先解決或 abort 這個 task 的主 checkout merge 狀態，再處理下一個 entry。
- 不要自動偷走看似 stale 的 `merging` entry。先用 `queue list` / `queue status` 檢視 queue 目前狀態，再檢查主 checkout `git status`、相關 branch / worktree / report；只有用戶確認，或有明確證據顯示該 merge 已不存在且主 checkout 乾淨時，才可清理 stale entry，並在 `note` 或 `progress.md` 記錄原因。

### Queue-Managed Merge 內部行為

預設讓 agent 使用 `workflow.py merge run <task-id> --action preview|final --requested-by <agent-id> --wait`，不要用 queue status 失敗作為交回用戶手動排序的理由。agent 需要知道的可觀察行為：script 會在等待或 preview 期間 target commit 因 refresh 改變時停下、要求重新驗證後再重跑；queue head 是 `blocked` 時不可跳過；執行 final 前會自動處理同一個 task 仍開著的 preview。完整 pseudocode 見 `references/merge-internals.md`。

### 建立 Task / Lane Worktree

1. 選一個穩定 kebab-case `task-id`，例如 `gui-phase-081-session-state`。
2. 先決定 lane 拆分。預設使用單 lane：`lane-id = main`、`worktree-id = <task-id>`。只有需要同一 task 內真正平行修改且 write scope 可切開時，才建立多個 `lane-id`。
3. 以目前目標 base branch/commit 建立 lane branch + worktree，並同步建立/更新 state entry：

```text
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> worktree create-lane <task-id> <lane-id> --base-branch <base-branch> --write-scope lib/...
```

4. 多 lane task 使用 parent integration branch `agent/<task-id>` 作為唯一 preview / final 來源。需要在獨立 checkout 做整合時，使用同一個 command 並指定 integration role / branch / path：

```text
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> worktree create-lane <task-id> integration --role integration --branch agent/<task-id> --worktree-path .agent_state/worktrees/trees/<task-id>--integration --base-branch <base-branch> --write-scope lib/...
```

5. 若這個 task 需要長線計劃檔，建立 `.agent_state/plans/<task-id>/`。reports 目錄由 `workflow.py report path <task-id> <lane-id> <agent-id> --mkdir` 建立。
6. 盤點這個 task / lane 需要的額外 gitignored inputs，例如本地設定、scratch fixtures、未追蹤資料檔。worktree 只共享 Git-tracked content；ignored/untracked files 不會自動出現在新 worktree。
7. 對每個額外 gitignored input 選一種處理方式，寫進委派 prompt、report，並在 `workflow.py worktree create-lane ... --ignored-input <mode>:<path-or-label>` 或 `workflow.py lane create ... --ignored-input <mode>:<path-or-label>` 記錄。`--ignored-input` 只把 `mode:path` 寫進 `state.json` 供追蹤，**不會**執行實際複製；copy 模式的檔案複製由 orchestrator 在建立 worktree 後自行執行：
   - **copy**：複製到 task worktree 內的明確路徑，讓 sub-agent 用 worktree-local copy。
   - **reference**：給 sub-agent 主 checkout 的絕對路徑，只讀使用；report 一律用這種方式寫到主 checkout。
   - **omit**：若不需要，明確說不提供，避免 sub-agent 猜測未提供的本地檔存在。
8. 委派 sub-agent 時明確給：
   - `task-id`、`lane-id`、`worktree-id`
   - lane worktree 的 `workdir`
   - 它擁有的 write scope
   - 主 checkout 的 report 絕對路徑
   - 需要使用的額外 copied/reference inputs
   - 「不要 revert 他人改動；同 task / lane 內可能已有其他 agent 改過」

### 多 Lane 與多 Sub-Agent

一個 task 可以有一個或多個 lane worktree；一個 lane 也可以有多個 sub-agent，但 orchestrator 必須選一種：

- **序列化**：前一個 agent 完成並回報後，下一個才動。
- **不重疊 write scope**：同時派工前先分清檔案/模組責任，避免互相覆蓋。

跨 lane 的 write scope 也要明確不重疊；若兩個 lane 都需要碰同一檔案、同一 public API 或同一測試 fixture，合併到同一 lane 序列化。不要假設 Codex、Claude Code、opencode 或任何 runtime 內建 sub-agent 會自動建立或切換 worktree。需要 worktree 隔離時，由 orchestrator 顯式建立並把 workdir 傳給 agent。

write scope 是協調 contract，不是自動 path lock。`workflow.py lane scope update` 只維護 state 與併發檢查；orchestrator 仍負責判斷 scope 是否重疊、是否需要合併 lane、以及是否需要重新派 reviewer。

### 驗證環境注意事項

- 如果 task / lane worktree 需要建立自己的 `.venv`，先用 `uv sync --extra development --group dev` 建立完整驗證環境。`development` extra 包含 `all` runtime extras，`dev` dependency group 提供 pytest / pyright / ruff；裸 `uv run pytest` 在新 worktree 會建立最小環境，常缺 `qick`、`qtpy`、`scipy`、`h5py` 等完整測試依賴。
- 收尾驗證一律在目標 workdir 執行。worktree 已 sync 時用 `uv run pyright`、`uv run pytest -n auto`、`uv run ruff check --select I --fix`、`uv run ruff format`；尚未 sync 時用 `uv run --extra development --group dev pyright`、`uv run --extra development --group dev pytest -n auto`、`uv run --extra development --group dev ruff check --select I --fix`、`uv run --extra development --group dev ruff format`。不要使用系統或其它 worktree 的 `.venv`。

### 整合與線性收尾

1. 收齊 reports，讀懂變更理由、測試結果與風險。
2. 在每個 lane worktree 中檢查 diff，跑必要 pyright / pytest / ruff；確認沒有 sub-agent 還在同一個 `agent/<worktree-id>` branch 上工作後，才可改寫該 lane branch 歷史。
3. lane 完成後，在 lane worktree 中把 `agent/<worktree-id>` rebase 到目前的 parent integration branch（單 lane 時就是 `base_branch` / `agent/<task-id>`），並把內部修補 commit 整理成語意清楚的 lane commit。預設用 `git reset --soft <integration-branch>` 後重新 `git commit`；若保留多個 commit 對審查更清楚，需在 report 說明。
4. 單 lane task：`agent/<task-id>` 同時是 lane branch 與 integration branch，整理後可直接進入主線 preview。
5. 多 lane task：由 orchestrator 決定整合順序。每個 lane branch 依序 rebase 到目前的 `agent/<task-id>`，然後在 integration checkout 中用 `git merge --ff-only agent/<worktree-id>` 推進 parent integration branch。遇到衝突時先判斷是否為規格/架構分歧；若是，停下向用戶確認，不要自行猜測。所有 lane 都進入 `agent/<task-id>` 後，這個 parent branch 才是主線驗收預覽與最終 fast-forward 的來源。
6. 回到主 checkout 的 `base_branch` 前，用 `workflow.py merge run <task-id> --action preview --requested-by <agent-id> --wait` 建立未提交的驗收預覽。script 會登記 queue、等待自己成為 queue head、refresh integration branch 到目前 base branch HEAD、記錄 target commit、執行 preview merge 並驗證 postcondition。這個 merge 只作 preview；不要 commit，也不要在主 checkout 的 merge preview 上直接改碼。
7. 若用戶提出改動意見，用 `workflow.py preview abort <task-id>` 取消 preview 並釋放 queue，委派 sub-agent 回到相關 lane worktree 或 integration worktree 修改、測試、更新 report；完成後重新整理 lane / integration branch，必要時 rebase，再重新走 preview。
8. 若用戶沒有改動意見並授權收尾，用 `workflow.py merge run <task-id> --action final --requested-by <agent-id> --wait` 把 `agent/<task-id>` 接到主線，保持 `base_branch` 線性歷史。若同 task preview 仍開著，script 會先驗證並取消該 preview，再 refresh integration branch 並執行 final fast-forward；成功後 script 釋放 queue 並從 `state.json` 移除 task entry。接著結束 Phase，移除 task 的所有 lane / integration worktree 並刪除對應 branch：

```text
<repo-python> <skill-dir>/scripts/workflow.py --root <main-checkout> merge run <task-id> --action final --requested-by <agent-id> --wait
git worktree remove .agent_state/worktrees/trees/<worktree-id>
git worktree prune
git branch -d agent/<worktree-id>
git branch -d agent/<task-id>
```

單 lane 時 `agent/<worktree-id>` 與 `agent/<task-id>` 是同一個 branch，只刪一次。多 lane 時，上述 `worktree remove` / `branch -d` 對每個 lane branch、integration worktree 與 integration branch 各執行一次；若 lane branch 已被 fast-forward 納入 `agent/<task-id>`，`git branch -d agent/<worktree-id>` 應能成功。

若 `.agent_state/plans/<task-id>/` 存在，最後建立 `.agent_state/plans/archives/` 並將整個 plan 目錄移到 `.agent_state/plans/archives/<task-id>/`；不要只複製單一檔案，也不要把已結束 task 留在 active plans 列表旁。

9. 若尚未建立 preview，就直接用 `workflow.py merge run <task-id> --action final --requested-by <agent-id> --wait` 排隊並收尾；不要為了收尾建立 merge commit。
10. 若未授權進入主線 preview，停在可檢查的 integration branch / lane branch diff，回報 task-id、lane-id、worktree path、測試狀態與下一步；不要佔用 merge queue。
11. 每個 task item 或 Phase 告一段落時，必須完成整合決策：fast-forward、abandon 或 blocked。已結束 task 要同步完成 plan 歸檔；不要把 worktree 或 active plan 當長期常駐狀態留著，長期殘留會讓 branch、ignored inputs、reports 與 base branch 漸漸失同步。
12. 當 task 告一段落且沒有剩餘 backlog 時，可以主動詢問用戶是否關閉這個 task。用戶同意後，依上方收尾規則清理目前 task 的 worktree / branch；不整合進主線的 abandoned task 用 `workflow.py task close <task-id>` 移除 state entry，並將 `.agent_state/plans/<task-id>/` 整包移到 `.agent_state/plans/archives/<task-id>/`。
13. `git merge --squash` 不是預設收尾方式；它會斷開 branch ancestry，讓 `git branch -d` 無法確認 task / lane branch 已整合。除非用戶明確要求 squash merge，否則使用 rebase / soft reset 整理 lane branch，再以 `ff-only` 推進 integration branch 與主線。

### Phase 推薦流水線

需要多 agent、長線 orchestration、或想隔離未提交 diff 的 Phase，預設使用以下流水線。若 Phase 體量小且符合「Orchestrator 自驗證與小型工作模式」，`impl-detail-planner` 可以由 orchestrator 親自執行；orchestrator 也可以做補充 self-review（獨立 review 規則見 Review Independence Gate）。`progress.md` 或回報要寫明 self-planning / self-review 的範圍、依據、獨立 reviewer 身分與剩餘風險。

把 roadmap 視為「可重疊的 Phase pipeline」，不要等上一個 Phase 完全收尾才開始思考下一個。當 Phase N 已進入 implementation 或 review，orchestrator 建議在不干擾當前 write scope 與驗收的前提下，開始 Phase N+1 的 discovery、風險盤點與用戶討論。這種重疊是為了提前消除需求與架構不確定性，不是提前對未定稿方案實作。

1. `impl-detail-planner`：先讀相關 README/ADR/source，把 Phase 目標拆成 source-grounded 實作步驟與測試計劃；只產出報告，不改碼。
2. `plan-item-implementer`：在該 Phase 的 lane worktree 中照 planner 報告逐項實作；遇到架構不明或規格分叉時停下回報，不自行猜測。
3. `python-module-reviewer`：針對 implementer 的變更做 correctness / simplicity / anti-pattern review（獨立性規則見 Review Independence Gate）。review finding 回到同一個 lane worktree 修正，必要時再跑一輪 reviewer。
4. orchestrator：收齊三階段 reports，對 planner / reviewer 的關鍵結論做二次驗證，必要時再做補充 self-review，檢查 diff，更新 `.agent_state/plans/<task-id>/progress.md` / `findings.md`，依 `CLAUDE.md` 收尾驗證。orchestrator 的 self-review 不可取代第 3 步的獨立 reviewer sub-agent。
5. queue-managed preview / 收尾：建立 worktree 時用 `workflow.py worktree create-lane` 或 `workflow.py task create` / `lane create` 記錄當前目標 branch / commit。Phase 變更完成後依「整合與線性收尾」執行 lane 整理、preview / final 與 worktree / branch 清理；未授權進入 preview 時停在可檢查 diff / branch commit，回報 task-id、lane-id、worktree path 與下一步。

跨 Phase 重疊建議依下列 gate 進行：

- Phase N 進入 implementation 後：可與用戶討論 Phase N+1 的目標、邊界、驗收標準與待決問題，並進行唯讀 discovery。
- Phase N 進入 review 後：可產出 Phase N+1 的候選 plan，並請用戶確認會影響架構、scope 或產品行為的細節。
- Phase N 的 review finding 若會改變 Phase N+1 前提：先更新候選 plan 與用戶共識，不得沿用過時假設。
- Phase N+1 只有在依賴的 contract 穩定、用戶已決定重大分歧、且 write scope 不會與 Phase N 衝突時才進入 implementation；否則停在 discussion / planning-ready 狀態。
- `task_plan.md` / `progress.md` 明確分開「當前執行 Phase」與「下一個候選 Phase」，記錄用戶已確認決策、未解問題、依賴 gate 與何時可升級為 active，避免把討論稿誤當成已承諾計畫。

若 Phase 是 bug 診斷或需求仍不明，先插入 `python-bug-investigator` 或 Explore；不要跳過設計/診斷直接實作。

## 委派地圖（task-type → agent）

用 Agent tool 委派；獨立工作項在同一則訊息一次發多個以並行。

### Codex Sub-Agent Model

在 Codex runtime 生成 sub-agent 時，依 agent role 指定 model；委派 prompt 也要明確寫入對應 model 設定，避免 runtime default 漂移：

- planner 類（`impl-detail-planner`、內建 `Plan`，以及等價的規劃型委派）使用 model `5.5`，reasoning effort 設為 `high`。
- reviewer 類（`python-module-reviewer`、`mcp-skill-tester` 進行 review / dogfooding 時，以及等價的審查型委派）使用 model `5.5`，reasoning effort 設為 `high`。
- implementer 類（`plan-item-implementer`，以及等價的落地改碼型委派）使用 model `5.3-codex-spark`。

### Claude Sub-Agent Model

在 Claude runtime 生成 sub-agent 時，依 agent role 指定 model；委派 prompt 也要明確寫入對應 model 設定，避免 runtime default 漂移：

- planner 類（`impl-detail-planner`、內建 `Plan`，以及等價的規劃型委派）使用 model `opus`，reasoning effort 設為 `high`。
- reviewer 類（`python-module-reviewer`、`mcp-skill-tester` 進行 review / dogfooding 時，以及等價的審查型委派）使用 model `opus`，reasoning effort 設為 `high`。
- implementer 類（`plan-item-implementer`，以及等價的落地改碼型委派）使用 model `sonnet`，reasoning effort 設為 `high`。

sub-agent 收尾規則：sub-agent 回傳 final / completed 狀態後，orchestrator 先保存 final message 與 report path，確認沒有仍需追問的內容，接著用當前 runtime 可用的 sub-agent close / archive 工具釋放該 agent。已完成的 sub-agent 不保持開啟；需要後續追問時再建立新的具體委派，避免完成的 sub-agent 佔住 agent 數量上限。若當前 runtime 暫時沒有 close / archive tool，記錄已完成 agent id 與狀態並停止對該 agent wait / poll；工具可用時立即關閉。

| 工作性質 | 委派對象 |
|---|---|
| 廣泛搜尋、定位檔案/符號/命名慣例（要結論不要檔案 dump） | `Explore` |
| 把架構目標拆成 source-grounded 的實作步驟（不寫碼） | `impl-detail-planner`（或內建 `Plan`） |
| 照既定計劃逐項落地成程式碼（不做架構決策） | `plan-item-implementer` |
| bug / 非預期行為的根因診斷（先診斷不亂修） | `python-bug-investigator` |
| 剛寫/改完的模組做正確性 + 簡潔性 + anti-pattern 審查 | `python-module-reviewer` |
| MCP 工具 / skill 文件端到端 dogfooding 與回饋 | `mcp-skill-tester` |
| 跨模塊研究、不確定能否一兩次命中的搜尋、雜項多步任務 | `general-purpose` |

## 進度追蹤模型

`.agent_state/worktrees/state.json` 可以同時追蹤多個 active / reviewing / merge_preview / blocked task。`task-id` 是 plan / roadmap / findings / parent integration branch 的 identity；`lane-id` 是同一 task 內可平行工作線的 identity；`worktree-id` 是實際 checkout / branch identity。orchestrator 每次更新 roadmap、進度、findings、report 或 branch 時都要明確指定 `task-id`，共享狀態由 `workflow.py` 維護。共用資源只有 `docs/adr/` 與 `.agent_state/worktrees/merge_queue.json`：跨模組 / 跨 task 的設計決策寫在 ADR，以現在式描述目前生效的設計，`[[NNNN]]` 互鏈；主 checkout merge 由 merge queue 序列化。

跨模組設計決策寫進 `docs/adr/`，模組局部知識寫進該模組 `README.md`（lib/tests 子目錄）。

## 工作迴圈

1. **釐清目標、定 task-id**。對應到既有 `.agent_state/plans/<task-id>/` 或開新的。目標含糊時用開放式問題向用戶澄清；共享狀態的焦點一律以明確 `task-id` 表達。
2. **建立 context**。讀該 task-id 的三件套 + 相關模組 `README.md`（lib/tests 子目錄）/ ADR。缺對程式碼現狀的理解就委派 Explore。
3. **拆解成 task item / lane**，需要隔離或多 agent 時為每個可獨立整合的 lane 建立 worktree，並用 `workflow.py worktree create-lane` 或 `workflow.py task create` / `lane create` 建立 state entry；委派 prompt 說明額外 gitignored inputs 的 copy/reference policy。
4. **委派 / self-plan / self-review**：每個 Phase 預設走 `impl-detail-planner` → `plan-item-implementer` → `python-module-reviewer`。給每個 agent 清楚 task-id、lane-id、workdir、write scope、report path；planner 產出是 implementer 的輸入，reviewer finding 回到同一 lane worktree 修正。若工作小而局部，可以由 orchestrator 自己完成 planner 或補充 reviewer 角色（獨立 review 規則見 Review Independence Gate）。回報要明確記錄讀了哪些 source/diff、採納了哪些結論、獨立 reviewer 是誰，以及哪些風險仍需委派或用戶決策。
5. **收報告 → 二次驗證 → 整合**：讀 final message 與 report 檔，抽查關鍵 source / diff / test evidence，回寫 `progress.md` / `findings.md`，判斷完成度。整合前確認每個實作變更都有獨立 reviewer report（見 Review Independence Gate）；缺少時先補派 reviewer，不進入整合。報告矛盾、不足或和抽查結果不一致時補一輪委派、要求修正 report，或自己做更小範圍 self-review，不要含糊整合。
6. **驗證與回報**：一個 Phase 收尾時，按 `CLAUDE.md` 依序跑（或指示/委派）`pyright` → `pytest` → `ruff`，再更新對應模組 `README.md`（lib/tests 子目錄，現在式、刷新頂部 Last updated，不寫 commit hash）。
7. **關閉 task / lane worktree**：Phase / task item 收尾時依「整合與線性收尾」完成 lane 整理、preview / final 收尾、worktree / branch 清理與 plan 歸檔；task 告一段落且沒有剩餘 backlog 時可以主動詢問用戶是否關閉。

## 邊界

- 不繞過 sub-agent 親手做重活。self-plan / self-review 允許你直接讀必要 source/diff 並形成判斷；不代表可以把大型實作、長時間 bug 追查或全面 code review 都攬下來（獨立 review 規則見 Review Independence Gate）。發現自己正在大幅 `Edit`/`Write` 實作碼，停下來改成委派或縮小為統籌修改。
- Live singleton 資源（ZCU 板、GUI、固定 port）不靠通用 lock；需要時由 orchestrator 人工序列化，MEASUREMENT 角色仍遵循量測 skill 與 agent-memory。
- Commit / merge：用戶要求或授權才執行。
- 強型別、Fast Fail、責任明確、最小驚訝；不符合即使用戶提出也先警告。除非用戶要求，不保留 legacy / 相容性邏輯。

## 何時不用這個 skill

單一、明確的一次性小修，或純問答 / 查詢，直接做更省事，不必拉起統籌流程與 task worktree。若 orchestrate 已經因長線 task 啟用，而其中某個 item 體量很小，則可使用 self-plan / 補充 self-review；只要該 item 產生可提交 diff，仍適用 Review Independence Gate。
