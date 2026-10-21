# 知識入口初始化

**讀者：** agent。核對並初始化獨立量測知識庫的入口。

知識入口是 `<repo>/measure_knowledge/`，由自身獨立 Git repo 管理。使用者可以用 symlink 讓不同 repo 共用；agent 不建立、改向或修復共享連結。

## 核對與初始化入口

首次使用或換 repo 時，先確認入口。不要因為是開發主機就啟動 GUI 或連線儀器，初始化只需要本機檔案與 Git。

- 已有目錄或有效 symlink：確認 `git -C <knowledge> rev-parse --show-toplevel` 的實體路徑正是知識目錄的實體路徑。Git 若向上找到外層 repo，不能當成初始化成功。確認 README.md 與 routing.md 可讀，沿用現有內容，不重套模板。
- 失效 symlink、一般檔案、非獨立 Git 目錄或入口文件缺失：回報具體狀態，保留內容，停止該初始化或修復動作。
- 路徑確實不存在，包含不是失效 symlink：按下面步驟建立。不猜測使用者的共享位置。

建立步驟：

1. 確認外層 Git 已忽略根入口，例如 `git -C <repo> check-ignore -- measure_knowledge`。若未忽略，先回報規則缺口，不在量測中自行修改程式碼 repo 的設定。
2. 使用不帶 `-p` 的 mkdir 建立目標，讓已有路徑的競爭修改直接失敗。從本 skill 的 [README 模板](../templates/knowledge/README.md) 與 [routing 模板](../templates/knowledge/routing.md) 複製兩份入口到新目錄，不覆寫現有檔案。相對模板路徑以本文件解析，不以 cwd 解析。
3. 在該目錄執行 `git init --initial-branch=main`，再確認 Git top-level 與實體知識目錄一致。不要在外層執行 git init，也不建立 remote。
4. 只 stage 新建立的 README.md 與 routing.md，檢查 diff，作一個本機初始 commit。沿用既有 Git identity；若尚未設定，回報並請使用者提供，不捏造作者或修改全域設定。
5. 確認知識庫自身狀態乾淨、入口可讀、外層仍忽略知識庫。第一條經驗出現時才建立 experiences 子目錄，不填入虛構案例。

中途失敗保留已建立內容並回報完成到哪一步，不刪除目錄重跑。下次應將它視為未完成初始化，不偽裝成全新缺失路徑。不同 agent 未安排寫入順序時，不同時初始化同一入口。
