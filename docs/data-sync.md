# 多台電腦的資料同步

**讀者：** operator、agent。設定雲端根、同步量測資料，並在彙整機開啟各個來源。

**Last updated:** 2026-11-25

## 資料放在哪裡

雲端根與彙整機的 `<labdata>` 底下用相同的相對路徑。每個來源 prefix 是一個獨立的 project root。例如 `lab-drive:QubitData/alice/pc1/sim/` 與本地 `~/QubitData/alice/pc1/sim/` 下都有 `result/` 和 `Database/`。

量測端不必搬資料。`push` 從這個 checkout 的 `result/` 與 `Database/` 讀取，只寫設定的雲端來源。`pull` 只寫本地彙整樹，不寫回雲端，也不會複製到 checkout 內或覆蓋其資料根。舊 Labber 目錄、非 entry 資料夾和 `Database/` 根的零散檔案都照原路徑複製。工具不改格式，也不會讓原本不支援的檔案出現在 GUI 的 Load 清單。

來源可用一段或多段名稱，例如 `alice`、`pc1/alice` 或 `alice/pc1/sim`。段與段用 `/` 分隔，不能有空段、`.`、`..`、反斜線，或名為 `result`、`Database` 的段。兩個 checkout 填同一個 prefix 就會寫進同一個來源。請替每個來源選不同的 prefix。改名或重灌後，若改用新主機名當 prefix，就會新增雲端來源，舊目錄仍在。

## 第一次設定

在 repo root 執行。建議用 `<短主機名>/<使用者>` 當來源，但 `--source` 可以給任何合法 prefix。

```bash
uv run --directory . --no-sync -- python scripts/sync_data.py setup --source pc1/alice
```

1. 首次啟動會下載並驗證固定版 rclone，再開啟官方 Web GUI。`setup --no-browser` 只印網址。網址含本機登入憑證，不要分享。
2. 在 **Remotes → Add New Remote** 選雲端服務並完成授權。已有 rclone remote 可直接重用。Google Drive 的共用 OAuth client 有 2026 年停用警告。若共用 client 無法授權，請用[自有 client](https://rclone.org/drive/#making-your-own-client-id)。只需拉資料的帳號可設定 `drive.readonly`。
3. 在 **Explorer** 進入雲端根，例如 `lab-drive:QubitData`，按 **Refresh**。不要進到來源目錄，也不要進到 `result` 或 `Database`。回終端機核對印出的路徑，按 Enter 確認。
4. 終端機詢問本地彙整根。直接 Enter 使用 `~/<雲端根資料夾名>`，例如 `~/QubitData`；選了整個 remote 時改用 remote 名稱。可以輸入 `../QubitData`，也可以加 `setup --labdata ../QubitData` 略過這個提示。相對路徑以命令執行時的工作目錄為準。
5. 核對終端機印出的雲端根、來源與本地絕對路徑，按 Enter 保存。Ctrl+C 取消，不覆寫既有位置設定。帳號變更由 rclone 即時保存。

![官方 Google Drive 連線表單](images/data-sync/drive-form.jpg)

每個 checkout 的 `.data-sync/sync.json` 保存位置，不含 token。`rclone.conf` 沿用使用者層設定，在同一使用者的 checkout 間共用。它含帳號憑證，不要提交或分享。改位置就重跑 `setup`。工具不讀取舊 checkout 傳輸設定。

## 上傳與下載

```bash
uv run --directory . --no-sync -- python scripts/sync_data.py status --json
uv run --directory . --no-sync -- python scripts/sync_data.py sources
uv run --directory . --no-sync -- python scripts/sync_data.py push --dry-run
uv run --directory . --no-sync -- python scripts/sync_data.py push
uv run --directory . --no-sync -- python scripts/sync_data.py pull alice/pc1/sim bob --dry-run
uv run --directory . --no-sync -- python scripts/sync_data.py pull --all
```

`status --json` 只讀本地設定，不測試雲端連線或授權。未設定時回傳 1。未設定的 `push`、`pull`、`sources` 回傳 2，提示先執行 setup，不開瀏覽器。`sources` 列出直接含 `result/` 或 `Database/` 的來源。`pull --all` 使用同一份清單，跳過本 checkout 的來源。巢狀來源如 `alice` 和 `alice/pc1/sim` 可以並存，各自只傳自己的兩棵資料樹。

傳輸使用 rclone `copy --update`，不刪目的地檔案，排除 `*.lock`。目的地較新的同名檔案會保留。工具不合併 YAML 或 ledger 內容；請勿讓兩台電腦寫同一來源的同名檔案。執行 `--dry-run` 可先看要複製的檔案。空目錄不會單獨建立，`push` 略過不存在的資料根。

**量測中上傳可能帶走未寫完的檔案。** 等 Run 結束後再 push 一次。

量測機若還在用舊 checkout，也可以用已設定雲端帳號的 rclone 直接上傳。請先設定下例的 `checkout`、`cloud_root` 與 `prefix`：

```bash
rclone copy --update --exclude '*.lock' "$checkout/Database" "$cloud_root/$prefix/Database"
rclone copy --update --exclude '*.lock' "$checkout/result" "$cloud_root/$prefix/result"
```

## 開啟外來來源

在彙整機以來源目錄當作 `--project-root`。路徑需用已存在的絕對路徑。不要傳 `<labdata>` 或 `Database/` 本身。

```bash
uv run --directory . --no-sync -- python -m apps.measure \
  --project-root "$HOME/QubitData/alice/pc1/sim"
```

`apps.fluxdep`、`apps.dispersive` 也用 `--project-root` 選資料根。MCP 開新 GUI 時填 `project_root`；已開啟的 GUI 不會因 MCP attach 改根。GUI 一次只開一個來源，要看別的來源就換 project root。在外來根執行 Save 會寫入本地副本。這些存檔不在 checkout 的上傳根內，不會透過這個 checkout 的 `push` 回傳。

## 把 checkout 資料搬進同構樹

可以保留原目錄。若想讓本機來源也和其他來源放在一起，先在該 checkout **push 一次並確認成功**，關閉所有使用這兩個目錄的 GUI。以下是 Linux／macOS 的例子，假設從 repo root 執行，設定來源是 `pc1/alice`，彙整根是 `../QubitData`，且目的地沒有同名的 `result` 或 `Database`。

```bash
mkdir -p ../QubitData/pc1/alice
mv result Database ../QubitData/pc1/alice/
ln -s ../QubitData/pc1/alice/result result
ln -s ../QubitData/pc1/alice/Database Database
```

連結的相對路徑從 checkout 解析。若 checkout 和 `QubitData` 隨上層目錄一起搬，連結仍指向同一批資料。`push` 讀這兩個連結的實體目錄。`pull` 比較實體路徑，不會透過別名寫進這兩棵樹。Measure 的唯讀 Results Browser 不跟隨 checkout 的資料根 symlink；要瀏覽這棵樹，請用實體來源目錄當作 `--project-root` 開啟。

以實體來源目錄執行升級工具時，預設備份放在來源旁，例如 `../QubitData/pc1/alice-backup/`。它不在 `result/` 或 `Database/` 裡，所以不會被 push。

固定版 rclone、快取與設定服務的維護方式見[同步模組說明](../lib/zcu_tools/utils/data_sync/README.md)。
