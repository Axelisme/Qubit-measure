---
date: 2026-11-25
status: active
superseded_by: null
related: []
basis:
  repo: 97a062904dcb156ef8daba6228a55adf9e6b8fa3
  external: null
---

# 多台電腦使用同構資料樹

## 結論

雲端根和彙整機使用相同的 `<prefix>/{result,Database}` 布局。每個來源子樹都是一個 project root。量測端仍可直接使用 checkout 內的目錄，不必搬資料。`push` 只讀這個 checkout 的資料，目標由設定決定。外來資料的分析存檔留在彙整機。新工具取代兩支以 entry 為單位的傳輸腳本。操作契約見[資料同步指南](../data-sync.md)。

2026-11-25，使用者在採用同構樹時把時程提前到週一前：「雲端的我採用建議，但我希望能提早，因為除了模擬外其他要報告的會基於真實數據」。來源名最初打算固定為主機和使用者，之後改成任意相對 prefix。使用者說：「那我傾向不侷限於主機/用戶，而是任意prefix，因為有可能我會設計成用戶/主機/<可能不同的環境>，或是直接少掉主機這層」。本機資料根也不必放在 home。使用者說：「預設放QubitData沒問題，但這個主機我想放在../QubitData」。

## 問題

2026-11-24，使用者說：「可以假設我讓每個人與電腦的entry名都不同。」「這臺電腦只會分析與使用模擬器模擬，真實測量的數據都是從遠端拿，所以你可以注意到我的../Qubit-measure的Database格式不正式」。當時的兩支腳本各選一個雲端父目錄，只複製該目錄下的 entry 資料夾。舊的 Labber 日期樹、不屬於任何 entry 的資料夾，以及 `Database/` 根下的獨立檔案也要能依原樣搬運。

使用者先提出：「我雲端的格式規劃是<...>/<主機名>/<用戶名或repo名>/<Database與result>」。雲端依來源分層後，若本地仍平鋪，兩端就要互相轉換路徑。使用者說：「我煩惱的就是，這樣就破壞了原本簡單的同步假設，就是雲端的根與本地的根目錄結構相同，這樣遠端主機/雲端/本地主機要同步就很容易，這種設計張力很麻煩」。

## 比較過的選項

- **本地和雲端同構。** 讓每個來源保留自己的 `result/` 與 `Database/`。Measure、fluxdep、dispersive 和 MCP launch 已能分開指定程式 checkout 與資料根，不必改 GUI 才能逐來源開啟。代價是同一個 GUI 不會列出所有來源。Notebook 也需綁定目標 entry，或明確開啟所需的資料檔。
- **雲端分層，本地平鋪加 manifest。** 一個 GUI 就能列出所有標準 entry，但每次下載、改名或上傳都要維護 entry 與來源的對照。只擋上傳仍擋不住本地 Save。Save 依目前資料根與 run 裡的 entry 名稱寫入，不依 Load 的原始檔案位置。在平鋪的外來 entry 上執行 Save 會改寫本地副本，連 ledger 也可能被改動。

## 被否決的其他選項

- **雲端共用平鋪。** 全域名稱不重複可避免不同 entry 放進同一路徑，但不能擋非擁有者寫回既有 entry。`copy --update` 比較整個檔案，不會把兩台電腦各自追加的 ledger 行合併。
- **按 entry 打包。** `<根>/<entry>/{result,Database}` 可一起查看單一 entry 的檔案，卻會改變既有雲端布局，還把原本分開的 Result 改名與 Database 位置綁在一起。
- **唯讀 mount、Syncthing 或手動逐來源傳輸。** Mount 讀取 HDF5 仍要連網，Syncthing 需要另外維護方向與常駐程序。沿用兩支舊 CLI 則每個來源要設定兩次，而且會漏掉 `Database/` 根下不屬於 entry 的檔案。
- **固定 repo 名或穩定主機 ID。** 同名 clone 會撞 repo 名，改名也會換路徑。穩定 ID 可保住改名前的身分，但要額外維護登記、別名和交接。最後讓使用者選 prefix，不新增來源清冊。

## 留下的問題

Prefix 不是存取控制。若兩個 checkout 選同一個 prefix，仍可覆寫同名檔案。改名或重灌後使用新的 prefix，舊資料會留在原位置，交接需記下舊路徑與帳號。時鐘偏差也會影響 `--update`。彙整機上較新的分析檔在 pull 時會保留，但後來更新的雲端同名檔仍可取代它。工具不保證多寫者一致性，也不管理授權或配額。

## 依據

遵循 [ADR-0063](../adr/0063-persistence-ownership.md#run-輸出位置與-writer-權責)。Database 是輸出位置，不是 entry 身分。拆開來源依靠各自的 project root，不新增資料格式或把同名 Result／Database 再綁一次。也沿用舊檔不遷移的原則：同步只複製現有路徑，舊 Labber 檔仍用合適的 reader。舊 CLI 的逐入口選目錄是當時的做法，不是需要維持的契約。

調查與兩份挑戰原稿未入庫。第一份盤點各 launcher、Notebook 與工具對資料根的假設；第二份比較五年內重灌、交接、找回舊資料與誤上傳的成本。上述選項保存了影響裁決的差異。
