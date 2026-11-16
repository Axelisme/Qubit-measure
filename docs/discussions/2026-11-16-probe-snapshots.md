---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../../tools/README.md
  - ../../lib/zcu_tools/experiment/README.md
basis:
  repo: 2a537164567bb157550bfc82c043a5545e85c9ee
  external: none
---

# 實驗 probe 的 snapshot 保存與比較

## 結論

使用者選擇工具只做「存＋比」。同一支 probe script 在兩個 worktree 各用自己的 interpreter
執行一次，再比較兩份 snapshot。工具的使用與 exit code 由 [tools README](../../tools/README.md#probe-snapshots)
擁有。環境清理 helper 由 [experiment README](../../lib/zcu_tools/experiment/README.md#explicit-simulation-environments)
擁有。

## 問題

離線 probe 重複撰寫陣列保存、PNG 保存與比較程式。PNG 檔案的 metadata 可以不同而像素相同，
直接比較 bytes 會把這種情況報成圖形差異。只比較數字又會漏掉 cfg、seed 或來源資料的改變。

## 考慮過的選項與比較

- 只提供文件範例，讓各 script 複製保存及比較程式。這保留了重複工作。
- 提供獨立 writer 與 compare，讓 caller 決定實驗、輸入、執行順序及輸出位置。使用者採用此方案。
- 讓工具啟動兩個 worktree，自動執行 case。這增加環境隔離、case 載入與程序管理的維護成本。
- 在 stage 結果上加 write 方法。這讓 experiment 承擔 snapshot 檔案格式。

[ADR-0072](../adr/0072-module-boundaries.md#分層) 規定 run 與 analysis 回 PNG bytes，不寫磁碟，
experiment 不負責檔案格式。本次遵循這個邊界，writer 不依賴 stage 簽名。
環境清理放在既有 simulation owner，以獨立 context manager 包裝 backend 與 devices 的釋放，
不改 frozen dataclass 的 context-manager protocol，也不批次改寫現有 script。

## 被否決的選項與理由

兩樹自動執行、case 檔契約與並排圖不在使用者選定的範圍。Stage 結果的 write 方法違反既有模組定位。
測試 adapter 留待之後有兩個 task 實際需要時再提，不和 snapshot 工具一起加入。

## 留下的問題

輸入指紋只涵蓋 caller 提供的值，不證明 caller 已列出所有物理模型或環境設定。
Compare 報差異或明示容差內，不裁決物理正確性，也不取代逐張看圖。

## 依據

使用者於 2026-11-14 選擇只做保存與比較，於 2026-11-15 追認工具位置、獨立清理函式與比較語意。
2026-11-16，coordinator 依介面細節權限確認完整 exit code、必填 keyword-only inputs 與只寫新 label。
當時的 dossier 與原型未入庫；本文保留選項、限制與採用理由，不依賴那些工作檔。
