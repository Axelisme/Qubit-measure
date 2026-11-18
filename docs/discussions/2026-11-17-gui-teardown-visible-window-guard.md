---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../../tests/README.md#gui-可見視窗的-teardown-guard
  - ../../tests/gui/conftest.py
  - ../../tests/test_gui_window_guard.py
  - ../../lib/zcu_tools/gui/widgets/form/README.md
  - ../../lib/zcu_tools/gui/widgets/form/editor.py
basis:
  repo_commit: edf07840c014d7b6736af4a2ca79f30a92ece3e2
  delivery_tree: 9ceabc70a93670bbe809525b13db172d87cc1674
  external_sources: none
---

# GUI teardown 的可見視窗 guard 與 owner 清理

## 結論

2026-11-17，使用者選擇一次處理同類 fixture 清理，並加入 teardown guard。測試結束後仍可見的 top-level 視窗讓該測試失敗。Guard 在所有 fixture finalizer 結束後立即取樣，不先交付 `DeferredDelete`、GC 或 drain，也不代關視窗。

Fixture 只退休自己建立或明確登記的視窗。產品退休的 form row 在解除 parent 後明確隱藏，有焦點時仍留在同一個 signal blocker 內。當時的規則與實作分別由 [tests README](../../tests/README.md#gui-可見視窗的-teardown-guard)、[form README](../../lib/zcu_tools/gui/widgets/form/README.md) 與各 owner 持有，本篇保存選項與判斷。

## 問題

驗收批次發生 segfault 後，最小同序案例指出 load dialog 的測試 fixture 只呼叫 `deleteLater()`，沒有先關閉仍可見且持有焦點的 dialog。修正 `d9595779bb9a0d96d450314d63ce385883c95939` 在 `tests/gui/app/fluxdep/ui/test_load_dialog.py` 的 fixture 中先 `close()` 再 `deleteLater()`。原正式批沒有 worker 執行順序，因此當時沒有據此認定不同崩潰位置同因。

接著的 AST 掃描在當時 integration 找到115個 teardown 含 `deleteLater()` 的 fixture。其中63個沒有先 `close()`，31個在 fixture 內顯示了視窗。這是啟發式清單，沒有把每筆都當成已證明的洩漏。

加入 guard 後，focused branch 測試暴露另一個問題。已退休的 `form.row.branch.one` 在事件處理後成為可見的獨立視窗。這次需要區分測試清理不足與產品退休行為，而不是讓 guard 先清理它看見的內容。

## 考慮過的選項與比較

同類 fixture 清理有三個選項：

- A，一次審閱63個 fixture，在需要時於 `deleteLater()` 前補 `close()`，不需要者逐項寫理由。加入可見 top-level guard，清單外被 guard 抓到的測試一併修。
- B，只處理31個顯示視窗的 fixture 與 `preview_dialog`，不加 guard。
- C，不改，登記待辦後繼續正式批次。

使用者在2026-11-17選「A」。範圍原先只含測試與測試基礎設施；guard 若揭露產品留下可見視窗，就另行裁定。既有 [dialog 關閉修正](../../tests/gui/app/fluxdep/ui/test_load_dialog.py)作為要攔下的錯誤寫法，立場是遵循，不用新的清理步驟讓它變成綠燈。

對退休 row，當時另比較取樣前交付 `DeferredDelete` 與產品明確隱藏兩條路徑。Offscreen probe 得到五項事實：

1. 已顯示的 `QDialog` 只 `deleteLater()`、不 `close()` 時，在交付 `DeferredDelete` 前仍可見，交付後消失。先交付再取樣會漏掉原本要抓的問題。
2. `setParent(None)` 本身會隱藏 widget。問題出在 layout 加入 child 時排定的延後 show：同一輪事件內先加入再解除 parent，之後 `processEvents()` 仍會把 row 顯示成 top-level。解除 parent 後明確 `hide()` 可阻止這次顯示。
3. 在相同 signal-blocked 範圍內先解除 parent 再 `hide()` 後，當時 form owner 的234個測試通過，teardown 都沒有可見 top-level。
4. 若在 blocker 外、解除 parent 前就 `hide()`，focus-out 會提交退休文字。當時的 focused-branch 案例多出 `{('branch', 'one'): 7.0}`，原斷言失敗。
5. 無 parent 的一般 `QWidget` 自動帶有 `Qt.Window`，值為0x1；`QDialog` 為0x3。Window flag 不能用來排除退休 row。

產品修正遵循 form README 原有的退休後不再提交文字語意，補上解除 parent 後的明確隱藏。Focused 分支沿用既有 `QSignalBlocker`，不把隱藏移到 blocker 外。這是當時唯一核可的產品修改。

## 被否決的選項與理由

- 取樣前交付 `DeferredDelete`。它會讓只刪不關的舊 dialog 寫法通過，與 guard 要保護的失敗條件相反。
- 只檢查 `QDialog`、`QMainWindow` 或 Window flag。一般 widget 也能成為可見 top-level，這些篩法會漏報。
- 在 blocker 外先 `hide()`。這會因 focus-out 提交已退休文字。
- Fixture 關閉所有測試期間新增的 top-level。新增不代表該 fixture 擁有，代關會掩蓋其他 owner 的洩漏。
- 關掉 guard、改 marker 或 skip／deselect。這些操作沒有修正視窗生命週期，也不保留原驗收。

Fixture ownership 問題在 land review 再次被指出。`main_window_control_factory` 原本以「新增 top-level 減 baseline」挑選清理對象，非 MainWindow 也會被關閉，然後才輪到 guard 取樣。

修正 `4aa292012c6cbf643bc634e27fdb3b67f0856fb2` 改成明確登記自有 roots，只退休登記的 trees。新增的 [pytester 回歸](../../tests/test_gui_window_guard.py)顯示未登記的 `QDialog` 與 row。舊 fixture 會代關兩者，內層沒有 teardown error，外層預期兩個 error 而失敗；新寫法讓原 guard 報出兩者的類別與 nodeid。回歸在觀察 report 後才清理自己故意留下的視窗。

## 留下的問題

當時另有 MainWindow 殘留，沒有把 form row 的成因直接套用到它。後續確認是測試 fixture 的 owner 清理，修正 `9a697686286a5d69c16a738e2ebdc7d2926b80e4` 在 controller 關閉後對自有視窗呼叫公開 `dispose()`，只改測試。

本次沒有再提出其他產品留下可見視窗的設計題。若之後 guard 抓到新的產品殘留，仍需由該 owner 判斷與修正，不能由無關 fixture 代關。

## 依據

- 本篇以 main `edf07840c014d7b6736af4a2ca79f30a92ece3e2` 與交付 tree `9ceabc70a93670bbe809525b13db172d87cc1674` 為基礎。
- 五項 probe 事實來自2026-11-17的 main `fa0980cd3`；當時相關產品程式與 guard WIP 相同。原始 probe 腳本與輸出未入庫，必要事實與數字已寫在本文。
- 可核對的程式與回歸為 [GUI guard](../../tests/gui/conftest.py)、[guard meta-tests](../../tests/test_gui_window_guard.py)、[明確 owner factory](../../tests/gui/app/measure/ui/_main_window_fakes.py)及 [FormEditor](../../lib/zcu_tools/gui/widgets/form/editor.py)。
