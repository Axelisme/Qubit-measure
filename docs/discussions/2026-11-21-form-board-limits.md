---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0067-gui-application.md
  - ../adr/0073-measure-form-derivation.md
  - 2026-10-17-measure-form-initial-values-and-sweep-mode.md
  - 2026-11-21-experiment-run-wall-time.md
basis:
  repo_commit: 32c7db0b57a71d503d2eca6ef29430416e9654cb
  external_sources: none
---

# 表單預設超過板子限制，錯誤解析放知識庫

## 結論

使用者決定不加 Run 檢查，也不在實驗指南說明板子限制。指南寫給懂物理但沒看過實作的讀者，只說明實驗和它的行為。四種 Run 錯誤的判讀、改法與 FTF 案例放在獨立的量測知識庫。Length Rabi 繼續沿用 `x180` 的形狀，掃描上限仍是 `x180` 的四倍長。`reset_check` 另把 gain 掃描上限寫成 `min(1.0, 4.0 * x180.gain)`，和 `singleshot/amp_rabi` 一致。CPMG 的 Minimum time 改用校準脈衝長度計算。兩項只修表單預設，不加驗證。

## 當時的離線失敗

第一張離線 Run 牆鐘表中，本次處理的七個實驗都以預設值失敗。`singleshot/len_rabi` 跑了 23 點，第 24 點才因 generator 的 8192 samples 上限停止。錯誤沒有寫該改哪個欄位，也沒有把 samples 換算成表單使用的 µs。`reset_check` 則不需查板子就能定位：測試 point 的 `x180.gain` 為 0.617，四倍是 2.468，超過 gain 的固定上限 1.0。CPMG 的 1 µs 預設在 16 次 refocusing 時，留給每段 interval 的時間只有約 31 ns，不夠放下測試 point 的脈衝。

## 考慮過的選項與比較

- 使用者先選在 Run 開頭拒絕超限值。這能先指出哪個欄位超限，符合 [ADR-0067](../adr/0067-gui-application.md) 對額外實驗驗證的放置規則，但需要新增能力查詢並修改各實驗。
- 使用者後改為不改 Run，先補指南，再把板子數字和錯誤解析移到知識庫。最後連中性的限制說明也留在知識庫，讓指南只描述實驗行為。操作員遇到錯誤時可按症狀查判讀方法。現有 Run 仍可能做到一半才報錯。
- 把 `reset_check` 的四倍 gain 在 expression 截到 1.0，沿用同類 Amplitude Rabi 寫法。這和 [2026-10-17 的表單討論](2026-10-17-measure-form-initial-values-and-sweep-mode.md) 中的負 `trig_offset` 不同：這次上限是固定的，不會依板子或校準輸入浮動。
- CPMG 同樣修預設。Minimum time 取 π/2 與 π 脈衝長度總和的 16 倍，配合預設的最大 refocusing 次數。任一脈衝未校準時，Minimum time 留空必填。使用者改次數時，Minimum time 不跟著重算。

## 被否決的選項與理由

- 不加 Run 前的 preflight。現有表單沒有板子設定，為這幾個案例增加前置檢查的維護成本較高。
- 不依板子容量縮短 Length Rabi 的預設掃描，也不把 `x180` 一律改成 flat top 或 constant。前者讓預設值隨板子變動，後者會改變擬合和回寫的 π pulse 長度所代表的意義。
- 不把板子限制寫進指南，就算不帶數字也是實作細節。`reset_check` 指南仍寫出 gain 上限的預設值，因為這是操作員開啟表單時會看到的行為。
- `reset_check` 不只靠報錯。四倍公式在這個 point 算出的預設值本來就不能執行，使用者要求在 expression 裡截斷。
- CPMG 不保留不能執行的 1 µs 預設，也不把 interval 檢查挪進 cfg。提早報錯仍要操作員手改，無法修正預設。手動填得太短時，仍由既有的 Run 檢查拒絕。

## 留下的問題

`reset_check` 的手填 gain 端點檢查另題處理。CPMG 的脈衝如果長到讓 Minimum time 超過預設的 Maximum time，操作員仍要調整時間窗。

## 依據

使用者在 2026-11-21 先答：「1a, 由於run內是黑盒，可以抱錯，而cfg只能自然做到簡單中性的檢查，這是可以接受的取捨，bc的維護複雜度略高，且目前沒有強烈需求 2a，簡單比複雜好，自動推導的正確性點到為止即可」。同日改判 Q1：「上一題的一我改變想法，我想先補guide就好」。對 `reset_check` 的公式另答：「關於4gain超過1.0使於是預設值設計不良，應該套一層min來在expression中截斷」。接著要求錯誤解析進知識庫：「然後guide請不要包含取決於板子資訊的內容，其應該是中性的說明，要提供錯誤解析，應該放知識沉澱的疑難排解」。最後定下指南的讀者與範圍：「Guide的定位是給只有物理知識，完全沒看過實作的使用者角度看的，寫太多實作細節沒有意義，所以板子限制不要放guide，guide是介紹這個實驗，與其行為描述的說明，底層qick抱錯屬於疑難雜症，進知識沉澱的judge」。

使用者同日選擇修 CPMG 的預設值：「1c，没错，这个是 这个是预设值的瑕疵，这种类似的东西都是改预设值就好了。」用各 0.48 µs 的 π/2 與 π 脈衝離線核對，表單給出的 15.36 µs 通過既有的 interval 檢查，手填 15.35 µs 則不通過。

原始牆鐘表未入庫。重跑入口和離線輸入見 [實驗 Run 牆鐘](../../tools/README.md#實驗-run-牆鐘)。板子數字用[入庫的 `qick_216_v2` profile](../../zcu_lab/simulation_envs/profiles/qick_216_v2.yaml)核對。

## 後記

[reset_check Form](../../zcu_lab/v2/singleshot/reset_check/form.py) 宣告 Gain Stop 初值，[CPMG Form](../../zcu_lab/v2/twotone/time_domain/cpmg/form.py) 宣告 Minimum time 初值。

設計者在審查時選擇保留 CPMG 的初值公式。即使兩個脈衝都已校準，只要任一脈衝用取樣點定義的 arbitrary waveform，表單仍將 Minimum time 留空，由操作員填入。缺少任一脈衝時也留空。
