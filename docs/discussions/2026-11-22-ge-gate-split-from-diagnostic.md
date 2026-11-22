---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-families-partial-delivery.md
  - 2026-11-22-data-figure-reading.md
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0072-module-boundaries.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: 5b2133fbba212e9409651b4aae06c820c5ff9b3b
  external_sources: none
---

# GE 寫回保護先於診斷圖交付

## 結論

2026-11-22，designer 選擇先獨立交付 GE 寫回保護，診斷圖整批留到續作。這是 designer 代裁，使用者未覆核交付順序。使用者原先要求的直方圖、同次擬合 SNR 與等比例 IQ 沒有取消。恢復版診斷圖未通過版面驗收，不能隨寫回保護發布。

## 問題

低訊雜比的 GE 擬合可能通過 r²、殘差與寬度檢查，但 g／e 中心仍估得不準。兩個 adapter 原先會預選這類寫回提議。寫回保護改查兩個中心相對誤差，不合格時保留提議，由操作員確認。這個判斷只需要既有擬合結果，不需要恢復診斷圖。

合併候選同時恢復了 2×2 診斷圖。左下格的縱軸標籤 `Fraction per bin` 和 `Prepared assignment` 標題相碰。14 pt 與唯一一次 12 pt 覆驗都在 360×375 px 的離線 mux 參考磚看到碰字；12 pt 的 High Q2、Low Q1、High Q1 三格仍相碰。12 pt 收據對應後來保存為 `b5161fa9b` 的工作樹。這只證實該參考尺寸的圖面失敗，不能歸因於 Qt，也不能當作實際桌面驗收。直方圖依準備態各自歸一，改回 `Counts` 會把每 bin 的比例說成計數。

## 選項與取捨

分開交付要從當時最新的 main 重建 [mux GE](../../zcu_lab/v2/mux/singleshot_ge/guide.md) 與 [singleshot GE](../../zcu_lab/v2/singleshot/ge/guide.md) 的 adapter 和 guide，保留其間新增的摘要及行為。原四檔提交 `06ca3a034` 已顯示寫回檢查可以獨立。新候選仍需重做公開 note 的合成對照、唯讀 Load／Analyze／Post 的提議檢查，以及全樹檢查和審查。低增益來源原先已因寬度或 Post 條件不預選，所以實際來源用來確認新中心理由及高增益仍預選；中心檢查能否單獨阻止預選由合成對照驗。

維持同批可以省下分流重建，卻得先找到一種不遮字、也不丟失原圖資訊的版型，再驗正常與窄尺寸、Data 閱讀面和 Save 匯出。當時沒有通過的替代版型。整批暫停不用重建，但會延後能獨立驗收的寫回保護。沒有找到使用者要求兩項同批發布的依據。

## 未採用

沒有再縮字級，也沒有退回 `Counts` 來發布。前者已用完限定的覆驗，後者改錯了量義。也沒有從讀圖目的出發重排圖面、拆出 population，因為那會改動使用者偏好的原圖畫法，須另定版型與驗收。

## 留下的問題

診斷圖保留原候選、修正與失敗收據，待確認可取回後才退休舊工作樹。續作仍要把直方圖、同次擬合 SNR 及其誤差與警示、等比例 IQ 和圖名碰撞修正一起驗收。依 [R7](../../zcu_lab/v2/figure-style.md#r7用-gui-與存檔的實際尺寸檢查版面) 檢查實際 Analysis、Data 的 `View figure`、Save 版面及存檔 PNG；之前的離線例外不延用。

寫回保護的 `review-done` 只表示這一項驗完，不能當作已合併。寫回保護與 presentation-families 都實際合併後，noise-isotropic 才按原流程驗證和合併。FTF preset 仍等 noise-isotropic；中心距除以共用寬度的定義不變。這次拆分不簽收診斷圖或第 11／12 步。

## 依據

使用者原先要求：「我需要看那个 histogram，没有 histogram，只看散点图根本看不出什么」。分流沒有改這項要求。[ADR-0072 的實驗分工](../adr/0072-module-boundaries.md#實驗計算與-gui-接線)讓 adapter 決定寫回，分析與繪圖不消費寫回 note。[Families 部分交付](2026-11-22-families-partial-delivery.md)同樣把未驗範圍列為缺口；GE 不同之處是重建一個不含圖的候選，沒有直接採用原合併候選。[Data 閱讀面](2026-11-22-data-figure-reading.md)的判讀契約也不能代替 Analysis 或 Save 的版面驗收。原始離線圖、調查記錄與批次收據未入庫。
