# Parallel Burst

`parallel-burst` 是 stateless execution capability，不是 risk profile、persistence mode或 completion authority。
它只縮短真正獨立工作形成的 critical path；最大 active-agent數不是目標。

## Activation gate

至少兩個節點同時滿足：write scope、public contract/schema/fixture、設計決策與 targeted acceptance互相獨立；
預期 `sequential estimate - longest wave - startup cost - integration cost` 明顯為正。否則使用 direct/delegated。

## Wave plan

Non-trivial parallel work使用短 artifact：

```markdown
## Wave 0 — Foundation
- item: scope, acceptance, unblocks
## Wave 1 — Parallel
- item-a
- item-b
## Wave 2 — Integration
- boundary review
- integration validation
```

並附 dependency matrix：`Item | Depends on | Write scope | Contract owner | Validation`。Foundation驗收後才啟動下游；
同 wave workers一次啟動，不逐一等待。新 evidence改變 contract或scope時停止該 wave並重新分圖。

## Runtime behavior

- 預設兩個 implementer，保留一個 reviewer/investigator slot。
- Quick reads/status留 foreground；長 test/build/log processing可background，但background operation不取得新的
  write scope，會寫檔者仍由owner lane執行。
- 以 completion/blocked mailbox event驅動；沒有 event才wait。單次timeout只做狀態probe，不判定worker死亡。
- Worker完成即釋放；不讓 idle worker長期占slot，也不允許worker再spawn sub-agent。
- 報告預設30行內；raw output寫report artifact。無 finding review建議10行內。

## Loop authority

Wave artifact必記 `existing_authority_id` 與 conflict decision。已有 continuation authority時：

- `refuse`：不啟動workers，也不建立新的continuation owner；
- `adopt_existing`：只向既有authority提交waves與evidence，completion authority仍屬原owner；
- `artifact_only`：只產生plan/handoff artifact，不執行workers。

只有既有authority或使用者可決定 `adopt_existing`；新的orchestrator不能自行奪取或轉移authority。沒有既有
authority時，由啟動task的orchestrator成為唯一owner，`parallel-burst`本身仍不取得authority。

## Handoff and boundaries

只在 design→implementation、foundation→consumers、implementation→integration、blocked→resumed寫10–20行
handoff，包含 `Decided / Rejected / Risks / Files / Remaining`。一般lane completion只用短report。

`parallel-burst` 完成代表節點執行結束，不代表 task完成；結果交回唯一 loop authority，由 orchestrator進行
integration、bounded verification、commit/merge authority與cleanup。
