# Candidate Backlog Lifecycle

```text
inbox --plan--> planned --close implemented--> resolved
inbox/planned --close declined|duplicate|obsolete|not-reproducible|out-of-scope--> closed
```

- `plan` 必須綁定正式 `.agent_state/plans/<task-id>/` 的 task-id；正式 task 反向引用 backlog ID。
- 實作完成但尚未整合仍是 `planned`。
- `implemented` 只接受 `planned`，且 task-id 必須與 plan 綁定值相同；必須記錄至少一個 commit 與
  validation evidence。完成表示缺口已消失、驗證完成且已整合至正式目標 branch。
- `duplicate` 必須指向另一個存在的 canonical ID，且不可指向自己。
- 其它 close reason 可從 inbox 或 planned 進入 closed，並保留原因與時間。
- 不直接刪除 resolved/closed 記錄；真正需要團隊長期共享的事項提升至 tracked 文件或 issue tracker。
