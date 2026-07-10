**Last updated:** 2026-07-11 — per-experiment tests

# autofluxdep experiment tests

本目錄集中 concrete measurement experiments 的 cfg/acquire/fit/Result/Patch 測試。`test_catalog.py` 驗證 catalog Fast Fail、每個 authoritative experiment 恰好註冊一次、package import 方向、predictor exclusion 與 workflow order boundary。

跨 experiment 的 orchestrator、persistence、cfg schema 與 artifact tests 留在上一層；共用 fixture 從 `tests.autofluxdep_gui._helpers` 引用，不在這裡建立第二份 production Builder registry。
