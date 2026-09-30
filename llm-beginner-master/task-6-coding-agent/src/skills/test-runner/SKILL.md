---
name: test-runner
description: 当你需要运行测试、定位失败原因并修复时加载。先跑 pytest，读失败断言，定位到具体实现，再最小改动修好后重跑直至全绿。
---

# Test Runner

- 先 `run_tests` 看失败输出。
- 从失败断言定位到具体函数/行（用 `read_file`）。
- 做**最小**修改修复（不动测试文件）。
- 再 `run_tests` 确认通过；若仍失败，回到上一步。
- 反复两次以上仍失败时，用 `git_diff` 复核改动。
