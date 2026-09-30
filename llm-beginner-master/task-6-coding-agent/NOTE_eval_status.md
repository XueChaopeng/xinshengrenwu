# 任务六：自检状态说明（本机显存限制）

## 结论
`toy_repo_patch`（M3）**未通过**，原因不是实现问题，而是**本机显存不足**：
本机为 NVIDIA GeForce RTX 3050 Laptop GPU，**仅 4GB 显存**，无法运行任务要求的本地
Qwen2.5-Coder-7B-Instruct（FP16 需 ~14GB；Q4_K_M 量化也需 ~4.5GB+ 推理开销，超出 4GB）。

## 已完成
- **M1（MCP server）通过**：`mcp_server_lists_tools` ✅，`list_tools()` 枚举到 5 个工具
  （read_file / write_file / run_tests / git_diff / git_apply），含路径越界保护与 subprocess list 防注入。
- **M2（Skill 加载器 + Skills）通过**：`skill_loader_metadata` ✅，`SkillLoader.list_skills()` 返回 3 个
  带 name+description 的 Skill（code-review / pr-description-writer / test-runner），渐进式披露。
- `src/mcp_server.py`、`src/skill_loader.py`、`src/skills/*/SKILL.md`、`src/agent.py`（CodingAgent
  agent loop + SubAgent）、toy-repo（data/toy-repo）全部就位。

## 受限项 / 实测情况
- `toy_repo_patch` ❌：用能塞进 4GB 的 **Qwen2.5-1.5B-Instruct** 实测，模型**未发出 write_file
  修复动作**（只返回文字），`calculator.add` 仍为 `a - b`。该修复需要可靠的编码/工具调用能力，预期需
  Qwen2.5-Coder-7B 级别。
- `swebench_lite_sample`（S4）：未跑（需 `data/download.py --with-swebench` 下载元数据 + clone 对应 repo）。

## 如何复现 / 复测
在有 ≥8GB（推荐 16GB）显存的机器上：
1. 部署本地 OpenAI 兼容端点，加载 `Qwen2.5-Coder-7B-Instruct`。
2. 设 `OPENAI_BASE_URL=http://<host>:<port>/v1`、`OPENAI_API_KEY`（任意）、`CODING_MODEL=qwen2.5-coder:7b-instruct`。
3. `python eval/run.py` 即可得到真实 `toy_repo_patch` 结果（每次自检会从 `calculator.py.orig` 恢复 buggy 版本）。
