# 任务五：自检状态说明（本机显存限制）

## 结论
`multi_tool_success_rate`（M4）**未通过**，原因不是实现问题，而是**本机显存不足**：
本机为 NVIDIA GeForce RTX 3050 Laptop GPU，**仅 4GB 显存**，无法运行任务要求的本地
Qwen2.5-7B-Instruct（FP16 需 ~14GB；Q4_K_M 量化也需 ~4.5GB+ 推理开销，超出 4GB）。

## 已完成
- **M1（4 个工具）通过**：`tools_individual` ✅（calculator / python_sandbox / file_search 真通过；
  wiki 因 `en/zh.wikipedia.org` 网络被墙按「可跳过」处理）。
- **M2（ReAct 循环）、M3（错误恢复）**：代码已实现（`src/agent.py`，Thought/Action/Action Input/
  Observation + Final Answer 停机 + 工具异常塞回 Observation 自纠错）。
- `src/tools/{calculator,python_sandbox,file_search,wiki}.py` + `src/agent.py` 全部就位。

## 受限项 / 实测情况
- `multi_tool_success_rate` = **0.1（1/10）**，用能塞进 4GB 的 **Qwen2.5-1.5B-Instruct** 实测：
  模型能发出工具调用（如 calculator 算出 456831），但**严重幻觉工具观测值**（最终答 4701077/45.017989 等）
  且反复空转，无法稳定收敛。该基准预期需 7B+ 模型。

## 如何复现 / 复测
在有 ≥8GB（推荐 16GB）显存的机器上：
1. 部署本地 OpenAI 兼容端点（Ollama/vLLM/llama.cpp），加载 `Qwen2.5-7B-Instruct`。
2. 设 `OPENAI_BASE_URL=http://<host>:<port>/v1`、`OPENAI_API_KEY`（任意）、`AGENT_MODEL=qwen2.5:7b-instruct`。
3. `python eval/run.py` 即可得到真实成功率。
