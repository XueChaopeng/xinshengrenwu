"""Mini Coding Agent：手写 agentic loop，暴露 M1-M4 所需接口。

- CodingAgent.run(repo_path, issue) -> Trace(dict，含 steps / patch / tests_passed)。
- 通过本地 OpenAI 兼容端点调用模型，驱动 read_file / write_file / run_tests 等 MCP 工具。
- 支持 Skill（test-runner 等）与 Subagent（独立 message 列表 + 步数上限 + 工具子集）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .mcp_server import call_tool, git_diff

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
MODEL_NAME = os.environ.get("CODING_MODEL", "qwen2.5-coder:7b-instruct")
MAX_STEPS = 12


class SubAgent:
    """独立 context 的子 agent：只跑一个限定子任务，返回摘要，不污染主 context。"""

    def __init__(self, system, base_url=None, model=None, max_steps=3):
        self.system = system
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model or MODEL_NAME
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        out = []
        for _ in range(self.max_steps):
            raw = _llm(self.base_url, self.model, msgs)
            out.append(raw)
            if "DONE" in raw:
                break
            msgs.append({"role": "assistant", "content": raw})
        return "\n".join(out)[:600]  # 只返回摘要


class CodingAgent:
    def __init__(self, base_url=None, api_key=None, model=None, max_steps=MAX_STEPS):
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or MODEL_NAME
        self.max_steps = max_steps

    def _system_prompt(self, issue: str, repo_path: str, tool_desc: str, skill_body: str):
        return (
            "你是本地仓库的编码助手，只能调用给出的工具，逐步修复问题直到测试全绿。\n"
            f"任务（ISSUE）：{issue}\n"
            f"仓库路径：{repo_path}\n"
            "可用工具：\n" + tool_desc + "\n"
            "相关技能（test-runner）：\n" + skill_body + "\n"
            "流程：先 read_file 读相关文件，定位 bug，再用 write_file 做最小修复（不要改测试文件），"
            "最后 run_tests 确认通过。每步输出一行：\n"
            "  THOUGHT: <你的思考>\n"
            "  TOOL: <工具名>\n"
            "  PARAMS: <json>\n"
            "完成时输出：\n"
            "  DONE: <改了什么，测试是否通过>\n"
        )

    def run(self, repo_path: str, issue: str) -> dict:
        repo_path = str(Path(repo_path).resolve())
        from .skill_loader import SkillLoader
        loader = SkillLoader(str(Path(__file__).resolve().parents[1] / "src" / "skills"))
        try:
            skill_body = loader.load("test-runner")
        except KeyError:
            skill_body = ""
        tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in
                              __import__("src.mcp_server", fromlist=["list_tools"]).list_tools())

        messages = [{"role": "system", "content": self._system_prompt(issue, repo_path, tool_desc, skill_body)}]
        messages.append({"role": "user", "content": issue})
        steps = []

        for step in range(self.max_steps):
            raw = _llm(self.base_url, self.model, messages)
            step_rec = {"step": step + 1, "thought": _field(raw, "THOUGHT")}
            done = _field(raw, "DONE")
            tool = _field(raw, "TOOL")
            params = _field(raw, "PARAMS")

            messages.append({"role": "assistant", "content": raw})

            if done:
                step_rec["done"] = done
                steps.append(step_rec)
                break

            if tool and tool in {t["name"] for t in __import__("src.mcp_server", fromlist=["list_tools"]).list_tools()}:
                try:
                    args = json.loads(params) if params else {}
                except Exception:
                    args = {}
                observation = call_tool(tool, args, repo_path)
            else:
                # 没给出合法工具调用：提示重试
                observation = "请以 TOOL: <工具名> + PARAMS: {json} 的格式调用工具。"
                messages.append({"role": "user", "content": observation})
                step_rec["observation"] = observation
                steps.append(step_rec)
                continue

            step_rec["tool_call"] = {"tool": tool, "args": json.loads(params) if params else {}}
            step_rec["observation"] = observation
            steps.append(step_rec)
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            # 测试通过即可提前停机
            if tool == "run_tests" and "passed" in observation.lower() and "failed" not in observation.lower():
                break

        trace = {"steps": steps, "patch": git_diff(repo_path), "tests_passed": False}
        # 用真实 pytest 确认
        try:
            pr = subprocess.run(["python", "-m", "pytest", "-q"], cwd=repo_path,
                                capture_output=True, text=True, timeout=120)
            trace["tests_passed"] = pr.returncode == 0
            trace["pytest_output"] = (pr.stdout + pr.stderr)[-400:]
        except Exception as e:
            trace["pytest_output"] = str(e)
        return trace


def _field(text: str, key: str):
    m = re.search(rf"{key}:\s*(.+)", text)
    return m.group(1).strip() if m else None


def _llm(base_url, model, messages, max_new=128):
    import requests
    resp = requests.post(
        base_url + "/chat/completions",
        headers={"Authorization": "Bearer ollama", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": max_new},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
