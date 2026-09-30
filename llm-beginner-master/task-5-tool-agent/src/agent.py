"""手写 ReAct Agent：Thought / Action / Action Input / Observation 循环 + Final Answer 终止。

- 调用本地 OpenAI 兼容 endpoint（默认 Ollama http://localhost:11434/v1，可用环境变量覆盖）。
- 工具路由：解析模型输出的 Action + Action Input，调用对应工具。
- 错误恢复：工具抛异常时把错误消息塞回 Observation，让 agent 自我纠错，不 crash 整个循环。
- 终止：模型给出 Final Answer，或步数达到上限。
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

from .tools import calculator, python_sandbox, file_search, wiki

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
MODEL_NAME = os.environ.get("AGENT_MODEL", "qwen2.5:7b-instruct")
MAX_STEPS = 8

_TOOLS = [calculator, python_sandbox, file_search, wiki]
_TOOL_REGISTRY = {t.TOOL_SCHEMA["function"]["name"]: t for t in _TOOLS}

_FEW_SHOT = """\
你是能调用工具的智能体。只能在下述工具里选一个调用。逐步：先 Thought，需要工具就给 Action + Action Input（参数必须是合法 JSON），拿到 Observation 后若已能得到答案就立即用 Final Answer 收尾，不要继续调用工具。

可用工具：
{TOOL_DESC}

工具使用对照：
- 数值/四则/开方计算 -> calculator，参数 {{"expression": "..."}}
- 运行 Python 代码 -> python_sandbox，参数 {{"code": "..."}}
- 检索本地文件/目录 -> file_search，参数 {{"pattern": "...", "dir": "..."}}
- 查维基百科 -> wiki，参数 {{"query": "..."}}

示例：
Thought: 需要计算 3*4+1。
Action: calculator
Action Input: {{"expression": "3 * 4 + 1"}}
Observation: 13
Final Answer: 结果是 13。
"""


class ReActAgent:
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None,
                 max_steps: int = MAX_STEPS):
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or MODEL_NAME
        self.max_steps = max_steps
        self.tool_desc = self._build_tool_desc()

    def _build_tool_desc(self) -> str:
        lines = []
        for t in _TOOLS:
            f = t.TOOL_SCHEMA["function"]
            params = f["parameters"]["properties"]
            args = ", ".join(f"{k}: {v.get('type')}" for k, v in params.items())
            lines.append(f"- {f['name']}: {f['description']} 参数: {args}")
        return "\n".join(lines)

    # ---------------- LLM 调用 ----------------
    def _llm(self, messages: List[dict]) -> str:
        import requests
        resp = requests.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 128},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ---------------- 解析 ----------------
    @staticmethod
    def _parse_action(text: str):
        """从模型输出解析 (action_name, action_input) 或 None（找不到 Action）。"""
        lines = text.splitlines()
        action, action_input = None, None
        for i, line in enumerate(lines):
            if line.startswith("Action:"):
                action = line.split("Action:", 1)[1].strip()
            if line.startswith("Action Input:"):
                raw = line.split("Action Input:", 1)[1].strip()
                action_input = raw
        if action and action_input:
            # 兼容偶尔给成 json 块 / 多行
            action_input = action_input.strip().strip("`").strip()
            if action_input.startswith("{") or action_input.startswith("["):
                pass
            else:
                # 尝试取花括号内容
                m = re.search(r"\{.*\}", action_input, re.S)
                if m:
                    action_input = m.group(0)
            try:
                return action, json.loads(action_input)
            except Exception:
                return action, {"_raw": action_input}
        return None

    @staticmethod
    def _parse_final(text: str) -> str:
        m = re.search(r"Final Answer:\s*(.+)", text, re.S)
        return m.group(1).strip() if m else None

    # ---------------- 主循环 ----------------
    def run(self, task: str) -> Dict:
        messages = [{"role": "system", "content": _FEW_SHOT.format(TOOL_DESC=self.tool_desc)}]
        messages.append({"role": "user", "content": task})
        steps: List[Dict] = []
        final_answer = ""
        success = False

        for step in range(self.max_steps):
            raw = self._llm(messages)
            final = self._parse_final(raw)
            action = self._parse_action(raw)
            steps.append({"step": step + 1, "raw": raw})

            if final:
                final_answer = final
                success = True
                break

            if not action:
                # 解析失败：把提示塞回，让模型重试
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "请重新输出：必须以 Final Answer 结束，或用 Action / Action Input 调用工具。"})
                continue

            name, args = action
            messages.append({"role": "assistant", "content": raw})
            step_rec = {"step": step + 1, "tool": name, "args": args}
            try:
                if name not in _TOOL_REGISTRY:
                    raise KeyError(f"未知工具：{name}")
                observation = str(_TOOL_REGISTRY[name].run(args))
            except Exception as e:
                # M3：错误恢复 —— 把错误消息塞回 Observation，让 agent 自我纠错
                observation = f"Error: {type(e).__name__}: {e}"
            step_rec["observation"] = observation
            steps[-1].update({"tool": name, "args": args, "observation": observation})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        if not final_answer:
            final_answer = "(agent 未给出最终答案)"
        return {"steps": steps, "final_answer": final_answer, "success": success}
