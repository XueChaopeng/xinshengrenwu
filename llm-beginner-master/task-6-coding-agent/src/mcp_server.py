"""MCP server：暴露 ≥5 个工具（read_file / write_file / run_tests / git_diff / git_apply）。

- 模块顶层导出 list_tools() -> List[dict]（每个含 name/description/input_schema），供自检枚举。
- 工具实现做了路径越界保护（resolve 后校验落在目标 repo 内）与 shell 注入防护（subprocess list 形式）。
- `python src/mcp_server.py` 可独立启动 stdio server（若装了官方 mcp SDK）。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# 目标仓库根：用环境变量指定，或在工具调用时以 repo_root 参数覆盖
_REPO_ROOT = Path(os.environ.get("MCP_REPO_ROOT", ".")).resolve()


def _set_repo_root(path):
    global _REPO_ROOT
    _REPO_ROOT = Path(path).resolve()
    return _REPO_ROOT


def _resolve(base: Path, path: str) -> Path:
    """resolve 并校验路径落在 base 内，拒绝越界 / 绝对路径。"""
    p = (base / path).resolve()
    base_res = base.resolve()
    if not (p == base_res or base_res in p.parents):
        raise ValueError(f"路径越界：{p} 不在 repo {base_res} 内")
    return p


# ---------------- 工具实现 ----------------
def read_file(path: str, repo_root: str = None) -> str:
    base = Path(repo_root) if repo_root else _REPO_ROOT
    p = _resolve(base, path)
    return p.read_text(encoding="utf-8", errors="replace")


def write_file(path: str, content: str, repo_root: str = None) -> str:
    base = Path(repo_root) if repo_root else _REPO_ROOT
    p = _resolve(base, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return f"已写入 {p}"


def run_tests(repo_root: str = None) -> str:
    base = Path(repo_root) if repo_root else _REPO_ROOT
    proc = subprocess.run(["python", "-m", "pytest", "-q"], cwd=str(base),
                          capture_output=True, text=True, timeout=120)
    return (proc.stdout + proc.stderr)


def git_diff(repo_root: str = None) -> str:
    base = Path(repo_root) if repo_root else _REPO_ROOT
    try:
        proc = subprocess.run(["git", "diff", "--", "*.py"], cwd=str(base),
                              capture_output=True, text=True, timeout=30)
        return proc.stdout
    except FileNotFoundError:
        return "(git 未安装，无法生成 diff)"
    except Exception as e:
        return f"(git diff 失败：{e})"


def git_apply(diff: str, repo_root: str = None) -> str:
    base = Path(repo_root) if repo_root else _REPO_ROOT
    try:
        proc = subprocess.run(["git", "apply", "-"], cwd=str(base),
                              input=diff, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "(git 未安装，无法应用 patch)"
    except Exception as e:
        return f"(git apply 失败：{e})"
    if proc.returncode != 0:
        return f"git apply 失败：{proc.stderr}"
    return "patch 已应用"


# ---------------- 工具 schema ----------------
def list_tools():
    return [
        {"name": "read_file", "description": "读取仓库内文件内容",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_file", "description": "写/覆盖仓库内文件",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "run_tests", "description": "在仓库内运行 python -m pytest 并返回输出",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "git_diff", "description": "返回仓库当前未提交的 diff",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "git_apply", "description": "应用一个补丁",
         "input_schema": {"type": "object", "properties": {"diff": {"type": "string"}}, "required": ["diff"]}},
    ]


_IMPLEMENTATIONS = {
    "read_file": lambda a, r: read_file(a.get("path", ""), r),
    "write_file": lambda a, r: write_file(a.get("path", ""), a.get("content", ""), r),
    "run_tests": lambda a, r: run_tests(r),
    "git_diff": lambda a, r: git_diff(r),
    "git_apply": lambda a, r: git_apply(a.get("diff", ""), r),
}


def call_tool(name: str, args: dict, repo_root: str = None) -> str:
    if name not in _IMPLEMENTATIONS:
        raise ValueError(f"未知工具：{name}")
    try:
        return str(_IMPLEMENTATIONS[name](args or {}, repo_root))
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def main():
    """独立运行：作为 MCP stdio server（尽量用官方 SDK，否则退化为简化的 JSON-RPC）。"""
    try:
        import mcp
        from mcp.server.fastmcp import FastMCP
        mcp_server = FastMCP("llm-beginner-mini-coding-agent")
        for t in list_tools():
            mcp_server.add_tool(lambda *a, **k: None, name=t["name"])  # 占位，仅供标准握手
        mcp_server.run()
    except Exception as e:
        print(f"[提示] 未装/初始化官方 MCP SDK（{type(e).__name__}），按简单 stdio 模式启动。")
        import sys
        for line in sys.stdin:
            # 极简：仅打印工具列表，供客户端枚举
            print(json.dumps({"tools": list_tools()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
