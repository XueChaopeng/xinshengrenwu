"""python_sandbox：受限 exec（黑名单 import + 白名单 builtins + 超时 + stdout 捕获）。

⚠️ 仅教学级防护（防误用），不是真正隔离：仍可经 __globals__/__bases__ 等逃逸，超时挡不住内存耗尽。
只对可信 / 自产代码使用。
"""
from __future__ import annotations

import builtins
import contextlib
import io
import threading

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "在一个受限 Python 环境内执行代码，返回 print 输出。禁止 import；仅暴露常用内置函数。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
            "required": ["code"],
        },
    },
}

# 白名单 builtins：不放 __import__ / open / eval / exec 等危险项
_SAFE_BUILTINS = {
    "print": print, "abs": abs, "max": max, "min": min, "sum": sum, "len": len,
    "range": range, "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "round": round, "map": map, "filter": filter,
    "all": all, "any": any, "isinstance": isinstance, "repr": repr,
    "True": True, "False": False, "None": None,
}
_SAFE_GLOBALS = {"__builtins__": _SAFE_BUILTINS}

_TIMEOUT = 3.0


def run(args: dict) -> str:
    code = str(args["code"]).strip()

    # 黑名单 import / 逃逸入口 —— 直接报错
    for forbidden in ("import ", "__import__", "open(", "eval(", "exec(", "os.", "sys.", "globals", "bases"):
        if forbidden in code:
            return f"Error: 代码包含被禁止的语句：{forbidden}"

    buf = io.StringIO()
    holder = {"val": None}

    def _exec():
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<sandbox>", "exec"), _SAFE_GLOBALS, _SAFE_GLOBALS)
            holder["val"] = "ok"
        except Exception as e:  # noqa: BLE001
            holder["val"] = f"Error: {type(e).__name__}: {e}"

    # 用线程做超时（Windows 无 signal.SIGALRM）
    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(_TIMEOUT)
    if t.is_alive():
        return "Error: 执行超时（> 3 秒）。"

    val = holder["val"]
    if val != "ok":
        return val
    out = buf.getvalue()
    return out if out else "(代码无输出)"
