"""calculator：受限的算术 / 数学函数计算（不用裸 eval 执行任意代码）。"""
from __future__ import annotations

import ast
import math
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行算术或数学函数计算，返回结果字符串。支持四则运算、幂、开方、三角函数等。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(2026)'"}},
            "required": ["expression"],
        },
    },
}

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_NAME_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的类型：{node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_safe_eval(node.operand)
    if isinstance(node, ast.Name) and node.id in _NAME_FUNCS:
        return _NAME_FUNCS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAME_FUNCS:
        args = [_safe_eval(a) for a in node.args]
        return _NAME_FUNCS[node.func.id](*args)
    raise ValueError(f"表达式包含不支持的运算/标识符：{ast.dump(node)}")


def run(args: dict) -> str:
    expression = str(args["expression"]).strip()
    if not expression:
        raise ValueError("expression 不能为空")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    # 数值尽量精确显示（丢弃浮点尾巴的 -0/过长）
    if isinstance(result, float):
        s = f"{result:.10f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(result)
