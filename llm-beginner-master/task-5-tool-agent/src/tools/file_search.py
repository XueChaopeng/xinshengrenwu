"""file_search：本地目录文件名 / 内容检索，带路径越界保护，返回匹配的文件路径 + 内容片段。"""
from __future__ import annotations

from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在给定目录下递归检索匹配 pattern（文件名或文件内容包含）的文件，返回文件路径与内容片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "要检索的子串（匹配文件名或文件内容）"},
                "dir": {"type": "string", "description": "要检索的目录绝对路径"},
            },
            "required": ["pattern", "dir"],
        },
    },
}

ALLOWED_ROOT = Path(__file__).resolve().parents[2]  # task-5-tool-agent

# 跳过二进制/大文件与常见非源码目录，避免枚举 models/、.git 等巨大目录
_IGNORE_DIRS = {".git", "__pycache__", "node_modules", "models", "data/cache", ".cache"}
_MAX_READ_BYTES = 2_000_000  # 只读小于 2MB 的文本文件


def _is_text_file(p: Path) -> bool:
    if p.stat().st_size > _MAX_READ_BYTES:
        return False
    try:
        with open(p, "rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return False  # 二进制
    except Exception:
        return False
    suffix = p.suffix.lower()
    return suffix in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml",
                      ".cfg", ".ini", ".c", ".cpp", ".h", ".java", ".js", ".ts",
                      ".html", ".css", ".csv", ".log", ".sh", ".ipynb", ""} or suffix not in {
        ".pt", ".bin", ".safetensors", ".onnx", ".gguf", ".zip", ".parquet", ".pdf"}


def _resolve_within(dir_path: Path) -> Path:
    base = dir_path.resolve()
    allowed = ALLOWED_ROOT.resolve()
    if not (base == allowed or allowed in base.parents):
        raise ValueError(f"路径越界：{base} 不在允许根目录 {allowed} 内")
    return base


def run(args: dict) -> str:
    pattern = str(args["pattern"]).lower()
    base = _resolve_within(Path(str(args["dir"])))
    if not base.exists():
        return f"Error: 目录不存在：{base}"

    matches = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part in _IGNORE_DIRS for part in p.parts):
            continue
        if not _is_text_file(p):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern in p.name.lower() or pattern in content.lower():
            snippet = content[:200].replace("\n", " ")
            matches.append(f"路径: {p}\n内容: {snippet}")
    return "\n\n".join(matches) if matches else "未找到匹配文件"
