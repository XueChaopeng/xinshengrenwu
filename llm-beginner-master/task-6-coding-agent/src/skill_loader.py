"""Skill 加载器：扫描 src/skills/*/SKILL.md，按 description 匹配，命中后再读正文（progressive disclosure）。"""
from __future__ import annotations

import re
from pathlib import Path


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)

    @staticmethod
    def _parse(path: Path):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
        if m:
            meta = _parse_yaml(m.group(1))
            body = m.group(2).strip()
        else:
            meta, body = {}, text.strip()
        return meta, body

    def _iter(self):
        for skill_dir in sorted(self.skills_dir.glob("*/")):
            md = skill_dir / "SKILL.md"
            if md.exists():
                yield skill_dir, *self._parse(md)

    def list_skills(self):
        result = []
        for skill_dir, meta, body in self._iter():
            result.append({
                "name": meta.get("name") or skill_dir.name,
                "description": meta.get("description") or body[:60],
                "dir": str(skill_dir),
                "body_preview": body[:120],
            })
        return result

    def load(self, name: str) -> str:
        for skill_dir, meta, body in self._iter():
            if (meta.get("name") or skill_dir.name) == name:
                # 渐进式披露：命中后才返回完整正文
                return f"# Skill: {meta.get('name')}\n\n{body}"
        raise KeyError(f"未找到 Skill：{name}")


def _parse_yaml(text: str) -> dict:
    """极简 YAML front-matter 解析（key: value），不依赖外部 yaml，避免依赖问题。"""
    into = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        # 去掉首尾引号
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        into[key.strip()] = val
    return into
