"""wiki：维基百科查询（中英文均可），返回条目摘要文本。"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "查询维基百科获取某概念/人物的信息，返回条目标题与摘要。支持中英文。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要查询的词条"}},
            "required": ["query"],
        },
    },
}

_TIMEOUT = 15


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-beginner-agent/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _api(query: str, lang: str, prop: str) -> dict:
    import json
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "utf8": "1",
        "list": "search", "srsearch": query, "srlimit": "3",
    }
    if prop == "extract":
        params = {"action": "query", "format": "json", "utf8": "1", "prop": "extracts",
                  "explaintext": "1", "exintro": "1", "titles": query}
    return json.loads(_fetch(base + "?" + urllib.parse.urlencode(params)))


def _first_title(search_result: dict) -> str:
    try:
        return search_result["query"]["search"][0]["title"]
    except Exception:
        return ""


def run(args: dict) -> str:
    query = str(args["query"]).strip()
    if not query:
        return "Error: query 为空"
    lang = "zh" if _is_chinese(query) else "en"

    try:
        search = _api(query, lang, "search")
        title = _first_title(search)
        info = []
        if title:
            info.append(f"标题: {title}")
        if title:
            ext = _api(title, lang, "extract")
            pages = ext["query"]["pages"]
            for _, page in pages.items():
                text = page.get("extract", "")
                info.append(text[:1200])
        result = "\n".join(x for x in info if x)
        return result if result else f"未找到相关条目：{query}"
    except Exception as e:
        # 尝试英文源，再失败则抛（网络不可用时由 self-check 按跳过处理）
        try:
            search = _api(query, "en", "search")
            title = _first_title(search)
            if title:
                ext = _api(title, "en", "extract")
                pages = ext["query"]["pages"]
                for _, page in pages.items():
                    return f"标题: {title}\n{page.get('extract','')[:1200]}"
        except Exception as e2:
            raise RuntimeError(f"wiki 查询失败：{e2}") from e
        return f"未找到相关条目：{query}"
