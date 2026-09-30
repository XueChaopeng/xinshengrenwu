"""Byte-level BPE tokenizer (no tiktoken / sentencepiece).

实现思路跟随 GPT-2 的 byte-level BPE：
- 先把 UTF-8 文本拆成 256 个字节，每个字节经 ``bytes_to_unicode`` 映射成一个可打印的
  unicode 字符，作为初始 token。
- 统计相邻 token 对频率，每次取出现次数最多的对合并成一个新 token，得到 BPE merge 表。
- encode：文本 -> 字节 -> unicode 字符序列 -> 按 merge 表贪心合并 -> id 序列。
- decode：id 序列 -> token 字符串拼接 -> 逐字符反查字节 -> UTF-8 解码成原文。

这种方式天然能处理中文（跨多字节 UTF-8）与任意未登录字符，encode/decode 可逆。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self) -> None:
        self.byte_to_unicode: Dict[int, str] = self._bytes_to_unicode()
        self.unicode_to_byte: Dict[str, int] = {v: k for k, v in self.byte_to_unicode.items()}
        # rank 表：pair (tokenA, tokenB) -> merge 顺序；merge 结果 token = tokenA + tokenB
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}   # token 字符串 -> id
        self.id_to_token: List[str] = []  # id -> token 字符串

    # ---------------- 基础工具 ----------------
    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """将 0..255 每个字节映射成一个唯一的可打印 unicode 字符（GPT-2 的方案）。"""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\u00a1"), ord("\u00ac") + 1))
            + list(range(ord("\u00ae"), ord("\u00ff") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))

    @staticmethod
    def _get_pairs(words: List[str]) -> set:
        pairs = set()
        for i in range(len(words) - 1):
            pairs.add((words[i], words[i + 1]))
        return pairs

    def _merge_single(self, words: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
        """把 words 中所有相邻的 pair 替换成 new_token。"""
        out = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and (words[i], words[i + 1]) == pair:
                out.append(new_token)
                i += 2
            else:
                out.append(words[i])
                i += 1
        return out

    # ---------------- 训练 ----------------
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """在给定文本上训练 BPE，使最终词表大小 >= vocab_size（至少 256 字节）。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 初始 token：每个字节映射成的 unicode 字符串
        words: List[str] = [self.byte_to_unicode[b] for b in text.encode("utf-8")]

        # 初始词表 = 256 个字节 token
        self.merges = {}
        self.vocab = {self.byte_to_unicode[b]: b for b in range(256)}
        self.id_to_token = [""] * 256
        for b in range(256):
            tok = self.byte_to_unicode[b]
            self.id_to_token[b] = tok

        num_merges = max(vocab_size, 256) - 256
        for m in range(num_merges):
            pairs = self._get_pairs(words)
            if not pairs:
                break
            # 统计相邻对频率，取最高者
            counts = {}
            for i in range(len(words) - 1):
                p = (words[i], words[i + 1])
                counts[p] = counts.get(p, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            new_token = best[0] + best[1]
            self.merges[best] = m
            words = self._merge_single(words, best, new_token)
            new_id = len(self.id_to_token)
            self.vocab[new_token] = new_id
            self.id_to_token.append(new_token)
            if verbose and (m + 1) % 200 == 0:
                print(f"  merge {m + 1}/{num_merges}: {best!r} -> {new_token!r}")

        if verbose:
            print(f"  训练完成，词表大小 = {len(self.id_to_token)}")

    # ---------------- 编解码 ----------------
    def _bpe_encode(self, words: List[str]) -> List[str]:
        """按 merge rank（从小到大）贪心合并，得到 token 字符串序列。"""
        while len(words) > 1:
            pairs = self._get_pairs(words)
            if not pairs:
                break
            best = None
            best_rank = None
            for p in pairs:
                r = self.merges.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best = p
                    best_rank = r
            if best is None:
                break
            words = self._merge_single(words, best, best[0] + best[1])
        return words

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        ints = [self.byte_to_unicode[b] for b in text.encode("utf-8")]
        tokens = self._bpe_encode(ints)
        return [self.vocab[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        s = "".join(self.id_to_token[i] for i in ids)
        return bytes([self.unicode_to_byte[c] for c in s]).decode("utf-8", errors="replace")

    # ---------------- 属性 / 持久化 ----------------
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "bpe",
            "byte_to_unicode": {str(k): v for k, v in self.byte_to_unicode.items()},
            "unicode_to_byte": {k: v for k, v in self.unicode_to_byte.items()},
            "merges": [[a, b, r] for (a, b), r in self.merges.items()],
            "id_to_token": self.id_to_token,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        if "byte_to_unicode" in data:
            tok.byte_to_unicode = {int(k): v for k, v in data["byte_to_unicode"].items()}
            tok.unicode_to_byte = {v: k for k, v in tok.byte_to_unicode.items()}
        tok.merges = {(a, b): r for a, b, r in data.get("merges", [])}
        tok.id_to_token = list(data["id_to_token"])
        tok.vocab = dict(data["vocab"])
        return tok


if __name__ == "__main__":
    # 简单自测
    txt = "床前明月光，疑是地上霜。深度学习需要数学基础。Hello, world!"
    tok = BPETokenizer()
    tok.train(txt, vocab_size=300)
    ids = tok.encode(txt)
    print("vocab_size:", tok.vocab_size)
    print("ids:", ids)
    print("roundtrip ok:", tok.decode(ids) == txt)
