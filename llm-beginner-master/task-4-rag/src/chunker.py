"""文本切分（chunking）。

- chunk_text：按字符数（chunk_size / overlap 都以字符计，不是词元）做固定大小 + 重叠切分，
  并优先在段落 / 换行处断开，避免在句子中间硬切。
- extract_pdf_text：从 data/kb.pdf 抽取文本。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """把文本切成若干重叠的 chunk。

    参数（均为字符数）：
      chunk_size: 每个 chunk 的目标字符数。
      overlap:    相邻 chunk 的重叠字符数。
    策略：优先按段落（两个换行）切；若段落过长，再按句子（。！？；）切；最后在 chunk_size 处硬切。
    """
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    text = text.strip()
    if not text:
        return []

    # 先按段落拆
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        # 长段落：按句子拆，再逐句攒到 chunk_size
        if len(para) <= chunk_size:
            candidate = para
        else:
            candidate = para
        if len(buffer) + len(candidate) + 1 <= chunk_size:
            buffer = (buffer + "\n" + candidate).strip() if buffer else candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = candidate

    # 收尾
    if buffer:
        chunks.append(buffer)

    # 对仍超过 chunk_size 的 chunk 做滑动窗口硬切
    final: List[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
        else:
            start = 0
            while start < len(c):
                final.append(c[start:start + chunk_size])
                start += chunk_size - overlap
    return [c for c in final if c]


def extract_pdf_text(pdf_path: Path | str) -> str:
    """用 pypdf 抽取 PDF 页文本，按页用换行连接。"""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)


if __name__ == "__main__":
    root_pdf = ROOT / "data" / "kb.pdf"
    if root_pdf.exists():
        t = extract_pdf_text(root_pdf)
        chunks = chunk_text(t, 256, 32)
        print(f"PDF text chars: {len(t)}, chunks: {len(chunks)}, avg len: {sum(map(len, chunks))/len(chunks):.1f}")
    else:
        print("kb.pdf 不存在")
        sample = "这是一段测试文本。" * 400
        chunks = chunk_text(sample, 256, 32)
        print(f"chunks: {len(chunks)}, avg len: {sum(map(len, chunks))/len(chunks):.1f}")
