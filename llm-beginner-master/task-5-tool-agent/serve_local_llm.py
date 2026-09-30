"""启动一个 OpenAI 兼容的本地 LLM 服务（用 transformers 加载 Qwen 指令模型，供 ReActAgent 调用）。

用法：
  python serve_local_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --port 11434
之后 ReActAgent 会通过 http://localhost:11434/v1/chat/completions 访问。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_handler(model, tokenizer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "local-qwen", "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/") == "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = payload.get("messages", [])
            max_new = payload.get("max_tokens", 64)
            temperature = payload.get("temperature", 0.2)
            # 拼接 chat 模板
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, temperature=temperature,
                                     do_sample=temperature > 0, top_p=0.9)
            new_tokens = out[0][inputs.input_ids.size(1):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": text.strip()}}]})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--port", type=int, default=11434)
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    print(f"[server] 模型加载完成：{args.model}  device={model.device}  port={args.port}")

    handler = build_handler(model, tokenizer)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"[server] 监听 http://localhost:{args.port}/v1 ...")
    server.serve_forever()


if __name__ == "__main__":
    main()
