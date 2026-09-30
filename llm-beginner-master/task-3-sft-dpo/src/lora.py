"""手写 LoRA（Low-Rank Adaptation），不依赖 peft。

思路：给目标 nn.Linear 层旁挂两个低秩矩阵 A, B。
  y = W x + alpha/r * (B (A x))
其中 A 为 (in, r)，B 为 (r, out)，r 远小于 min(in, out)。训练时只更新 A、B，
原始权重 W 冻结（requires_grad=False）。B 初始化为 0、A 用 kaiming，保证一开始不改变输出。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path


class LoRALinear(nn.Linear):
    """在 nn.Linear 上叠加低秩分支的版本。"""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: float,
                 bias: bool = True, base_weight=None, base_bias=None):
        super().__init__(in_features, out_features, bias=bias)
        # 复制原权重，并保持其 dtype（含 bf16/fp16 情况）
        if base_weight is not None:
            self.weight = nn.Parameter(base_weight.clone())
        if bias and base_bias is not None:
            self.bias = nn.Parameter(base_bias.clone())
        # 冻结原权重
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # A: (in, r), B: (r, out)，dtype 与 weight 一致
        self.lora_A = nn.Parameter(torch.zeros(in_features, r, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, dtype=self.weight.dtype))
        # A 用 kaiming 初始化，B 用 0：初始 sigma=alpha/r * B A x = 0
        nn.init.kaiming_uniform_(self.lora_A, a=(5 ** 0.5))
        nn.init.zeros_(self.lora_B)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def forward(self, x):
        out = super().forward(x)  # W x (+bias)
        if self.r > 0:
            delta = (x @ self.lora_A) @ self.lora_B
            out = out + self.scaling * delta
        return out


def inject_lora(model: nn.Module, target_modules, r: int = 8, alpha: float = 16.0) -> nn.Module:
    """给 model 中名字以 target_modules 里任一结尾的 nn.Linear 层注入 LoRA。

    注入后冻结整个基座（所有原参数 requires_grad=False），只保留新注入的 A/B 可训练。
    target_modules: 如 ["q_proj", "v_proj"]。
    """
    # 先冻结所有原参数
    for p in model.parameters():
        p.requires_grad_(False)
    # 再用 LoRALinear 替换目标线性层（其 A/B 默认 requires_grad=True）
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name == t or name.endswith("." + t) for t in target_modules):
            continue
        parent = _get_parent(model, name)
        attr = name.split(".")[-1]
        lora_linear = LoRALinear(module.in_features, module.out_features, r, alpha,
                                 bias=module.bias is not None,
                                 base_weight=module.weight.data,
                                 base_bias=module.bias.data if module.bias is not None else None)
        setattr(parent, attr, lora_linear)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """把 LoRA 合并回原权重：W <- W + scaling * (A @ B)^T，并把 A/B 清零。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                # A: (in,r), B: (r,out) -> A@B: (in,out)，转成 nn.Linear 的 (out,in)
                correction = (module.lora_A.data @ module.lora_B.data).t()
                module.weight.data += module.scaling * correction
                module.lora_A.data.zero_()
                module.lora_B.data.zero_()
    return model


def _get_parent(model: nn.Module, fullname: str) -> nn.Module:
    parts = fullname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


def lora_state_dict(model: nn.Module) -> dict:
    """收集所有 LoRALinear 模块的 A/B 张量（key 用模块全名）。"""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            sd[f"{name}.r"] = torch.tensor([module.r])
            sd[f"{name}.alpha"] = torch.tensor([module.alpha])
            sd[f"{name}.scaling"] = torch.tensor([module.scaling])
    return sd


def save_lora_weights(model: nn.Module, dir_path) -> None:
    import json
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict(model)
    torch.save(sd, str(dir_path / "adapter_weights.pt"))
    # 记录 LoRA 配置（仅供查看 / 复现）
    target_modules = sorted({fullname_target(name) for name, m in model.named_modules()
                             if isinstance(m, LoRALinear)})
    (dir_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": target_modules,
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def fullname_target(name: str) -> str:
    return name.split(".")[-1]


def load_lora_weights(model: nn.Module, dir_path) -> nn.Module:
    from pathlib import Path
    dir_path = Path(dir_path)
    sd = torch.load(str(dir_path / "adapter_weights.pt"), map_location="cpu")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and f"{name}.lora_A" in sd:
            module.lora_A.data.copy_(sd[f"{name}.lora_A"].to(module.lora_A.device))
            module.lora_B.data.copy_(sd[f"{name}.lora_B"].to(module.lora_B.device))
    return model
