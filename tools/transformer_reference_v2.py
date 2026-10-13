"""Independent float64 oracle for the Qwen2/Qwen3 operators added in DS07.

GQA attention, half-split (NeoX) RoPE with configurable base and rotary dimension, per-head RMS norm and
bias add. Only inputs are rounded to float32; all arithmetic is float64.
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(70207)
f = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)

# --- Inputs ---
q_heads, kv_heads, head_dim, tokens = 4, 2, 8, 3
query = f(rng.normal(size=q_heads * head_dim))
keys = f(rng.normal(size=(tokens, kv_heads * head_dim)))
values = f(rng.normal(size=(tokens, kv_heads * head_dim)))
rope_in_full = f(rng.normal(size=2 * head_dim))
rope_in_partial = f(rng.normal(size=2 * head_dim))
norm_in = f(rng.normal(size=3 * head_dim))
norm_w = f(rng.normal(size=head_dim))
bias_base = f(rng.normal(size=8))
bias_addend = f(rng.normal(size=8))

# --- GQA attention ---
group = q_heads // kv_heads
gqa = np.zeros(q_heads * head_dim)
for h in range(q_heads):
    kv = h // group
    sl = slice(kv * head_dim, (kv + 1) * head_dim)
    scores = keys[:, sl] @ query[h * head_dim:(h + 1) * head_dim] / np.sqrt(head_dim)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    gqa[h * head_dim:(h + 1) * head_dim] = probs @ values[:, sl]


def rope_half(vec, position, base, rotary):
    out = vec.reshape(-1, head_dim).copy()
    half = rotary // 2
    for kk in range(half):
        angle = position / (base ** (2 * kk / rotary))
        left = out[:, kk].copy()
        right = out[:, kk + half].copy()
        out[:, kk] = left * np.cos(angle) - right * np.sin(angle)
        out[:, kk + half] = left * np.sin(angle) + right * np.cos(angle)
    return out


# --- Per-head RMS norm ---
heads = norm_in.reshape(-1, head_dim)
qk_norm = (heads / np.sqrt(np.mean(heads ** 2, axis=1, keepdims=True) + float(np.float32(1e-6))) * norm_w).ravel()

data = dict(
    provenance="NumPy float64 independent formulas; seed 70207",
    numpy=np.__version__,
    gqa=dict(query_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, tokens=tokens,
             query=query.tolist(), keys=keys.ravel().tolist(), values=values.ravel().tolist(),
             output=gqa.tolist()),
    rope_full=dict(head_dim=head_dim, position=5, base=1e6, rotary=8,
                   input=rope_in_full.tolist(), output=rope_half(rope_in_full, 5, 1e6, 8).ravel().tolist()),
    rope_partial=dict(head_dim=head_dim, position=3, base=1e4, rotary=4,
                      input=rope_in_partial.tolist(), output=rope_half(rope_in_partial, 3, 1e4, 4).ravel().tolist()),
    qk_norm=dict(epsilon=1e-6, input=norm_in.tolist(), weights=norm_w.tolist(), output=qk_norm.tolist()),
    bias=dict(base=bias_base.tolist(), bias=bias_addend.tolist(), output=(bias_base + bias_addend).tolist()),
)

Path("fixtures/operators/transformer-v2.json").write_text(json.dumps(data, indent=2) + "\n")
print("wrote fixtures/operators/transformer-v2.json")
