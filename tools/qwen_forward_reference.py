#!/usr/bin/env python3
"""Independent float64 forward and greedy reference for a dense qwen2/qwen3 GGUF artifact.

A second implementation written from the published architecture description and the GGUF layout, with no
Satori code involved: it parses the container itself, decodes Q8_0/F32 tensors byte by byte, implements the
forward pass in float64 and writes a fixture the C# forward and generator are compared against.

Memory: weight tensors are decoded one at a time to float64 and released after their layer, except the
token embedding, which is needed both as the lookup table and (when tied) as the output projection. This
keeps the peak bounded well below any full-model expansion.

Usage:
  qwen_forward_reference.py <model.gguf> <output.json> [--tokens 16,13,30,5,100,200] [--top 32]
                            [--samples 64] [--greedy 8]
"""
import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

SCALARS = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4),
           6: ('f', 4), 7: ('B', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}


class Container:
    """Minimal read-only GGUF v3 parser (little-endian)."""

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, 'rb')
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        self.metadata = {}
        self.tensors = {}
        self._parse()

    def _u32(self) -> int:
        value = struct.unpack_from('<I', self.map, self.pos)[0]
        self.pos += 4
        return value

    def _u64(self) -> int:
        value = struct.unpack_from('<Q', self.map, self.pos)[0]
        self.pos += 8
        return value

    def _string(self) -> str:
        length = self._u64()
        value = self.map[self.pos:self.pos + length].decode('utf-8')
        self.pos += length
        return value

    def _value(self, kind: int):
        if kind == 8:
            return self._string()
        if kind == 9:
            element = self._u32()
            return [self._value(element) for _ in range(self._u64())]
        fmt, size = SCALARS[kind]
        value = struct.unpack_from('<' + fmt, self.map, self.pos)[0]
        self.pos += size
        return value

    def _parse(self) -> None:
        if self.map[:4] != b'GGUF':
            raise ValueError('not a GGUF file')
        self.pos = 4
        version = self._u32()
        if version != 3:
            raise ValueError(f'unsupported GGUF version {version}')
        tensor_count = self._u64()
        metadata_count = self._u64()
        for _ in range(metadata_count):
            key = self._string()
            self.metadata[key] = self._value(self._u32())
        directory = []
        for _ in range(tensor_count):
            name = self._string()
            rank = self._u32()
            dims = [self._u64() for _ in range(rank)]
            kind = self._u32()
            offset = self._u64()
            directory.append((name, dims, kind, offset))
        alignment = int(self.metadata.get('general.alignment', 32))
        self.data_start = (self.pos + alignment - 1) // alignment * alignment
        for name, dims, kind, offset in directory:
            self.tensors[name] = (dims, kind, self.data_start + offset)

    def raw(self, name: str) -> memoryview:
        dims, kind, offset = self.tensors[name]
        elements = 1
        for d in dims:
            elements *= d
        block, size = {0: (1, 4), 1: (1, 2), 8: (32, 34)}[kind]
        return self.map[offset:offset + elements // block * size]

    def matrix(self, name: str) -> np.ndarray:
        """Rank-2 tensor as float64 with shape (rows, columns); GGUF dim0 is the contiguous column axis."""
        dims, kind, _ = self.tensors[name]
        columns, rows = int(dims[0]), int(dims[1])
        if kind == 8:
            blocks = np.frombuffer(self.raw(name), dtype=np.uint8).reshape(rows, columns // 32, 34)
            scale = blocks[:, :, :2].copy().view('<f2').astype(np.float64).reshape(rows, columns // 32)
            codes = blocks[:, :, 2:].copy().view(np.int8).astype(np.float64).reshape(rows, columns // 32, 32)
            return (scale[:, :, None] * codes).reshape(rows, columns)
        if kind == 0:
            return np.frombuffer(self.raw(name), dtype='<f4').reshape(rows, columns).astype(np.float64)
        raise ValueError(f'unsupported tensor type {kind} for {name}')

    def vector(self, name: str) -> np.ndarray:
        dims, kind, _ = self.tensors[name]
        if kind != 0:
            raise ValueError(f'{name} is not F32')
        return np.frombuffer(self.raw(name), dtype='<f4').astype(np.float64)


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope_half(x, head_dim, position, base, rotary):
    """Half-split NeoX rotation: the pair (k, k + rotary/2) rotates, only the first `rotary` lanes."""
    out = x.reshape(-1, head_dim).copy()
    half = rotary // 2
    for k in range(half):
        angle = position / (base ** (2 * k / rotary))
        left = out[:, k].copy()
        right = out[:, k + half].copy()
        out[:, k] = left * np.cos(angle) - right * np.sin(angle)
        out[:, k + half] = left * np.sin(angle) + right * np.cos(angle)
    return out


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)


class Model:
    """Float64 forward for one dense qwen2/qwen3 container."""

    def __init__(self, path: Path):
        self.path = path
        self.container = Container(path)
        meta = self.container.metadata
        architecture = meta['general.architecture']
        self.architecture = architecture
        self.layers = int(meta[f'{architecture}.block_count'])
        self.embedding = int(meta[f'{architecture}.embedding_length'])
        self.feed_forward = int(meta[f'{architecture}.feed_forward_length'])
        self.query = int(meta[f'{architecture}.attention.head_count'])
        self.kv_heads = int(meta.get(f'{architecture}.attention.head_count_kv', self.query))
        self.eps = float(meta[f'{architecture}.attention.layer_norm_rms_epsilon'])
        self.base = float(meta[f'{architecture}.rope.freq_base'])
        self.head_dim = int(meta.get(f'{architecture}.attention.key_length', self.embedding // self.query))
        self.rotary = int(meta.get(f'{architecture}.rope.dimension_count', self.head_dim))
        self.emb = self.container.matrix('token_embd.weight')
        self.vocabulary = int(meta.get(f'{architecture}.vocab_size', self.emb.shape[0]))
        output_name = 'output.weight' if 'output.weight' in self.container.tensors else 'token_embd.weight'
        self.output = self.container.matrix(output_name) if output_name != 'token_embd.weight' else self.emb
        self.eos = self.container.metadata.get('tokenizer.ggml.eos_token_id')

    def config(self) -> dict:
        return dict(layers=self.layers, embedding=self.embedding, head_count=self.query,
                    head_count_kv=self.kv_heads, head_dim=self.head_dim, feed_forward=self.feed_forward,
                    vocabulary=self.vocabulary, epsilon=self.eps, rope_base=self.base, rotary=self.rotary)

    def logits(self, tokens, stages=None):
        """Float64 logits of the final position. When `stages` is a list, first/last-layer intermediates are
        appended to it."""
        container = self.container
        query, kv_heads, head_dim, rotary = self.query, self.kv_heads, self.head_dim, self.rotary
        group = query // kv_heads
        count = len(tokens)
        last = count - 1

        hidden = self.emb[np.array(tokens)].copy()
        x_norm = np.zeros_like(hidden)
        mask = np.triu(np.ones((count, count)), k=1).astype(bool)

        def record(layer, stage, values):
            if stages is not None:
                stages.append(dict(layer=layer, stage=stage, values=np.asarray(values, dtype=np.float64).ravel().tolist()))

        for layer in range(self.layers):
            track = stages is not None and layer in (0, self.layers - 1)
            prefix = f'blk.{layer}.'
            attn_norm = container.vector(prefix + 'attn_norm.weight')
            ffn_norm = container.vector(prefix + 'ffn_norm.weight')
            wq = container.matrix(prefix + 'attn_q.weight')
            wk = container.matrix(prefix + 'attn_k.weight')
            wv = container.matrix(prefix + 'attn_v.weight')
            wo = container.matrix(prefix + 'attn_output.weight')
            wg = container.matrix(prefix + 'ffn_gate.weight')
            wu = container.matrix(prefix + 'ffn_up.weight')
            wd = container.matrix(prefix + 'ffn_down.weight')
            has_bias = prefix + 'attn_q.bias' in container.tensors
            has_head_norm = prefix + 'attn_q_norm.weight' in container.tensors

            x_norm[:] = rms_norm(hidden, attn_norm, self.eps)
            if track:
                record(layer, 'attn_norm', x_norm[last])
            q = x_norm @ wq.T
            k = x_norm @ wk.T
            v = x_norm @ wv.T
            if has_bias:
                q += container.vector(prefix + 'attn_q.bias')
                k += container.vector(prefix + 'attn_k.bias')
                v += container.vector(prefix + 'attn_v.bias')

            q = q.reshape(count, query, head_dim)
            k = k.reshape(count, kv_heads, head_dim)
            v = v.reshape(count, kv_heads, head_dim)
            if has_head_norm:
                q = rms_norm(q, container.vector(prefix + 'attn_q_norm.weight'), self.eps)
                k = rms_norm(k, container.vector(prefix + 'attn_k_norm.weight'), self.eps)
            for position in range(count):
                q[position] = rope_half(q[position], head_dim, position, self.base, rotary)
                k[position] = rope_half(k[position], head_dim, position, self.base, rotary)
            if track:
                record(layer, 'q', q[last])

            context = np.zeros((count, query, head_dim))
            for head in range(query):
                kv_head = head // group
                scores = q[:, head, :] @ k[:, kv_head, :].T / np.sqrt(head_dim)
                context[:, head, :] = softmax(np.where(mask, -np.inf, scores)) @ v[:, kv_head, :]
            if track:
                record(layer, 'attn_ctx', context[last])

            hidden += context.reshape(count, query * head_dim) @ wo.T
            if track:
                record(layer, 'attn_residual', hidden[last])

            x_norm[:] = rms_norm(hidden, ffn_norm, self.eps)
            gate = x_norm @ wg.T
            activated = (gate / (1 + np.exp(-gate))) * (x_norm @ wu.T)
            if track:
                record(layer, 'ffn_act', activated[last])
            hidden += activated @ wd.T
            if track:
                record(layer, 'out', hidden[last])

            del attn_norm, ffn_norm, wq, wk, wv, wo, wg, wu, wd, q, k, v, context, gate, activated

        x_norm[:] = rms_norm(hidden, container.vector('output_norm.weight'), self.eps)
        record(-1, 'final_norm', x_norm[last])
        return x_norm[last] @ self.output.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tokens', default='16,13,30,5,100,200')
    parser.add_argument('--top', type=int, default=32)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--greedy', type=int, default=8)
    parser.add_argument('--greedy-prompt', default=None,
                        help='comma-separated token ids for the greedy continuation (default: --tokens)')
    args = parser.parse_args()

    tokens = [int(t) for t in args.tokens.split(',')]
    model = Model(args.model)
    stages = []
    logits = model.logits(tokens, stages)

    order = np.argsort(logits)[::-1]
    sample_indices = sorted({int(i) for i in np.linspace(0, model.vocabulary - 1, args.samples).astype(np.int64)})

    # Greedy continuation from a realistic prompt: argmax appended repeatedly, stopping at EOS. The prompt
    # ids come from the official tokenizer (fixtures/tokenizers), independent of the C# generator.
    greedy_prompt = ([int(t) for t in args.greedy_prompt.split(',')]
                     if args.greedy_prompt is not None else list(tokens))
    greedy = []
    eos = int(model.eos) if model.eos is not None else None
    stop = 'length'
    current = list(greedy_prompt)
    for _ in range(args.greedy):
        step = int(np.argmax(model.logits(current)))
        if eos is not None and step == eos:
            stop = 'eos'
            break
        greedy.append(step)
        current.append(step)

    result = dict(
        provenance='independent NumPy float64 implementation of the dense qwen2/qwen3 forward and greedy '
                   'continuation; own GGUF parser and Q8_0 decode, no Satori code',
        model=args.model.name,
        sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        numpy=np.__version__,
        architecture=model.architecture,
        config=model.config(),
        eos_token_id=model.eos,
        tokens=tokens,
        argmax=int(order[0]),
        top=[[int(i), float(logits[i])] for i in order[:args.top]],
        samples=[[i, float(logits[i])] for i in sample_indices],
        logits_sum=float(np.sum(logits)),
        logits_abs_sum=float(np.sum(np.abs(logits))),
        stages=stages,
        greedy=dict(prompt=greedy_prompt, tokens=greedy, eos=eos, stop=stop, max_new=args.greedy),
    )
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{args.model.name}: argmax={result["argmax"]} vocab={model.vocabulary} greedy={greedy} stop={stop}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
