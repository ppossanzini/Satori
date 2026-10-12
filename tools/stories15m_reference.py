"""Prepare exact-weight FP64 and F32-model references from the fixed Q8_0 GGUF.

Requires the pinned official gguf-py reader and NumPy only for offline generation.
Usage: script <source.gguf> <official-q8-reference.json> <output-dir> <derived-f32.gguf>
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import gguf

source, official_json, output, derived = map(Path, sys.argv[1:])
expected_hash = '2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
    raise ValueError('Wrong model checksum')
output.mkdir(parents=True, exist_ok=True)
reader = gguf.GGUFReader(str(source))
metadata = {key: field.contents() for key, field in reader.fields.items() if not key.startswith('GGUF.')}
writer = gguf.GGUFWriter(str(derived), 'llama')
for key, field in reader.fields.items():
    if key.startswith('GGUF.') or key in ('general.architecture', 'general.file_type', 'general.quantization_version'):
        continue
    writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None)
writer.add_uint32('general.file_type', 0)
weights = {}
catalogue = []
for tensor in reader.tensors:
    if int(tensor.tensor_type) == 8:
        blocks = tensor.data.view(np.uint8).reshape(-1, 34)
        scale = blocks[:, :2].copy().view('<f2').reshape(-1).astype(np.float32)
        codes = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
        values = (scale[:, None] * codes).reshape(tuple(map(int, tensor.shape[::-1])))
    elif int(tensor.tensor_type) == 0:
        values = np.array(tensor.data, dtype=np.float32)
    else:
        raise ValueError('Unexpected tensor format')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite weights')
    writer.add_tensor(tensor.name, values)
    weights[tensor.name] = values.astype(np.float64)
    catalogue.append(dict(name=tensor.name, shape=tensor.shape.tolist(), type=int(tensor.tensor_type),
                          elements=int(values.size), bytes=int(tensor.n_bytes)))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

layers, width, heads, head_width = 6, 288, 6, 48
epsilon = float(metadata['llama.attention.layer_norm_rms_epsilon'])
frequency = np.power(10000., -np.arange(0, head_width, 2, dtype=np.float64) / head_width)


def norm(x, scale):
    return x / np.sqrt(np.mean(x * x) + epsilon) * scale


def rope(x, position):
    x = x.reshape(heads, head_width)
    c, s = np.cos(position * frequency), np.sin(position * frequency)
    result = np.empty_like(x)
    result[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    result[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return result


def step(token, cache):
    x = weights['token_embd.weight'][token].copy()
    position = len(cache[0][0])
    states = {}
    for layer in range(layers):
        prefix = f'blk.{layer}.'
        normalized = norm(x, weights[prefix + 'attn_norm.weight'])
        q = rope(weights[prefix + 'attn_q.weight'] @ normalized, position)
        k = rope(weights[prefix + 'attn_k.weight'] @ normalized, position)
        v = (weights[prefix + 'attn_v.weight'] @ normalized).reshape(heads, head_width)
        cache[layer][0].append(k)
        cache[layer][1].append(v)
        keys, values = np.array(cache[layer][0]), np.array(cache[layer][1])
        scores = np.einsum('hd,thd->ht', q, keys) / np.sqrt(head_width)
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        attention = np.einsum('ht,thd->hd', probabilities, values).reshape(width)
        attention_output = weights[prefix + 'attn_output.weight'] @ attention
        ffn_input = x + attention_output
        ffn_normalized = norm(ffn_input, weights[prefix + 'ffn_norm.weight'])
        gate = weights[prefix + 'ffn_gate.weight'] @ ffn_normalized
        up = weights[prefix + 'ffn_up.weight'] @ ffn_normalized
        activated = gate / (1 + np.exp(-gate)) * up
        ffn_output = weights[prefix + 'ffn_down.weight'] @ activated
        x = ffn_input + ffn_output
        if layer == 0:
            states.update({'attn_norm-0': normalized, 'Qcur-0': q.reshape(width),
                           'Kcur-0': k.reshape(width), 'Vcur-0': v.reshape(width),
                           'attn_out-0': attention_output, 'ffn_inp-0': ffn_input,
                           'ffn_norm-0': ffn_normalized, 'ffn_out-0': ffn_output})
        states[f'l_out-{layer}'] = x.copy()
    normalized = norm(x, weights['output_norm.weight'])
    states['result_norm'] = normalized
    return weights['output.weight'] @ normalized, states


def save(name, values):
    path = output / name
    values.astype('<f8').tofile(path)
    return dict(file=name, elements=int(values.size), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


records = []
for case in json.loads(official_json.read_text())['forward_cases']:
    cache = [([], []) for _ in range(layers)]
    for token in case['tokens']:
        logits, states = step(token, cache)
    tensors = [dict(name=name, **save(case['id'] + '-' + name + '.f64le', value)) for name, value in states.items()]
    logit_entry = save(case['id'] + '-logits.f64le', logits)
    generated, margins = [], []
    for i in range(12):
        token = int(np.argmax(logits))
        best_two = np.partition(logits, -2)[-2:]
        margins.append(float(best_two.max() - best_two.min()))
        generated.append(token)
        if token == 2:
            break
        logits, _ = step(token, cache)
    records.append(dict(id=case['id'], tokens=case['tokens'], logits=logit_entry, intermediates=tensors,
                        greedy_tokens=generated, greedy_top2_margins=margins))
result = dict(source_sha256=expected_hash, derived_f32_sha256=hashlib.sha256(derived.read_bytes()).hexdigest(),
              numpy_version=np.__version__, accumulation='FP64, exact dequantized Q8_0 weights; no activation quantization',
              epsilon=epsilon, catalogue=catalogue,
              metadata={k:v for k,v in metadata.items() if k not in ('tokenizer.ggml.tokens','tokenizer.ggml.scores','tokenizer.ggml.token_type')},
              vocabulary_entries=len(metadata['tokenizer.ggml.tokens']), forward_cases=records)
(output / 'reference.json').write_text(json.dumps(result, indent=2) + '\n')
print('FP64 references generated:',len(records))
