"""Independent FP64 references using a separately supplied official gguf-py reader.

Run with PYTHONPATH pointing to gguf-py at the revision in docs/gguf-subset-v1.md.
Only reference generation uses gguf/numpy; Satori has no runtime dependency on them.
"""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import gguf

model = Path(sys.argv[1])
reader = gguf.GGUFReader(str(model))
records = []
for tensor in reader.tensors:
    if int(tensor.tensor_type) not in (2, 8) or len(tensor.shape) != 2:
        continue
    if tensor.name not in ("blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"):
        continue
    columns, rows = map(int, tensor.shape)
    # Independent byte-position decoding; no Satori packer or kernel involved.
    blocks = tensor.data.view(np.uint8).reshape(-1, 18 if int(tensor.tensor_type) == 2 else 34)
    scales = blocks[:, :2].copy().view('<f2').reshape(-1).astype(np.float64)
    if int(tensor.tensor_type) == 8:
        codes = blocks[:, 2:].copy().view(np.int8).astype(np.float64)
    else:
        packed = blocks[:, 2:]
        codes = np.concatenate((packed & 15, packed >> 4), axis=1).astype(np.float64) - 8
    weights = (scales[:, None] * codes).reshape(rows, columns)
    x = (np.arange(columns, dtype=np.int64) * 17 % 31 - 15).astype(np.float64) / 16
    records.append(dict(name=tensor.name, shape=[columns, rows], type=int(tensor.tensor_type),
                        output=np.sum(weights * x, axis=1).tolist(),
                        sum_abs=np.sum(np.abs(weights * x), axis=1).tolist()))
    if len(records) == 3:
        break
result = dict(model=model.name, sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
              reader_revision='efe3626b119ff81033e309c3ba07a770bef2c546',
              catalogue=[dict(name=t.name, shape=t.shape.tolist(), type=int(t.tensor_type),
                              offset=int(t.data_offset), bytes=int(t.n_bytes)) for t in reader.tensors], tensors=records)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
print(f'{model.name}: {len(records)} reference tensors')
