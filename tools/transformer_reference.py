"""Independent float64 operator oracle; only inputs are rounded to float32."""
import json
from pathlib import Path
import numpy as np
rng = np.random.default_rng(20019)
f = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)
x = f(rng.normal(size=96)); w = f(rng.normal(size=96))
g = f([-1000, -80, -2, -0.0, 0, 2, 80, 1000]); u = f(rng.normal(size=8))
q = f(rng.normal(size=96)); k = f(rng.normal(size=(7, 96))); v = f(rng.normal(size=(7, 96)))
rope = x.reshape(2,48).copy()
a = 37 / (10000 ** (np.arange(0,48,2) / 48))
left, right = rope[:,::2].copy(), rope[:,1::2].copy()
rope[:,::2] = left*np.cos(a)-right*np.sin(a)
rope[:,1::2] = left*np.sin(a)+right*np.cos(a)
att = []
for h in range(2):
    sl = slice(h*48,(h+1)*48)
    s = k[:,sl] @ q[sl] / np.sqrt(48)
    p = np.exp(s-s.max()); p /= p.sum()
    att.extend(p @ v[:,sl])
sig = np.exp(-np.logaddexp(0,-g))
data = dict(provenance='NumPy float64 independent formulas; seed 20019', numpy=np.__version__,
    x=x.tolist(), weights=w.tolist(), rms=(x/np.sqrt(np.mean(x*x)+float(np.float32(1e-5)))*w).tolist(),
    rope=rope.ravel().tolist(), gate=g.tolist(), up=u.tolist(), swiglu=(g*sig*u).tolist(),
    query=q.tolist(), keys=k.ravel().tolist(), values=v.ravel().tolist(), attention=att)
Path('fixtures/operators/transformer-v1.json').write_text(json.dumps(data,indent=2)+'\n')
