"""Verify the independent forward oracles and write their checksum manifest.

Run from the repository root after generating all three reference directories.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np

project = Path(__file__).resolve().parents[2]
root = project / 'fixtures/models/stories15m-q8-v1'
pinned = 'efe3626b119ff81033e309c3ba07a770bef2c546'
checkout = project / '.temp/t19/llama.cpp'
if subprocess.check_output(['git','rev-parse','HEAD'], cwd=checkout, text=True).strip() != pinned:
    raise ValueError('Official runtime revision mismatch')
primary = json.loads((root/'numpy/reference.json').read_text())
official = {name:json.loads((root/name/'reference.json').read_text()) for name in ['official-f32','official-q8']}
if official['official-f32']['tokenizer_cases'] != official['official-q8']['tokenizer_cases']:
    raise ValueError('Tokenizer fixtures differ between weight representations')
cache = {}
for line in (project/'.temp/t19/build/CMakeCache.txt').read_text().splitlines():
    if ':' in line and '=' in line and not line.startswith('//'):
        name, value = line.split('=',1)
        cache[name.split(':',1)[0]] = value
comparisons = []
for case in primary['forward_cases']:
    expected = np.fromfile(root/'numpy'/case['logits']['file'],dtype='<f8')
    if expected.size != 32000 or not np.isfinite(expected).all():
        raise ValueError('Invalid primary logits')
    for name, reference in official.items():
        actual_case = next(c for c in reference['forward_cases'] if c['id']==case['id'])
        actual = np.fromfile(root/name/actual_case['logits_file'],dtype='<f4').astype(np.float64)
        maximum = float(np.max(np.abs(actual-expected)))
        rms = float(np.sqrt(np.mean((actual-expected)**2)))
        if name == 'official-f32' and (maximum > 5e-5 or rms > 1e-5):
            raise ValueError('Independent exact-weight logits disagree')
        if case['greedy_tokens'] != actual_case['greedy_tokens']:
            raise ValueError('Greedy sequences differ between references')
        comparisons.append(dict(case=case['id'],oracle=name,maximum_logit_error=maximum,rms_logit_error=rms,
                                greedy_equal=True))
        if name == 'official-f32':
            if len(actual_case['intermediates']) != 15:
                raise ValueError('Missing or duplicate intermediate tensors')
            for tensor in case['intermediates']:
                values=np.fromfile(root/'numpy'/tensor['file'],dtype='<f8')
                target=next(t for t in actual_case['intermediates'] if t['name']==tensor['name'])
                actual_values=np.fromfile(root/name/target['file'],dtype='<f4').astype(np.float64)
                if values.size != 288 or actual_values.size != 288 or np.max(np.abs(values-actual_values)) > 5e-5:
                    raise ValueError('Independent intermediate tensors disagree')
files={}
for directory in ['official-f32','official-q8','numpy']:
    for path in sorted((root/directory).iterdir()):
        files[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest=dict(model='stories15M-q8_0.gguf',model_revision='99dd1a73db5a37100bd4ae633f4cfce6560e1567',
              model_sha256=primary['source_sha256'],model_bytes=26671328,
              original_model_revision='0bd21da7698eaf29a0d7de3992de8a46ef624add',
              original_model_declared_license='MIT',official_runtime_revision=pinned,
              derived_f32_sha256=primary['derived_f32_sha256'],derived_f32_bytes=98357952,
              environment=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                               compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]),
              build_configuration={**{name.lower(): cache['GGML_' + name] == 'ON' for name in
                                      ['NATIVE','AVX','AVX2','FMA','BLAS','OPENMP','CPU_REPACK']},
                                   'build_type': cache['CMAKE_BUILD_TYPE']},
              license_evidence=dict(url='https://huggingface.co/karpathy/tinyllamas/raw/0bd21da7698eaf29a0d7de3992de8a46ef624add/README.md',
                                    sha256='f5d36081d79fc18b2b8ee80fbf05f307c30ea8d6f3ce605c3f393be29f7d8863'),
              generator_sources={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [project/'tools/stories15m_reference.py',project/'tools/model-reference/reference.cpp',
                   project/'tools/model-reference/CMakeLists.txt',Path(__file__).resolve()]},
              comparisons=comparisons,files=files)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Verified {len(files)} fixture files, {len(comparisons)} oracle comparisons, 16 tokenizer cases')
